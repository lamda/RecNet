# -*- coding: utf-8 -*-


"""This filters the bookcrossing data
http://www.informatik.uni-freiburg.de/~cziegler/BX/
prerequisite: the data is in the folder /BX-SQL-Dump
excludes all implicit ratings
excludes all users having < 20 ratings
writes the dump to .dat files (same format as in the MovieLens files)
To use this, import the Bookcrossing data set into MySQL first
mysql -u root -p books < BX-Users.sql
mysql -u root -p books < BX-Books.sql
mysql -u root -p books < BX-Book-Ratings.sql
Before doing this, add 'SET autocommit=0;' to the beginnen and 'COMMIT;' to the
end of the files to massively speed up importing.

Should this not work, try importing the CSV dump as follows:
LOAD DATA INFILE "C:\\PhD\\...\\BX-SQL-Dump\\test.csv" IGNORE
INTO TABLE `bx-books`
COLUMNS TERMINATED BY ';'
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(`ISBN`, `Book-Title`, `Book-Author`, `Year-of-Publication`, @dummy, @dummy, @dummy, @dummy);
"""


from __future__ import division, unicode_literals, print_function
import collections
from BeautifulSoup import BeautifulSoup
import urllib
import HTMLParser
import io
import MySQLdb as mdb
import operator
import os
import pandas as pd
import pdb
from pijnu.library.error import IncompleteParse
import random
import re
import sqlite3
import time
import urllib
import urllib2
import xml.etree.cElementTree as etree


from mediawiki_parser.preprocessor import make_parser as make_prep_parser
from mediawiki_parser.text import make_parser


def prepare_data():
    print('getting ratings...')
    user, isbn, rating = [], [], []
    with open('BX-SQL-Dump/BX-Book-Ratings.csv') as infile:
        for line in infile:
            try:
                line = line.encode('utf-8', 'ignore')
            except UnicodeDecodeError:
                # skip ratings with garbage bytes in the ISBN
                continue
            parts = line.strip().split(';')
            parts = [p.strip('"') for p in parts]
            if parts[-1] == '0': # TODO test run
                continue
            user.append(parts[0])
            isbn.append(parts[1])
            rating.append(parts[2])
            # rating.append(1) # TODO test run
    df_ratings = pd.DataFrame(data=zip(user, isbn, rating),
                              columns=['user', 'isbn', 'rating'])

    print('getting books...')
    isbn, title, author, year = [], [], [], []
    with open('BX-SQL-Dump/BX-Books.csv') as infile:
        for line in infile:
            try:
                line = line.rsplit('";"', 4)[0].encode('utf-8', 'ignore')
            except UnicodeDecodeError:
                continue
            parts = line.strip().split('";"')
            parts = [p.strip('"') for p in parts]
            isbn.append(parts[0])
            title.append(parts[1])
            author.append(parts[2])
            year.append(parts[3])
    df_books = pd.DataFrame(data=zip(isbn, title, author, year),
                            columns=['isbn', 'title', 'author', 'year'])

    print('saving...')
    df_ratings[['user', 'rating']] = df_ratings[['user', 'rating']].astype(int)
    df_books['year'] = df_books['year'].astype(int)
    df_ratings.to_pickle('df_ratings.obj')
    df_books.to_pickle('df_books.obj')


def eliminate_duplicates():
    df_ratings = pd.read_pickle('df_ratings.obj')
    df_books = pd.read_pickle('df_books.obj')
    print(df_ratings.shape, df_books.shape)

    # elminate all users with < 5 ratings
    agg = df_ratings.groupby('user').count()
    users_to_keep = set(agg[agg['isbn'] >= 5].index)
    df_ratings = df_ratings[df_ratings['user'].isin(users_to_keep)]

    # eliminate all books with 0 ratings
    isbns = set(df_ratings['isbn'])
    df_books = df_books[df_books['isbn'].isin(isbns)]
    df_books.index = range(0, df_books.shape[0])
    print(df_ratings.shape, df_books.shape)

    # compute Jaccard distances between titles
    # df_books = df_books.iloc[:2500]
    # titles_original = df_books['title'].tolist()
    # authors = df_books['author'].tolist()
    # isbns = df_books['isbn'].tolist()
    # years = df_books['year'].tolist()
    # titles = [frozenset(t.lower().split(' ')) for t in titles_original]
    # titles = [t - {'the', 'a', 'an'} for t in titles]
    # idx2title = {t: i for t, i in enumerate(titles_original)}
    # idx2author = {t: i for t, i in enumerate(authors)}
    # idx2isbn = {t: i for t, i in enumerate(isbns)}
    # idx2year = {t: i for t, i in enumerate(years)}
    #
    # merges = collections.defaultdict(list)
    # for idx1, t1 in enumerate(titles):
    #     print('\r', idx1, '/', len(titles), end=' | ')
    #     for idx2, t2 in enumerate(titles):
    #         if idx2 >= idx1:
    #             continue
    #         jcd = (len(t1 & t2) / len(t1 | t2))
    #         if jcd >= 0.8:
    #             if idx2year[idx1] != idx2year[idx2]:
    #                 continue
    #             merges[idx2isbn[idx1]].append(idx2isbn[idx2])
    #             if 0.8 <= jcd < 1:
    #                 print('%.2f %d %d\n%s (%s)\n%s (%s)\n' %
    #                       (jcd, idx1, idx2, idx2title[idx1], idx2author[idx1],
    #                        idx2title[idx2], idx2author[idx2]))
    # duplicates = [[[k] + v] for k, v in merges.items()]
    # print('\nfound %d duplicates' % len(duplicates))

    # # merge all books with identical titles and authors
    # title_author2isbn = collections.defaultdict(list)
    # print('finding duplicates...')
    # for ridx, row in df_books.iterrows():
    #     print('\r', ridx+1, '/', df_books.shape[0], end='')
    #     key = (row['title'].lower(), row['author'].lower())
    #     title_author2isbn[key].append(row['isbn'])
    # print()
    # duplicates = [sorted(v) for v in title_author2isbn.values() if len(v) > 1]

    # merge all books with identical titles
    # df_books = df_books.iloc[:2500]
    title2isbn = collections.defaultdict(list)
    print('finding duplicates...')
    titles = df_books['title'].tolist()
    titles = [re.sub(r'[\(\),!\.\?\-]', '', t.lower()) for t in titles]
    titles = [frozenset(t.split(' ')) for t in titles]
    stopwords = {'the', 'a', 'an', ' ', 'unabridged', 'paperback', 'hardcover'}
    titles = [t - stopwords for t in titles]
    for ridx, row in df_books.iterrows():
        print('\r', ridx+1, '/', df_books.shape[0], end='')
        key = titles[ridx]
        title2isbn[key].append(row['isbn'])
    print()
    duplicates = [sorted(v) for v in title2isbn.values() if len(v) > 1]

    print('merging duplicates...')
    to_drop = set()
    for dsidx, ds in enumerate(duplicates):
        print('\r', dsidx+1, '/', len(duplicates), end='')
        isbn_keep = ds[0]
        for d in ds[1:]:
            df_ratings['isbn'].replace(d, isbn_keep, inplace=True)
            to_drop.add(d)
    print()
    df_books = df_books[~df_books['isbn'].isin(to_drop)]
    df_ratings.to_pickle('df_ratings_merged.obj')
    df_books.to_pickle('df_books_merged.obj')


def condense_data(user_ratings=5, book_ratings=20):
    df_ratings = pd.read_pickle('df_ratings_merged.obj')
    df_books = pd.read_pickle('df_books_merged.obj')
    df_books = df_books[df_books['year'] > 1500]
    valid_isbns = set(df_books['isbn'])
    df_ratings = df_ratings[df_ratings['isbn'].isin(valid_isbns)]

    old_shape = (0, 0)
    books_to_keep = 0
    while old_shape != df_ratings.shape:
        print(df_ratings.shape)
        old_shape = df_ratings.shape
        agg = df_ratings.groupby('isbn').count()
        books_to_keep = set(agg[agg['user'] > book_ratings].index)

        agg = df_ratings.groupby('user').count()
        users_to_keep = set(agg[agg['isbn'] > user_ratings].index)

        df_ratings = df_ratings[df_ratings['isbn'].isin(books_to_keep)]
        df_ratings = df_ratings[df_ratings['user'].isin(users_to_keep)]
        df_books = df_books[df_books['isbn'].isin(books_to_keep)]

    print('%d/%d: found %d books with %d ratings' %
          (user_ratings, book_ratings, len(books_to_keep), df_ratings.shape[0]))

    df_ratings.to_pickle('df_ratings_condensed.obj')
    df_books.to_pickle('df_books_condensed.obj')


def export_data():
    df_ratings = pd.read_pickle('df_ratings_condensed.obj')
    df_books = pd.read_pickle('df_books_condensed.obj')

    with open('books.dat', 'w') as outfile:
        for ridx, row in df_books.iterrows():
            outfile.write(row['isbn'] + '::' + row['title'] + ' (' +
                          str(row['year']) + ')::' + row['author'] + '\n')

    with open('ratings.dat', 'w') as outfile:
        for ridx, row in df_ratings.iterrows():
            outfile.write(str(row['user']) + '::' + row['isbn'] + '::')
            outfile.write(str(row['rating']) + '\n')


def create_database():
        """set up the database scheme (SQLITE)"""
        db_file = '../database_new.db'
        try:
            os.remove(db_file)
        except OSError:
            pass
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        label = 'books'

        # create item table
        if label == 'movies':
            pkey = 'id INTEGER PRIMARY KEY, '
        else:
            pkey = 'id VARCHAR(13) PRIMARY KEY, '

        create_stmt = """CREATE TABLE """ + label + """ (""" + \
                      pkey + \
                      """original_title TEXT,
                         cf_title TEXT,
                         wp_title TEXT,
                         wp_text TEXT,
                         wp_id INT)"""
        cursor.execute(create_stmt)
        conn.commit()

        # create category table
        cursor.execute(""" PRAGMA foreign_keys = ON;""")
        pkey = 'id INTEGER PRIMARY KEY,'
        create_stmt = """CREATE TABLE categories (""" + \
                      pkey + \
                      """name TEXT)"""
        cursor.execute(create_stmt)
        conn.commit()

        # create item-category relation table
        pkey = 'id INTEGER PRIMARY KEY, '
        if label == 'movies':
            item_id = 'item_id INTEGER, '
        else:
            item_id = 'item_id VARCHAR(13),'
        create_stmt = """CREATE TABLE item_cat (""" + \
                      pkey + \
                      item_id + \
                      """cat_id INTEGER,
                      FOREIGN KEY(item_id) REFERENCES """ + label + \
                      """(id),
                      FOREIGN KEY (cat_id) REFERENCES categories(id))"""
        cursor.execute(create_stmt)
        conn.commit()


def prune_database():
    df_books = pd.read_pickle('df_books_condensed.obj')
    df_ids = set(df_books['isbn'])

    db_file = 'database_new_full.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # get items already in the database
    stmt = 'SELECT id FROM books ORDER BY id ASC'
    cursor.execute(stmt)
    response = cursor.fetchall()
    db_ids = set([i[0] for i in response])

    ids_to_delete = db_ids - (df_ids & db_ids)
    for isbn in ids_to_delete:
        stmt = 'DELETE FROM books WHERE id=?;'
        data = (isbn.strip(),)
        cursor.execute(stmt, data)
    conn.commit()


def populate_database(wp_text=False):
    df_books = pd.read_pickle('df_books_condensed.obj')
    db_file = '../database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # get items already in the database
    stmt = 'SELECT id, wp_id FROM books ORDER BY id ASC'
    cursor.execute(stmt)
    response = cursor.fetchall()
    db_ids = set([i[0] for i in response])
    wp_ids = set([i[1] for i in response])
    df_books.index = range(0, df_books.shape[0])

    max_counter = -1
    for db_id in db_ids:
        idx = df_books[df_books['isbn'] == db_id].index.tolist()[0]
        if idx > max_counter:
            max_counter = idx

    counter = max_counter if max_counter > 0 else 0
    print('starting at id', counter)

    if wp_text:
        for ridx, row in df_books.iloc[counter:].iterrows():
            counter += 1
            print(counter, '/', df_books.shape[0],
                  row['title'], '|', row['author'])
            if row['isbn'] in db_ids:
                print('    already in database')
                continue
            if row['year'] < 1000:
                print('    no year present')
                continue  # year of publication must be present
            it = Book(row['title'] + ' (' + str(row['year']) + ')', row['isbn'],
                      row['author'])
            it.generate_title_candidates()
            it.get_wiki_texts()
            it.select_title()
            if it.wp_id in wp_ids:
                it.wikipedia_text = ''
                print('item already in database')
            # if it.wikipedia_text:
            #     it.categories = it.obtain_categories()
            if it.wikipedia_text:
                it.write_to_database(db_file)
                print('YES -', end='')
                wp_ids.add(it.wp_id)
            else:
                print('NO -', end='')
            print(it.wikipedia_title)
            it.wikipedia_text = ''
            print('----------------')
    else:
        for ridx, row in df_books.iloc[counter:].iterrows():
            print('\r', ridx+1, '/', df_books.shape[0], end='')
            stmt = 'INSERT OR REPLACE INTO books' +\
                   '(id, cf_title, original_title)' +\
                   'VALUES (?, ?, ?)'
            data = (row['isbn'], row['title'],
                    row['title'] + ' (' + str(row['year']) + ')')
            cursor.execute(stmt, data)
            if (ridx % 100) == 0:
                conn.commit()
        conn.commit()


def add_genres():
    db_file = os.path.join('..', 'database_new.db')
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # get items already in the database
    stmt = '''SELECT id, wp_id, original_title, wp_text
              FROM books ORDER BY id ASC'''
    cursor.execute(stmt)
    response = cursor.fetchall()
    df = pd.DataFrame(data=response,
                      columns=['isbn', 'wp_id', 'original_title', 'wp_text'])

    stmt = """SELECT id, name from categories"""
    cursor.execute(stmt)
    db_cat2id = {c[1]: c[0] for c in cursor.fetchall()}

    # get items already in the item_cat database
    stmt = 'SELECT item_id FROM item_cat'
    cursor.execute(stmt)
    response = cursor.fetchall()
    categories_present = set(r[0] for r in response)
    item_count = df.shape[0]
    df = df[~df['isbn'].isin(categories_present)]
    for ridx, row in df.iterrows():
        print(ridx, '/', item_count, row['original_title'])
        if DEBUG:
            t = 1
            print('    DEBUG')
        else:
            t = random.randint(2, 10)
        print('    sleeping for', t, 'seconds')
        time.sleep(t)
        url = u'http://www.goodreads.com/search?q=' + row['isbn']
        try:
            request = urllib2.Request(url)
            # choose a random user agent
            ua = random.choice(Item.url_headers)
            request.add_header('User-agent', ua)
            data = Item.url_opener.open(request).read()
            data = data.decode('utf-8')
        except (urllib2.HTTPError, urllib2.URLError) as e:
            print('    !+!+!+!+!+!+!+!+ URLLIB ERROR !+!+!+!+!+!+!+!+')
            print('    URLError', e)
            pdb.set_trace()
        rexes = [
            r'bookPageGenreLink"\s*href="[^"]+">([^<]+)',
        ]
        re_cat = re.compile('|'.join(rexes))
        cats = [e for e in re.findall(re_cat, data)]
        # remove duplicates from e.g., "A > AB" and "A" both being present
        cats = list(set(cats))
        print('   ', row['original_title'])
        print('   ', cats)
        if not cats:  # mark item to delete from books table
            print('    no cats found for', row['isbn'])
            with open('books_to_delete.txt', 'a') as outfile:
                outfile.write(row['isbn'] + '\n')
        else:
            # write categories to databse
            for c in cats:
                if c not in db_cat2id:
                    # insert category if not yet present
                    stmt = """INSERT INTO categories(id, name) VALUES (?, ?)"""
                    i = len(db_cat2id)
                    data = (i, c)
                    cursor.execute(stmt, data)
                    conn.commit()
                    db_cat2id[c] = i
                # insert item-category relation
                stmt = """INSERT INTO item_cat(item_id, cat_id) VALUES (?, ?)"""
                data = (row['isbn'], db_cat2id[c])
                cursor.execute(stmt, data)
                conn.commit()


def delete_genreless():
    with open('books_to_delete.txt') as infile:
        isbns = infile.readlines()

    db_file = os.path.join('..', 'database_new.db')
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    for isbn in isbns:
        print(isbn)
        stmt = 'DELETE FROM books WHERE id=?;'
        data = (isbn.strip(),)
        cursor.execute(stmt, data)
        conn.commit()


def delete_yearless():
    isbns = [u'0571197639', u'0349101779', u'0099771519', u'0330312367',
             u'0450411435', u'0316639842', u'0099521016', u'0099993805',
             u'0330306839', u'0330262130', u'0330267388', u'0451082028',
             u'0316095133', u'0006480764', u'0140276904', u'0099478110',
             u'0553107003', u'0330282565', u'0553227041', u'0330294008',
             u'0330305735', u'0553100777', u'0439078415', u'0002242591',
             u'0330330276', u'0099479419', u'0099760118', u'0571173004',
             u'0140048332', u'0006548539', u'0330345605', u'0001046438',
             u'0099201410', u'0002558122', u'014026583X', u'0006546684',
             u'0451110129', u'0099288559', u'0440846536', u'059044168X',
             u'0590433180', u'0002243962', u'034068478X', u'0684174693',
             u'0440118697', u'0140118365', u'0099268817', u'0099283417',
             u'0099750813', u'0445002972', u'0006716652', u'0590479865',
             u'0553200674', u'0340128720', u'0425043657', u'0739413317',
             u'0340546727', u'0140037896']
    db_file = '../database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    for isbn in isbns:
        stmt = 'DELETE FROM books WHERE id=?;'
        data = (isbn.strip(),)
        cursor.execute(stmt, data)
    conn.commit()


def add_text():
    db_file = os.path.join('..', 'database_new.db')
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # get items already in the database
    stmt = '''SELECT id, wp_id, original_title, wp_text
              FROM books ORDER BY id ASC'''
    cursor.execute(stmt)
    response = cursor.fetchall()
    df = pd.DataFrame(data=response,
                      columns=['isbn', 'wp_id', 'original_title', 'wp_text'])
    item_count = df.shape[0]
    df = df[pd.isnull(df['wp_text'])]
    for ridx, row in df.iterrows():
        print(ridx+1, '/', item_count, row['original_title'], row['isbn'])
        if DEBUG:
            t = 1
            print('    DEBUG')
        else:
            t = random.randint(2, 10)
        print('    sleeping for', t, 'seconds')
        time.sleep(t)
        url = u'http://www.goodreads.com/search?q=' + row['isbn']
        data = ''
        trials = 0
        while not data:
            try:
                request = urllib2.Request(url)
                # choose a random user agent
                ua = random.choice(Item.url_headers)
                request.add_header('User-agent', ua)
                data = Item.url_opener.open(request).read()
                data = data.decode('utf-8')
            except (urllib2.HTTPError, urllib2.URLError) as e:
                print('    !+!+!+!+!+!+!+!+ URLLIB ERROR !+!+!+!+!+!+!+!+')
                print('    URLError', e)
                if trials > 5:
                    pdb.set_trace()
        re_text = r'<div id="descriptionContainer">(.+?)(?:</div>|<a)'
        text = re.findall(re_text, data, flags=re.DOTALL)[0]
        # remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        text = text.strip('\n ')
        text = text.replace('\n', '')

        # write to database
        stmt = 'UPDATE books SET wp_text = ? WHERE id = ?'
        data = (text, row['isbn'])
        cursor.execute(stmt, data)
        conn.commit()


def add_title_to_text():
    db_file = os.path.join('..', 'database_new.db')
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # get items already in the database
    stmt = '''SELECT id, wp_id, cf_title, wp_text
              FROM books ORDER BY id ASC'''
    cursor.execute(stmt)
    response = cursor.fetchall()
    df = pd.DataFrame(data=response,
                      columns=['isbn', 'wp_id', 'cf_title', 'wp_text'])
    item_count = df.shape[0]
    for ridx, row in df.iterrows():
        print(ridx+1, '/', item_count, row['cf_title'], row['isbn'])

        # write to database
        stmt = 'UPDATE books SET wp_text = ? WHERE id = ?'
        data = (row['wp_text'] + ' ' + row['cf_title'], row['isbn'])
        cursor.execute(stmt, data)
        conn.commit()


def delete_textless():
    pdb.set_trace()


class Item(object):
    # init static members
    preprocessor = make_prep_parser({})
    parser = make_parser()
    html_parser = HTMLParser.HTMLParser()
    url_opener = urllib2.build_opener()
    with io.open('user_agents.txt', encoding='utf-8-sig') as infile:
        url_headers = infile.readlines()
    url_headers = [u.strip('"\n') for u in url_headers]

    # url_headers = ['Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:25.0) Gecko/20100101 Firefox/25.0',
    #                'Mozilla/5.0 (Windows NT 6.1; rv:21.0) Gecko/20130328 Firefox/21.0',
    #                'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1667.0 Safari/537.36',
    #                'Mozilla/5.0 (X11; CrOS i686 4319.74.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.57 Safari/537.36',
    #                'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
    #                'Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 5.0; Trident/4.0; InfoPath.1; SV1; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729; .NET CLR 3.0.04506.30)',
    #                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/537.13+ (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2',
    #                'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_5; ar) AppleWebKit/533.19.4 (KHTML, like Gecko) Version/5.0.3 Safari/533.19.4',
    #                'Opera/9.80 (Windows NT 6.0; U; pl) Presto/2.10.229 Version/11.62',
    #                'Opera/9.80 (Windows NT 5.1; U; zh-tw) Presto/2.8.131 Version/11.10']
    db_cat2id = {}  # holds the list of categories present in the database

    def __init__(self, cf_title):
        self.cf_title = cf_title
        self.original_title = cf_title
        self.wikipedia_title = ''
        self.title_candidates = {}
        self.wikipedia_text = ''
        self.id = -1
        self.wp_id = -1
        self.categories = []

    def get_wiki_text(self, title):
        """download a Wikipedia article for a given title
        and resolve any redirects"""
        t = BeautifulSoup(title, convertEntities=BeautifulSoup.HTML_ENTITIES)
        title = t.contents[0]
        title = title.replace(' ', '_')
        title = urllib.quote(urllib.unquote(title.encode('utf-8')))
        if 'disambig' in title:
            return '', title
        url = 'http://en.wikipedia.org/wiki/Special:Export/' + title
        data = None
        trials = 0
        while True:
            try:
                trials += 1
                request = urllib2.Request(url)
                request.add_header('User-agent',
                                   random.choice(Item.url_headers))
                data = Item.url_opener.open(request).read()
                data = data.decode('utf-8', 'ignore')
                data_l = data.lower()
                if not '<text' in data_l or '{{disambig}}' in data_l or\
                       '{{ disambig }}' in data_l or\
                       '{{ disambiguation }}' in data_l or\
                       '{{disambiguation}}' in data_l:
                    data = ''
                break
            except (urllib2.HTTPError, urllib2.URLError) as e:
                print('!+!+!+!+!+!+!+!+ URLLIB ERROR !+!+!+!+!+!+!+!+')
                print('URLError', e)
                if trials >= 5:  # avoid endless repetition
                    pdb.set_trace()
        print(title)
        if '#redirect' in data.lower() and len(data) < 5000:
            data = Item.html_parser.unescape(data)
            data = data[data.find('<text'):]
            r_pos = data.lower().find('#redirect')
            r_offset = data[r_pos:].find('[[')
            close_pos = data[r_pos:].find(']]')
            link = data[r_pos + r_offset + 2:r_pos + close_pos]
            data, title = self.get_wiki_text(link.encode('utf-8'))
        title = urllib.unquote(title.encode('utf-8')).decode('utf-8')

        return data, title

    def get_wiki_texts(self):
        """download the Wikipedia pages corresponding to the title candidates"""
        new_title_candidates = {}
        for t in self.title_candidates:
            text, title = self.get_wiki_text(t.encode('utf-8'))
            new_title_candidates[title] = text
        self.title_candidates = new_title_candidates

    def strip_text(self, text):
        """strip the Wikipedia article export (XML) from tags and Wiki markup
        return only the plain article text
        """
        root = etree.fromstring(text.encode('utf-8'))
        for child in root[1]:
            if 'export-0.10/}id' in child.tag:
                self.wp_id = int(child.text)
            elif 'export-0.10/}revision' in child.tag:
                for child2 in child:
                    if '/export-0.10/}text' in child2.tag:
                        text = child2.text

        # heuristics to remove parts that are not relevant but hard to parse
        rx = re.compile(r'<!--.*?-->', flags=re.DOTALL)
        text = re.sub(rx, r'', text)
        for headline in ['References', 'External links', 'Further reading']:
            text = text.split('==' + headline + '==')[0]
            text = text.split('== ' + headline + ' ==')[0]
        text = re.sub(r'<ref[^/]*?/>', r'', text)
        rx = re.compile(r'<ref[^<]*?/>|<ref.*?</ref>', flags=re.DOTALL)
        text = re.sub(rx, r'', text)
        rx = re.compile(r'<gallery.*?</gallery>', flags=re.DOTALL)
        text = re.sub(rx, r'', text)
        text = text.replace('<', '')
        text = text.replace('>', '')

        # parse the text from Wikipedia markup to plain text
        trials = 0
        try:
            trials += 1
            preprocessed_text = Item.preprocessor.parse(text + '\n')
            output = Item.parser.parse(preprocessed_text.leaves())
        except (AttributeError, IncompleteParse), e:
            print('!+!+!+!+!+!+!+!+ PARSER ERROR !+!+!+!+!+!+!+!+')
            print(self.wikipedia_title)
            print(e)
            if trials >= 5:  # avoid endless repetition
                pdb.set_trace()
        output = unicode(output).replace('Category:', ' Category: ')
        output = unicode(output).replace('Template:', ' Template: ')
        return output

    def select_title(self, relevant_categories):
        """select a title among the candidates,
        based on the obtained Wikipedia text.
        If several articles exist, choose the one with most relevant categories
        """
        def extract_categories(title, relevant_categories):
            """ extract all categories from a given article"""
            regex = re.compile('\[\[Category:([^#\|\]]+)', flags=re.IGNORECASE)
            categories = ' '.join(regex.findall(self.title_candidates[title]))
            occurrences = 0
            for c in relevant_categories:
                occurrences += categories.lower().count(c)
            return occurrences

        titles = [t for t in self.title_candidates if self.title_candidates[t]]
        if len(titles) == 0:
            self.wikipedia_title = ''
        elif len(titles) == 1:
            self.wikipedia_title = titles[0]
        elif len(titles) > 1:
            categories = {t: extract_categories(t, relevant_categories)
                          for t in titles}
            ranked = sorted(categories.items(),
                            key=operator.itemgetter(1), reverse=True)
            if ranked[0][1] == ranked[1][1]:
                if ranked[1] in ranked[0]:
                    self.wikipedia_title = ranked[0][1]
            self.wikipedia_title = ranked[0][0]
        if self.wikipedia_title:
            self.wikipedia_text = self.title_candidates[self.wikipedia_title]
            self.wikipedia_text = self.strip_text(self.wikipedia_text)
        print('selected', self.wikipedia_title)

    def write_to_database(self, table, db_file):
        """write this object to the database"""
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # create row for category, if required
        if not Item.db_cat2id:
            stmt = """SELECT id, name from categories"""
            cursor.execute(stmt)
            Item.db_cat2id = {c[1]: c[0] for c in cursor.fetchall()}
        for c in self.categories:
            if c not in Item.db_cat2id:
                # insert category if not yet present
                stmt = """INSERT INTO categories(id, name) VALUES (?, ?)"""
                i = len(Item.db_cat2id)
                data = (i, c)
                cursor.execute(stmt, data)
                conn.commit()
                Item.db_cat2id[c] = i
            # insert item-category relation
            stmt = """INSERT INTO item_cat(item_id, cat_id) VALUES (?, ?)"""
            data = (self.id, Item.db_cat2id[c])
            cursor.execute(stmt, data)
            conn.commit()

        # write item
        stmt = """INSERT OR REPLACE INTO """ + table +\
               """(id, wp_id, cf_title, original_title, wp_title, wp_text)
                  VALUES (?, ?, ?, ?, ?, ?)"""
        data = (self.id, self.wp_id, self.cf_title, self.original_title,
                self.wikipedia_title, self.wikipedia_text)
        cursor.execute(stmt, data)
        conn.commit()


class Book(Item):
    def __init__(self, cf_title, bid, author):
        super(Book, self).__init__(cf_title)
        self.id = bid
        self.author = author

    def generate_title_candidates(self):
        """ generate title candidates for books"""
        for c in '{}[]\n.':
            self.cf_title = self.cf_title.replace(c, '')
        self.cf_title = self.cf_title.split(':')[0]
        self.cf_title = self.cf_title.split('(')[0]
        if len(self.cf_title) > 1:
            if self.cf_title[0] != self.cf_title[0].upper() or \
                    self.cf_title[1] != self.cf_title[1].lower():
                self.cf_title = self.cf_title[0].upper() +\
                    self.cf_title[1:].lower()
        ce = BeautifulSoup.HTML_ENTITIES
        self.cf_title = BeautifulSoup(self.cf_title, convertEntities=ce)
        self.cf_title = self.cf_title.contents[0]
        self.cf_title = self.cf_title.replace('reg;', '')
        self.cf_title = self.cf_title.replace(';', '')
        self.cf_title = self.cf_title.replace('(R)', '')
        self.cf_title = self.cf_title.replace('(r)', '')
        keys = {self.cf_title.strip()}

        # handle prefix/suffix swaps, e.g., "Haine, La"
        prefixes = {'The', 'A', 'An', 'La', 'Le', 'Les', 'Die', 'Das', 'Der',
                    'Ein', 'Il', "L'", 'Lo', 'Le', 'I', 'El', 'Los', 'Las', 'O'}
        new_keys = set()
        for k in keys:
            parts = k.split(' ')
            if len(parts) > 1 and parts[0].strip() in prefixes:
                new_keys.add(' '.join(parts[1:]))
        keys |= new_keys

        # add "The" to the beginning, if it is not already there
        new_keys = set()
        for k in keys:
            p = k.split(' ')[0]
            if p not in prefixes:
                new_keys.add('The ' + k)
        keys |= new_keys

        # adapt captialization to the Wikipedia Manual of Style
        # (this is only a heuristic)
        new_keys = set()
        minuscles = {'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for',
                     'yet', 'of', 'to', 'in', 'for', 'on', 'with'}

        for k in keys:
            parts = k.split(' ')
            parts = [p for p in parts if p]
            parts_new = [parts[0]]
            for p in parts[1:]:
                if p.lower() not in minuscles:
                    parts_new.append(p[0].upper() + p[1:])
                else:
                    parts_new.append(p)
            new_keys.add(' '.join(parts_new))
        keys |= new_keys

        author_last = self.author.rsplit(' ', 1)[-1]
        book = [k + ' (' + author_last + ' book)' for k in keys]
        booka = [k + ' (book)' for k in keys]
        novel = [k + ' (novel)' for k in keys]
        novela = [k + ' (' + author_last + ' novel)' for k in keys]
        keys.update(set(book), set(novel), set(booka), set(novela))
        self.title_candidates = {k: '' for k in keys}

    def select_title(self):
        """ select the title among the candidates
        and check if it's actually a book
        """
        super(Book, self).select_title(['books', 'novels', 'plays'])

        # sanity check - is this really a relevant article?
        if self.wikipedia_text:
            regex = re.compile('\[\[Category:([^#\|\]]+)', flags=re.IGNORECASE)
            data = self.title_candidates[self.wikipedia_title]
            categories = ' '.join(regex.findall(data))
            occurrences = categories.lower().count('books')
            occurrences += categories.lower().count('novels')
            occurrences += categories.lower().count('plays')
            occurrences += categories.lower().count('short story')
            if not occurrences:
                self.wikipedia_text = ''
                print('did not pass sanity check')
            if not self.author.split()[-1].lower() in self.wikipedia_text.lower():
                if DEBUG:
                    pdb.set_trace()
                self.wikipedia_text = ''
                print('author not in text')
            del self.title_candidates

    def obtain_categories(self):
        """scrape book categories from Google"""
        # sleep in-between to not get banned for too frequent requests
        if DEBUG:
            t = 1
        else:
            t = random.randint(10, 19)
        print('DEBUG')
        print('sleeping for', t, 'seconds')
        time.sleep(t)
        title = urllib.quote(urllib.unquote(self.wikipedia_title.encode()))
        query = '"' + title.replace('_', '+') + '"+' + 'genre'
        url = u"https://www.google.com/search?hl=en&biw=1195&bih=918" +\
              u"&sclient=psy-ab&q=" + query + u"&btnG=&oq=&gs_l=&pbx=1"
        try:
            request = urllib2.Request(url)
            # choose a random user agent
            ua = random.choice(Item.url_headers)
            request.add_header('User-agent', ua)
            data = Item.url_opener.open(request).read()
            data = data.decode('utf-8')
            if self.author.split()[-1].lower() not in data.lower():  # sanity check
                self.wikipedia_text = ''
                return []
        except (urllib2.HTTPError, urllib2.URLError) as e:
            print('!+!+!+!+!+!+!+!+ URLLIB ERROR !+!+!+!+!+!+!+!+')
            print('URLError', e)
            pdb.set_trace()

        rexes = [
            # r'<span class="kno-a-v">([^</]+)',
            #  r'<span class="answer_slist_item_title nonrich">([^</]+)',
            #  r'<span class="answer_slist_item_title">([^</]+)',
            r'Genres\s*(?:</span>)?(?:</a>)?:\s*(?:</span>)?\s*<span class="[-\_\sa-zA-Z]+">([^</]+)',
            r'Genre</td><td(?:[^</]*)>([^</]+)',
            r'Genre</th></tr><td(?:[^</]*)>([^</]+)',
        ]
        re_cat = re.compile('|'.join(rexes))
        cats = [e for g in re.findall(re_cat, data) for e in g if e]
        # cats = [g for g in re.findall(re_cat, data) if g]
        print(self.wikipedia_title)
        print(cats)
        if DEBUG:
            pdb.set_trace()
        cats = list(set(cats))
        if not cats:  # sanity check
            self.wikipedia_text = ''
        return cats

    def write_to_database(self, db_file):
        super(Book, self).write_to_database('books', db_file)


DEBUG = False  # TODO


if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    # prepare_data()
    # eliminate_duplicates()
    # condense_data(user_ratings=20, book_ratings=5)
    # prune_database()
    # export_data()
    # create_database()
    # populate_database(wp_text=False)
    # add_genres()
    # delete_genreless()
    # delete_yearless()
    # add_text()
    add_title_to_text()
    # delete_textless()

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
