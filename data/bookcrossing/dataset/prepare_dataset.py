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


def extract_from_db():
    """do some preprocessing on the database
    and write it to MovieLens style dataset files
    """
    conn = mdb.connect('localhost', 'root', 'master', 'books')
    cursor = conn.cursor()

    # ignore implicit ratings
    stmt = """DELETE FROM ratings WHERE `Book-Rating` = 0"""
    cursor.execute(stmt)
    conn.commit()

    # users
    stmt = """SELECT DISTINCT `User-ID` FROM ratings
              WHERE `User-ID` IN
                  (SELECT `User-ID` FROM ratings
                   GROUP BY `User-ID` HAVING COUNT(`User-ID`) >= 5)
            """
    cursor.execute(stmt)
    users = cursor.fetchall()
    users = set([u[0] for u in users])
    print(len(users), 'users')

    # books
    stmt = """SELECT DISTINCT `ISBN` FROM ratings
            WHERE `User-ID` IN (%s)
            AND `ISBN` IN
                (SELECT `ISBN` FROM ratings
                 GROUP BY `ISBN` HAVING COUNT(`ISBN`) >= 20);
            """ % ', '.join([str(u) for u in users])
    cursor.execute(stmt)
    books = cursor.fetchall()
    books = set([u[0] for u in books])
    books = ["'" + b + "'" for b in books]

    stmt = """SELECT `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication` FROM books
            WHERE `ISBN` in (%s)""" % ', '.join([b for b in books])
    cursor.execute(stmt)
    books = cursor.fetchall()
    print(len(books), 'books')

    # ratings
    stmt = """SELECT * FROM ratings
            WHERE `User-ID` in (%s)""" % ', '.join([str(u) for u in users])
    cursor.execute(stmt)
    ratings = cursor.fetchall()
    print(len(ratings), 'ratings')

    # write to files
    with open('users.dat', 'w') as outfile:
        for u in sorted(users):
            outfile.write(str(u) + '\n')

    with open('books_full.dat', 'w') as outfile:
        for i, t, a, y in sorted(books):
            #outfile.write(i + '::' + t + '\n')
            outfile.write(i + '::' + t + ' (' + str(y) + ')::' + a + '\n')

    with open('ratings.dat', 'w') as outfile:
        for u, i, r in ratings:
            outfile.write(str(u) + '::' + i + '::' + str(r) + '\n')


def extract_random_sample(n, exclude_fnames):
    """ extract a random sample from the full data file
    may exclude books from other sampled files (exclude_fnames)"""
    def get_id2line(fname):
        id2line = {}
        with open(fname) as infile:
            for line in infile:
                if not line.strip():
                    continue
                id2line[line.split('::')[0]] = line
        return id2line

    full_id2line = get_id2line('books_full.dat')

    feid2line = {}
    for fe in exclude_fnames:
        id2line = get_id2line(fe)
        if set(feid2line.keys()) & set(id2line.keys()):
            print("Error - shouldn't get here")
        for k, v in id2line.items():
            feid2line[k] = v
    id2line = {k: v for k, v in full_id2line.items() if not k in feid2line}

    ids = random.sample(id2line.keys(), n)
    with io.open('books.dat', 'w', encoding='utf-8') as outfile:
        for i in sorted(ids):
            outfile.write(id2line[i].decode('ISO-8859-1'))


def get_titles():
    # titles = []
    # with open('books.dat') as infile, open('titles_books.txt', 'w') as outfile:
        # for line in infile:
            # t = line.split('::')[1].rsplit('(', 1)[0]
            # outfile.write(t + '\n')

    titlesdb, titlesfiltered = set(), set()
    for line in open('titles_books.txt'):
        titlesfiltered.add(line.strip())
    for line in open('titles_db.txt'):
        titlesdb.add(line.strip())
    td = titlesdb
    tf = titlesfiltered
    pdb.set_trace()


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
            if parts[-1] == '0':
                continue
            user.append(parts[0])
            isbn.append(parts[1])
            rating.append(parts[2])
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

    print('finding duplicates...')
    title_author2isbn = collections.defaultdict(list)
    for ridx, row in df_books.iterrows():
        key = (row['title'].lower(), row['author'].lower())
        title_author2isbn[key].append(row['isbn'])
    duplicates = [sorted(v) for v in title_author2isbn.values() if len(v) > 1]

    print('merging duplicates...')
    to_drop = set()
    for dsidx, ds in enumerate(duplicates):
        print('\r', dsidx+1, '/', len(duplicates), end='')
        isbn_keep = ds[0]
        for d in ds[1:]:
            df_ratings['isbn'].replace(d, isbn_keep, inplace=True)
            to_drop.add(d)
    df_books = df_books[~df_books['isbn'].isin(to_drop)]
    print()

    print('saving...')
    df_ratings[['user', 'rating']] = df_ratings[['user', 'rating']].astype(int)
    df_books['year'] = df_books['year'].astype(int)
    df_ratings.to_pickle('df_ratings.obj')
    df_books.to_pickle('df_books.obj')


def condense_data(user_ratings=5, book_ratings=10):
    df_ratings = pd.read_pickle('df_ratings.obj')
    df_books = pd.read_pickle('df_books.obj')
    valid_isbns = set(df_books['isbn'])
    df_ratings = df_ratings[df_ratings['isbn'].isin(valid_isbns)]

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
            outfile.write(str(row['user']) + '::' + row['isbn'] + '::' +
                          str(row['rating']) + '\n')


def create_database():
        """set up the database scheme (SQLITE)"""
        db_file = 'database_new.db'
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


def populate_database():
    df_books = pd.read_pickle('df_books_condensed.obj')
    db_file = 'database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # get items already in the database
    stmt = 'SELECT id, wp_id FROM books ORDER BY id ASC'
    cursor.execute(stmt)
    response = cursor.fetchall()
    db_ids = set([i[0] for i in response])
    wp_ids = set([i[1] for i in response])
    counter = max(db_ids) if db_ids else 0

    for ridx, row in df_books.iterrows():
        counter += 1
        # if counter < 24:
        #     continue
        print(counter, '/', df_books.shape[0] - len(db_ids), end=' ')
        if row['isbn'] in db_ids:
            continue
        if row['year'] < 1000:
            continue  # year of publication must be present
        print(row['title'], '|', row['author'])
        it = Book(row['title'] + ' (' + str(row['year']), row['isbn'], row['author'])
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


class Item(object):
    # init static members
    preprocessor = make_prep_parser({})
    parser = make_parser()
    html_parser = HTMLParser.HTMLParser()
    url_opener = urllib2.build_opener()
    with io.open('user_agents.txt', encoding='utf-8-sig') as infile:
        url_headers = infile.readlines()
    url_headers = [u.strip('"') for u in url_headers]
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

        book = [k + ' (book)' for k in keys]
        novel = [k + ' (novel)' for k in keys]
        keys.update(set(book), set(novel))
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


DEBUG = False


if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    # extract_from_db()
    # extract_random_sample(60000)
    # get_titles()

    # prepare_data()
    # condense_data()
    # export_data()
    # create_database()
    populate_database()

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
