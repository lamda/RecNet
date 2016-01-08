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
import io
import MySQLdb as mdb
import os
import pandas as pd
import pdb
import random
import sqlite3


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

    for ridx, row in df_books.iterrows():
        stmt = '''INSERT OR REPLACE INTO `books`
                  (id, original_title, cf_title)
                  VALUES (?, ?, ?)'''
        data = (row['isbn'], row['title'] + ' (' + str(row['year']) + ')',
                row['title'])
        cursor.execute(stmt, data)
        if (ridx % 100) == 0:
            print('\r', ridx, '/', df_books.shape[0], end='')
            conn.commit()


if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    # extract_from_db()
    # extract_random_sample(60000)
    # get_titles()

    # prepare_data()
    condense_data()
    export_data()
    create_database()
    populate_database()

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
