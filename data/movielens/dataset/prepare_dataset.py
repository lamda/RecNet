# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

import atexit
import os
import pandas as pd
import pdb
import pymysql
import sqlite3


"""This filters the MovieLens 1M data
http://grouplens.org/datasets/movielens/
prerequisite:
    the data is in the folder /ml-1m
    links.csv from the 20M dataset is in /ml-1m
"""


class DbConnector(object):
    def __init__(self):
        self.db_host = '127.0.0.1'
        self.db_connection = pymysql.connect(host=self.db_host,
                                             port=3306,
                                             user='root',
                                             passwd='master',
                                             db='tkaras',
                                             charset='utf8')
        self.db_cursor = self.db_connection.cursor(pymysql.cursors.DictCursor)
        self.db_cursor_nobuff = self.db_connection.cursor(
            pymysql.cursors.SSCursor)
        self.db = 'tkaras'
        atexit.register(self.close)

    def close(self):
        self.db_cursor.close()
        self.db_connection.close()

    def execute(self, _statement, _args=None):
        self.db_cursor.execute(_statement, _args)

        if _statement.lower().startswith("select"):
            return self.db_cursor.fetchall()

    def commit(self):
        self.db_cursor.connection.commit()

    def fetch_cursor(self, _statement, _args):
        self.db_cursor.execute(_statement, _args)
        return self.db_cursor

    def last_id(self):
        return self.db_connection.insert_id()

    def fetch_cursor_nobuff(self, _statement, _args):
        self.db_cursor_nobuff.execute(_statement, _args)
        return self.db_cursor_nobuff


def create_database():
    """set up the database scheme (SQLITE)"""
    db_file = '../database_new.db'
    try:
        os.remove(db_file)
    except OSError:
        pass
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    label = 'movies'

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


def populate_database(wp_text=False):
    db_file = '../database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    df = pd.read_csv('ml-1m/movies.dat', sep='::',
                     names=['id', 'title', 'genres'], encoding='utf-8')
    df_link = pd.read_csv('ml-1m/links.csv', encoding='utf-8')
    df = pd.merge(df, df_link, how='left', left_on='id', right_on='movieId')

    for ridx, row in df.iterrows():
        print('\r', ridx+1, '/', df.shape[0], end='')
        stmt = 'INSERT OR REPLACE INTO movies' +\
               '(id, wp_id, cf_title, original_title)' +\
               'VALUES (?, ?, ?, ?)'
        title = row['title'].decode('utf-8')
        data = (row['id'], row['imdbId'], title.rsplit('(', 1)[0], title)
        cursor.execute(stmt, data)
    conn.commit()

    db_cat2id = {}
    for ridx, row in df.iterrows():
        for c in row['genres'].split('|'):
            if c not in db_cat2id:
                # insert category if not yet present
                stmt = 'INSERT INTO categories(id, name) VALUES (?, ?)'
                i = len(db_cat2id)
                data = (i, c)
                cursor.execute(stmt, data)
                conn.commit()
                db_cat2id[c] = i
            # insert item-category relation
            stmt = 'INSERT INTO item_cat(item_id, cat_id) VALUES (?, ?)'
            data = (row['id'], db_cat2id[c])
            cursor.execute(stmt, data)
    conn.commit()


def add_text():
    db_file = '../database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # get items already in the database
    stmt = 'SELECT id, wp_id, wp_text FROM movies ORDER BY id ASC'
    cursor.execute(stmt)
    response = cursor.fetchall()
    df = pd.DataFrame(data=response, columns=['id', 'imdb_id', 'text'])
    item_count = df.shape[0]
    df = df[pd.isnull(df['text'])]

    db_connector = DbConnector()
    for ridx, row in df.iterrows():
        print(ridx+1, '/', item_count, row['imdb_id'])
        mid = 'tt' + unicode(row['imdb_id']).zfill(7)
        stmt = '''SELECT title, plot, storyline FROM scrape
                  WHERE `movie_id` = "%s"''' % mid
        text = db_connector.execute(stmt)
        if not text:
            continue
        text = ' '.join(text[0].values())

        # write to database
        stmt = 'UPDATE movies SET wp_text = ? WHERE id = ?'
        data = (text, row['id'])
        cursor.execute(stmt, data)
        conn.commit()


def condense_data(user_ratings=5, movie_ratings=20):
    db_file = '../database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    df_ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', encoding='utf-8',
                             names=['user_id', 'movie_id',
                                    'rating', 'timestamp']
                             )
    df_ratings.drop('timestamp', 1, inplace=True)

    # get items already in the database
    stmt = '''SELECT id, cf_title, original_title
              FROM movies ORDER BY id ASC'''
    cursor.execute(stmt)
    response = cursor.fetchall()
    df_movies = pd.DataFrame(data=response,
                             columns=['movie_id', 'cf_title', 'original_title'])

    valid_ids = set(df_movies['movie_id'])
    df_ratings = df_ratings[df_ratings['movie_id'].isin(valid_ids)]

    old_shape = (0, 0)
    movies_to_keep = 0
    while old_shape != df_ratings.shape:
        print(df_ratings.shape)
        old_shape = df_ratings.shape
        agg = df_ratings.groupby('movie_id').count()
        movies_to_keep = set(agg[agg['user_id'] > movie_ratings].index)

        agg = df_ratings.groupby('user_id').count()
        users_to_keep = set(agg[agg['movie_id'] > user_ratings].index)

        df_ratings = df_ratings[df_ratings['movie_id'].isin(movies_to_keep)]
        df_ratings = df_ratings[df_ratings['user_id'].isin(users_to_keep)]
        df_movies = df_movies[df_movies['movie_id'].isin(movies_to_keep)]

    print('%d/%d: found %d movies with %d ratings' %
          (user_ratings, movie_ratings, len(movies_to_keep), df_ratings.shape[0]))

    # delete movies not satisfying these conditions from the database
    print('deleting...')
    to_delete = valid_ids - set(df_movies['movie_id'])
    for movie_id in to_delete:
        stmt = 'DELETE FROM movies WHERE id=?;'
        data = (str(movie_id),)
        cursor.execute(stmt, data)
    conn.commit()

    # export
    print('exporting...')
    with open('ratings.dat', 'w') as outfile:
        for ridx, row in df_ratings.iterrows():
            outfile.write(str(row['user_id']) + str('::'))
            outfile.write(str(row['movie_id']) + str('::'))
            outfile.write(str(row['rating']) + str('\n'))


if __name__ == '__main__':
    # create_database()
    # populate_database()
    add_text()
    ## condense_data()
