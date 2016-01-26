# -*- coding: utf-8 -*-


"""This filters the IMDb data from a crawl by Tomas Karas
"""


from __future__ import division, unicode_literals, print_function
import atexit
import numpy as np
import os
import pandas as pd
import pdb
import pymysql
import sqlite3


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


def striplines(s):
    return s.replace('\n', ' ')


def populate_database():
    print('getting movie data...')
    db_connector = DbConnector()
    stmt = '''SELECT movie_id, title, title2, title3,
                     plot, storyline, genre, years
              FROM scrape'''
    result = db_connector.execute(stmt)
    df = pd.DataFrame(result)

    print('cleaning movie data...')
    df['plot'] = df['plot'].apply(striplines)
    df['storyline'] = df['storyline'].apply(striplines)
    df.replace('n/a', np.NaN, inplace=True)
    df.dropna(how='all', subset=['plot', 'storyline'], inplace=True)
    df.dropna(how='any', subset=['genre', 'years'], inplace=True)
    df_titles = df

    print('getting rating data...')
    db_connector = DbConnector()
    stmt = '''SELECT movie_id, user_id, rating
              FROM ratings'''
    result = db_connector.execute(stmt)
    df_ratings = pd.DataFrame(result)

    print('condensing data')
    user_ratings = 5
    title_ratings = 20
    valid_ids = set(df_titles['movie_id'])
    df_ratings = df_ratings[df_ratings['movie_id'].isin(valid_ids)]
    old_shape = (0, 0)
    titles_to_keep = 0
    while old_shape != df_ratings.shape:
        print(df_ratings.shape)
        old_shape = df_ratings.shape
        agg = df_ratings.groupby('movie_id').count()
        titles_to_keep = set(agg[agg['user_id'] > title_ratings].index)

        agg = df_ratings.groupby('user_id').count()
        users_to_keep = set(agg[agg['movie_id'] > user_ratings].index)

        df_ratings = df_ratings[df_ratings['movie_id'].isin(titles_to_keep)]
        df_ratings = df_ratings[df_ratings['user_id'].isin(users_to_keep)]
        df_titles = df_titles[df_titles['movie_id'].isin(titles_to_keep)]

    print('%d/%d: found %d titles with %d ratings' %
          (user_ratings, title_ratings, len(titles_to_keep),
           df_ratings.shape[0]))

    df_ratings.to_pickle('df_ratings_condensed.obj')
    df_titles.to_pickle('df_titles_condensed.obj')

if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    # create_database()
    populate_database()

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
