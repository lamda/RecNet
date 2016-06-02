# -*- coding: utf-8 -*-


"""This filters the IMDb data from a crawl by Tomas Karas
"""


from __future__ import division, unicode_literals, print_function
import atexit
import HTMLParser
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


def retrieve_and_condense():
    # print('getting movie data...')
    # db_connector = DbConnector()
    # stmt = '''SELECT movie_id, title, title2, title3,
    #                  plot, storyline, genre, years
    #           FROM scrape
    #           '''
    # result = db_connector.execute(stmt)
    # df = pd.DataFrame(result)
    #
    # print('cleaning movie data...')
    # df['plot'] = df['plot']
    # df['storyline'] = df['storyline']
    # df.replace('n/a', np.NaN, inplace=True)
    # df.dropna(how='all', subset=['plot', 'storyline'], inplace=True)
    # # df.dropna(how='any', subset=['genre', 'years'], inplace=True)
    # # df.dropna(how='any', subset=['plot', 'storyline', 'genre', 'years'],
    # #           inplace=True)
    # # pdb.set_trace()  # TODO: decide which dropna to use
    # df_titles = df
    #
    # print('getting rating data...')
    # db_connector = DbConnector()
    # stmt = '''SELECT movie_id, user_id, rating FROM ratings'''
    # # stmt = '''SELECT movie_id, user_id, rating FROM ratings_test'''
    # # stmt = '''SELECT movie_id, user_id, rating FROM revs'''
    # result = db_connector.execute(stmt)
    # df_ratings = pd.DataFrame(result)

    # df_titles.to_pickle('df_titles_all.obj')
    # df_ratings.to_pickle('df_ratings_all.obj')

    df_titles = pd.read_pickle('df_titles_all.obj')
    df_ratings = pd.read_pickle('df_ratings_all.obj')

    print('compiling dictionary...')
    ttid2year_last = {row['movie_id']: row['years'][-4:]
                      for ridx, row in df_titles.iterrows()
                      if not isinstance(row['years'], float)}
    # first_year = '2005'  # 27,711
    first_year = '2010'  # 13,460
    ttid2year_last = {k: v for k, v in ttid2year_last.iteritems()
                      if v > first_year}

    print('condensing data')
    user_ratings = 5
    title_ratings = 20
    valid_ids = set(df_titles['movie_id']) & set(ttid2year_last.keys())
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

    # suffix = str(user_ratings) + '_' + str(title_ratings)
    suffix = first_year
    df_ratings.to_pickle('df_ratings_condensed_' + suffix + '.obj')
    df_titles.to_pickle('df_titles_condensed_' + suffix + '.obj')

    print('%d/%d: found %d titles with %d ratings by %d users' %
          (user_ratings, title_ratings, len(titles_to_keep),
           df_ratings.shape[0], df_ratings['user_id'].unique().shape[0]))


def populate_database_titles():
    df = pd.read_pickle('df_titles_condensed_2010.obj')
    df.index = range(0, df.shape[0])
    db_file = '../database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    hparser = HTMLParser.HTMLParser()
    df['years'] = df['years'].replace(np.nan, '0', regex=True)

    for ridx, row in df.iterrows():
        print('\r', ridx+1, '/', df.shape[0], end='')
        stmt = '''INSERT OR REPLACE INTO movies
                  (id, cf_title, original_title, wp_text)
                  VALUES (?, ?, ?, ?)'''
        text = ' '.join(
            row[t] for t in ['title', 'title2', 'title3', 'plot', 'storyline']
            if isinstance(row[t], unicode))
        text = hparser.unescape(striplines(text))
        data = (int(row['movie_id'][2:]), row['title'],
                row['title'] + ' (' + unicode(row['years'] + ')'), text)
        cursor.execute(stmt, data)
        if (ridx % 10000) == 0:
            conn.commit()
    conn.commit()
    print()


def populate_database_genres():
    df = pd.read_pickle('df_titles_condensed_2010.obj')
    df.index = range(0, df.shape[0])
    db_file = '../database_new.db'
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    for stmt in [
        'DELETE FROM categories;',
        'DELETE FROM item_cat;'
    ]:
        cursor.execute(stmt)
        conn.commit()

    db_cat2id = {}
    for ridx, row in df.iterrows():
        print('\r', ridx+1, '/', df.shape[0], end='')
        genres = row['genre'].strip().replace('; ', ';').split(';')
        genres = [g for g in genres if g != '']
        for c in genres:
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
            data = (int(row['movie_id'][2:]), db_cat2id[c])
            cursor.execute(stmt, data)
        if (ridx % 10000) == 0:
            conn.commit()
    conn.commit()
    print()


def export_ratings():
    df = pd.read_pickle('df_ratings_condensed_2010.obj')
    df.index = range(0, df.shape[0])
    with open('ratings.dat', 'w') as outfile:
        for ridx, row in df.iterrows():
            if (ridx % 1000) == 0:
                print('\r', ridx, '/', df.shape[0], end='')
            outfile.write(row['user_id'][2:] + '::' + row['movie_id'][2:] + '::')
            outfile.write(str(row['rating']) + '\n')
    print()


def sample_ratings():
    with open('ratings.dat') as infile,\
            open('ratings_sampled.dat', 'w') as outfile:
        for linecount, line in enumerate(infile):
            outfile.write(line)
            if linecount > 1000:
                break


def sample_ratings_large():
    df_titles = pd.read_pickle('df_titles_condensed.obj')
    df_ratings = pd.read_pickle('df_ratings_condensed.obj')
    df_titles.dropna(how='any', subset=['plot', 'storyline', 'genre', 'years'], inplace=True)
    valid_ids = set(df_titles['movie_id'])
    df_ratings = df_ratings[df_ratings['movie_id'].isin(valid_ids)]

    def condense(df_titles, df_ratings, title_ratings, user_ratings=20):
        valid_ids = set(df_titles['movie_id'])
        df_ratings = df_ratings[df_ratings['movie_id'].isin(valid_ids)]
        old_shape = (0, 0)
        titles_to_keep = 0
        while old_shape != df_ratings.shape:
            print(df_titles.shape)
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

        df_ratings.to_pickle('df_ratings_condensed_2.obj')
        df_titles.to_pickle('df_titles_condensed_2.obj')

    pdb.set_trace()


if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    # retrieve_and_condense()

    create_database()
    populate_database_titles()
    populate_database_genres()
    export_ratings()

    # sample_ratings()
    # sample_ratings_large()

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
