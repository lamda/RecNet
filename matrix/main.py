# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import io
import numpy as np
import os
import pdb
import cPickle as pickle
from scipy import sparse as sparse
import sqlite3


class Matrix(object):
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == 'bookcrossing':
            self.db_main_table = 'books'
        elif dataset in ('movielens', 'imdb'):
            self.db_main_table = 'movies'
        else:
            print('Error - dataset not supported')
            pdb.set_trace()

    def create_matrix(self):
        print('creating utility matrix...')
        # with open('ratings_reviews.pickle', 'rb') as infile:
        #     ratings_reviews = pickle.load(infile)
        # with open('ratings_profiles.pickle', 'rb') as infile:
        #     ratings_profiles = pickle.load(infile)
        # ratings = ratings_reviews + ratings_profiles
        # # format: (movie_id, user_id, rating)

        ratings = []
        fpath = os.path.join('..', 'data', self.dataset, 'dataset', 'ratings.dat')
        with open(fpath) as infile:
            for line in infile:
                user_id, movie_id, rating = line.strip().split('::')[:3]
                ratings.append((int(movie_id), int(user_id), int(rating)))

        users = sorted(set([a[1] for a in ratings]))
        user2matrix = {user: i for user, i in zip(users, range(len(users)))}
        with open('user2matrix.pickle', 'wb') as outfile:
            pickle.dump(user2matrix, outfile, -1)

        ttids = sorted(set([a[0] for a in ratings]))
        ttid2matrix = {ttid: i for ttid, i in zip(ttids, range(len(ttids)))}
        with open('ttid2matrix.pickle', 'wb') as outfile:
            pickle.dump(ttid2matrix, outfile, -1)

        ratings = [(user2matrix[r[1]], ttid2matrix[r[0]], r[2])
                   for r in ratings]
        row_ind = [r[0] for r in ratings]
        col_ind = [r[1] for r in ratings]
        data = [r[2] for r in ratings]
        utility = sparse.csr_matrix((data, (row_ind, col_ind)))
        with open('utility.pickle', 'wb') as outfile:
            pickle.dump(utility, outfile, -1)

    def compute_cosine_sim_1(self):
        print(1)
        # via http://stackoverflow.com/questions/17627219
        with open('utility.pickle', 'rb') as infile:
            A = pickle.load(infile)

        print(2)
        # transpose, as the code below compares rows
        A = A.T

        print(3)
        # base similarity matrix (all dot products)
        similarity = A.dot(A.T)

        print(4)
        # squared magnitude of preference vectors (number of occurrences)
        square_mag = similarity.diagonal()

        print(5)
        # inverse squared magnitude
        inv_square_mag = 1 / square_mag

        print(6)
        # if it doesn't occur, set the inverse magnitude to 0 (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        print(7)
        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        print(8)
        # cosine similarity (elementwise multiply by inverse magnitudes)
        col_ind = range(len(inv_mag))
        row_ind = np.zeros(len(inv_mag))
        inv_mag2 = sparse.csr_matrix((inv_mag, (col_ind, row_ind)))

        print(9)
        cosine = similarity.multiply(inv_mag2)
        for v, l in [
            (cosine, '_cosine'),
            (inv_mag2, '_inv_mag2')
        ]:
            np.save('data' + l, v.data)
            np.save('indices' + l, v.indices)
            np.save('indptr' + l, v.indptr)

    def compute_cosine_sim_2(self):
        data = np.load('data_cosine.npy')
        indices = np.load('indices_cosine.npy')
        indptr = np.load('indptr_cosine.npy')
        cosine = sparse.csr_matrix((data, indices, indptr))

        data = np.load('data_inv_mag2.npy')
        indices = np.load('indices_inv_mag2.npy')
        indptr = np.load('indptr_inv_mag2.npy')
        inv_mag2 = sparse.csr_matrix((data, indices, indptr))

        print(10)
        cosine = cosine.T.multiply(inv_mag2)

        print(11)
        cosine.setdiag(1)

        print(12)
        # pickling doesn't work for some reason --> np.save to the rescue
        np.save('data', cosine.data)
        np.save('indices', cosine.indices)
        np.save('indptr', cosine.indptr)

    def get_top_50(self):
        print(1)
        sims = self.load_sim_matrix()
        print(2)
        sims = sims.tocoo(copy=False)
        print(3)
        data = zip(sims.row, sims.col, sims.data)
        print(4)
        data.sort(key=lambda x: (x[0], x[2]), reverse=True)

        print(5)
        fpath = 'ttid2matrix.pickle'
        with open(fpath, 'rb') as infile:
            ttid2matrix = pickle.load(infile)
        matrix2ttid = {v: k for k, v in ttid2matrix.iteritems()}
        cur_row = -1
        cur_data = []
        fpath = 'top50.txt'
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            for idx, entry in enumerate(data):
                if (idx+1) % 1000 == 0:
                    print(idx+1, '/', len(data), end='\r')
                row, col, val = entry
                if cur_row != row and cur_row != -1:
                    if len(cur_data) >= 100:
                        nbs = ';'.join(unicode(matrix2ttid[v]) for v in cur_data[:50])
                        outfile.write(unicode(matrix2ttid[cur_row]) + u'\t' + nbs + u'\n')
                    cur_data = []
                cur_row = row
                cur_data.append(col)
            if len(cur_data) >= 100:
                nbs = ';'.join(unicode(matrix2ttid[v]) for v in cur_data[:50])
                outfile.write(unicode(matrix2ttid[cur_row]) + u'\t' + nbs + u'\n')

    def get_top_50_alternative(self):
        sims = self.load_sim_matrix()
        nz = sims.getnnz(axis=1)
        nz = np.where(nz >= 100)[0]
        ttid2matrix = self.load_ttid2matrix()
        matrix2ttid = {v: k for k, v in ttid2matrix.iteritems()}

        ofname = 'rating_ids_top50_imdb_' + self.ftype + \
                 ('_sample' if self.use_sample else '') + '.txt'
        fpath = os.path.join(self.graph_folder, ofname)
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            for idx, ridx in enumerate(nz):
                if (idx+1 % 100) == 0:
                    print(idx+1, '/', len(nz), end='\r')
                indices = sims.indices[sims.indptr[ridx]:sims.indptr[ridx+1]]
                data = sims.data[sims.indptr[ridx]:sims.indptr[ridx+1]]
                vals = [(v, r) for v, r in zip(data, indices)]
                nbs = sorted(vals, reverse=True)[:50]
                nbs = ';'.join(matrix2ttid[v[1]] for v in nbs)
                outfile.write(matrix2ttid[ridx] + '\t' + nbs + '\n')

    def load_sim_matrix(self, folder='.'):
        data = np.load(os.path.join(folder, 'data.npy'))
        indices = np.load(os.path.join(folder, 'indices.npy'))
        indptr = np.load(os.path.join(folder, 'indptr.npy'))
        sims = sparse.csr_matrix((data, indices, indptr))
        return sims

    def load_ttid2matrix(self, folder='.'):
        fpath = os.path.join(folder, 'ttid2matrix.pickle')
        with open(fpath, 'rb') as infile:
            ttid2matrix = pickle.load(infile)
        return ttid2matrix

    def resolve_graphs(self):
        print('resolving graphs...')
        fpath = os.path.join('..', 'data', self.dataset, 'database_new.db')
        conn = sqlite3.connect(fpath)
        cursor = conn.cursor()
        query = 'SELECT id, cf_title FROM ' + self.db_main_table
        cursor.execute(query)
        data = cursor.fetchall()
        conn.close()
        id2title = {unicode(d[0]): d[1] for d in data}

        for fname in [
            'top50.txt',
        ]:
            print(fname)
            with io.open(fname, encoding='utf-8') as infile, \
                    io.open(fname + '_resolved.txt', 'w', encoding='utf-8') as outfile:
                for line in infile:
                    left, right = line.strip().split('\t')
                    left = id2title[left]
                    right = [id2title[r] for r in right.split(';')]
                    try:
                        outfile.write(left + u'\t' + u';'.join(right) + u'\n')
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        pdb.set_trace()

if __name__ == '__main__':
        m = Matrix(dataset='imdb')
        m.create_matrix()
        m.compute_cosine_sim_1()
        m.compute_cosine_sim_2()
        m.get_top_50()
        m.resolve_graphs()
