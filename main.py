# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import io
import itertools
import numpy as np
import numpy.matlib
np.matlib = numpy.matlib
import cPickle as pickle
import operator
import os
import pandas as pd
import pdb
import scipy.sparse
import scipy.spatial.distance
import sklearn.feature_extraction.text
import sqlite3
import sys

import recsys
import recsys_sparse


pd.set_option('display.width', 1000)
# np.random.seed(2014)
# DEBUG = True
DEBUG = False
DEBUG_SIZE = 255
# DEBUG_SIZE = 750
DATA_BASE_FOLDER = 'data'
NUMBER_OF_RECOMMENDATIONS = [1, 2, 3, 4, 5, 10, 15, 20]
# NUMBER_OF_RECOMMENDATIONS = [10]
FRACTION_OF_DIVERSIFIED_RECOMMENDATIONS = 0.4  # should be 0.4
NUMBER_OF_POTENTIAL_RECOMMENDATIONS = 50  # should be 50


class SimilarityMatrix(object):
    def __init__(self, sims):
        self.sims = sims
        self.sims_argsorted = None

    def get_similar_items(self, c0=0, c1=NUMBER_OF_POTENTIAL_RECOMMENDATIONS):
        """delete diagonal entries from a matrix and return columns c0...c1"""
        if self.sims_argsorted is None:
            zeros = np.zeros((self.sims.shape[0], self.sims.shape[1]-1))
            self.sims_argsorted = zeros
            for index, line in enumerate(self.sims.argsort()):
                line_filtered = np.delete(line, np.where(line == index)[0])
                self.sims_argsorted[index, :] = line_filtered
            self.sims_argsorted = self.sims_argsorted.astype(int)
            # reverse argsort order to get similar items first
            self.sims_argsorted = np.fliplr(self.sims_argsorted)
        return self.sims_argsorted[:, c0:c1]

    def get_top_n(self, n):
        return self.get_similar_items(c1=n)


class RecommendationStrategy(object):
    def __init__(self, similarity_matrix):
        self.sims = similarity_matrix
        self.label = ''

    def get_recommendations(self, n):
        raise NotImplementedError

    def get_top_n_recommendations(self, n):
        return self.sims.get_top_n(n)

    def get_div_rec_basis(self, n, nd):
        """return base recommendations + zero columns for diversification
        n is the number of desired base recommendations
        nd is the number of zero columns to be diversified
        """
        base_recs = self.sims.get_top_n(n - nd)
        # add nd columns to base_recs for the diversified recommendations
        recs = np.zeros((base_recs.shape[0], base_recs.shape[1] + nd))
        recs[:, :n-nd] = base_recs
        return recs


class TopNRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNRecommendationStrategy, self).__init__(similarity_matrix)

    def get_recommendations(self, n):
        return self.get_top_n_recommendations(n).astype(int)


class TopNDivRandomRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivRandomRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = '_div_random'

    def get_recommendations(self, n):
        nd = int(n * FRACTION_OF_DIVERSIFIED_RECOMMENDATIONS)
        recs = self.get_div_rec_basis(n, nd)
        divs = self.sims.get_similar_items(c0=n)
        div_range = range(divs.shape[1])
        r_idx = [np.random.permutation(div_range)
                 for x in range(recs.shape[0])]
        r_idx = np.array(r_idx)[:, :nd]
        for c_idx in range(r_idx.shape[1]):
            div_col = divs[np.arange(r_idx.shape[0]), r_idx.T[c_idx, :]]
            recs[:, n-nd+c_idx] = div_col
        return recs.astype(int)


class TopNDivDiversifyRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivDiversifyRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = '_div_diversify'

    def get_recommendations(self, n):
        nd = int(n * FRACTION_OF_DIVERSIFIED_RECOMMENDATIONS)
        recs = self.get_div_rec_basis(n, nd)
        recs[:, n-nd:] = self.get_diversified_columns(n, nd)
        return recs.astype(int)

    def get_diversified_columns(self, n, nd):
        results = []
        idx2sel = {idx: set(vals[:n-nd])
                   for idx, vals in enumerate(self.sims.sims_argsorted)}
        for div_col_idx in range(nd):
            div_column = np.zeros(self.sims.sims.shape[0], dtype=int)
            for row_idx in range(self.sims.sims.shape[0]):
                node_min, val_min = 50000, 50000
                for col_idx in range(NUMBER_OF_POTENTIAL_RECOMMENDATIONS):
                    try:
                        sims_col_idx = self.sims.sims_argsorted[row_idx, col_idx]
                    except IndexError:
                        pdb.set_trace()
                    if sims_col_idx in idx2sel[row_idx] or\
                                    sims_col_idx == row_idx:
                        continue
                    val = sum(self.sims.sims[sims_col_idx, r]
                              for r in idx2sel[row_idx])
                    if val < val_min:
                        val_min = val
                        node_min = sims_col_idx
                div_column[row_idx] = node_min
            results.append(div_column)
            for didx, dnode in enumerate(div_column):
                idx2sel[didx].add(dnode)
        return np.array(results).T


class TopNDivExpRelRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivExpRelRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = '_div_exprel'

    def get_recommendations(self, n):
        nd = int(n * FRACTION_OF_DIVERSIFIED_RECOMMENDATIONS)
        recs = self.get_div_rec_basis(n, nd)
        recs[:, n - nd:] = self.get_exprel_columns(n, nd)
        return recs.astype(int)

    def get_exprel_columns(self, n, nd):
        results = []
        idx2sel = {idx: set(vals[:n - nd])
                   for idx, vals in enumerate(self.sims.sims_argsorted)}
        for div_col_idx in range(nd):
            div_column = np.zeros(self.sims.sims.shape[0])
            for row_idx in range(self.sims.sims.shape[0]):
                node_max, val_max = -1000, -1000
                neighborhood1 = idx2sel[row_idx]
                n_sets = [idx2sel[i] for i in neighborhood1]
                neighborhood2 = reduce(lambda x, y: x | y, n_sets)
                neighborhood = neighborhood1 | neighborhood2
                vals = []
                for col_idx in range(NUMBER_OF_POTENTIAL_RECOMMENDATIONS):
                    sims_col_idx = self.sims.sims_argsorted[row_idx, col_idx]
                    if sims_col_idx in idx2sel[row_idx] or\
                            sims_col_idx == row_idx:
                        continue
                    rel_nodes = {sims_col_idx} | \
                        set(self.sims.sims_argsorted[sims_col_idx, :n-nd]) -\
                        neighborhood
                    val = sum([self.sims.sims[row_idx, r] for r in rel_nodes])
                    if val > val_max:
                        val_max = val
                        node_max = sims_col_idx
                    vals.append(val)
                if node_max == -1:
                    pdb.set_trace()
                div_column[row_idx] = node_max
            results.append(div_column)
            for didx, dnode in enumerate(div_column):
                idx2sel[didx].add(dnode)
        return np.array(results).T


class TopNPersonalizedRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix, example_users, user_rated,
                 user_predictions):
        super(TopNPersonalizedRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.example_users = example_users
        self.user_rated = user_rated
        self.user_predictions = user_predictions
        self.user_predictions_sorted = [np.argsort(up)[::-1] for up in self.user_predictions]
        self.label = '_personalized'

    def get_recommendations(self, n, user_type, ss=50):
        base_recommendations = self.get_top_n_recommendations(n+ss).astype(int)
        base = [set(l) for l in base_recommendations]

        # do not recommend already rated items
        # base = [l - self.user_rated[user_type] for l in base]

        recs = np.zeros((base_recommendations.shape[0], n))
        for bidx, b in enumerate(base):
            i = 0
            for j in self.user_predictions_sorted[user_type]:
                if j in b:
                    recs[bidx, i] = j
                    i += 1
                if i >= n:
                    break
        return recs

    def get_recommendations_selection_sizes(self, n, user_type):
        recs = []
        print('getting selection sizes...')
        for ss in range(150):
            print('\r   ', ss, end='')
            recs.append(self.get_recommendations(n, user_type, ss=ss))
        print()
        return recs


class TopNPersonalizedMixedRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix, example_users, user_rated,
                 user_predictions):
        super(TopNPersonalizedMixedRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.example_users = example_users
        self.user_rated = user_rated
        self.user_predictions = user_predictions
        self.user_predictions_sorted = [np.argsort(up)[::-1] for up in self.user_predictions]
        self.label = '_personalized_mixed'

    def get_recommendations(self, n, user_type, ss=50):
        base_recommendations = self.get_top_n_recommendations(n+ss).astype(int)
        base = [set(l) for l in base_recommendations]
        recs = np.zeros((base_recommendations.shape[0], n))
        middle = int(n/2)
        for bidx, b in enumerate(base):
            for i in range(middle):
                recs[bidx, i] = base_recommendations[bidx, i]
            i = middle
            for j in self.user_predictions_sorted[user_type]:
                if j in b and j not in recs[bidx, :]:
                    recs[bidx, i] = j
                    i += 1
                if i >= n:
                    break
        return recs

    def get_recommendations_selection_sizes(self, n, user_type):
        recs = []
        print('getting selection sizes...')
        for ss in range(150):
            print('\r   ', ss, end='')
            recs.append(self.get_recommendations(n, user_type, ss=ss))
        print()
        return recs


class Recommender(object):
    def __init__(self, dataset, label, load_cached):
        print(label)
        self.dataset = dataset
        self.label = label
        self.load_cached = load_cached
        self.data_folder = os.path.join(DATA_BASE_FOLDER, self.dataset)
        self.dataset_folder = os.path.join(self.data_folder, 'dataset')
        self.graph_folder = os.path.join(self.data_folder, 'graphs')
        self.recommendation_data_folder = os.path.join(
            self.data_folder,
            'recommendation_data'
        )
        db_file = 'database_new.db'
        self.db_file = os.path.join(self.data_folder, db_file)
        if dataset == 'bookcrossing':
            self.db_main_table = 'books'
        elif dataset in ('movielens', 'imdb'):
            self.db_main_table = 'movies'
        else:
            print('Error - dataset not supported')
            pdb.set_trace()
        if not os.path.exists(self.graph_folder):
            os.makedirs(self.graph_folder)
        if not os.path.exists(self.recommendation_data_folder):
            os.makedirs(self.recommendation_data_folder)

        data = self.query_db(
            'SELECT id, cf_title, wp_title, wp_text, original_title, wp_id '
            'FROM ' + self.db_main_table
        )
        data = [(d[0], d[1], d[2], d[4], d[5], d[3]) for d in data]
        cols = ['dataset_id', 'cf_title', 'wp_title', 'original_title',
                'wp_id', 'wp_text']
        self.df = pd.DataFrame(data=data, columns=cols)
        self.df['dataset_id'] = self.df['dataset_id'].apply(
            lambda i: unicode(i))
        self.id2title = {
            t[0]: t[1] for t in zip(self.df.index, self.df['original_title'])
        }
        self.title2id = {v: k for k, v in self.id2title.items()}
        ttids = self.df['dataset_id']
        if DATASET in ['movielens', 'imdb']:
            ttids = map(int, ttids)
        ttids = sorted(ttids)
        self.id2dataset_id = {
            ttid: unicode(i) for ttid, i in zip(range(len(ttids)), ttids)
        }
        if DEBUG:
            self.df = self.df.iloc[:DEBUG_SIZE]
        self.similarity_matrix = None

    def query_db(self, query):
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute(query)
        if query.startswith('SELECT'):
            data = cursor.fetchall()
            conn.close()
            if len(data[0]) == 1:
                data = [d[0] for d in data]
            return data
        else:
            conn.close()

    def save_graph(self, recs, label, n, selection_size=False):
        if selection_size:
            folder = os.path.join(self.graph_folder, 'selection_sizes')
            if not os.path.exists(folder):
                os.makedirs(folder)
        else:
            folder = self.graph_folder
        file_name = os.path.join(
            folder,
            self.label + '_' + unicode(n) + label + GRAPH_SUFFIX
        )
        with io.open(file_name + '.txt', 'w', encoding='utf-8') as outfile:
            for ridx, rec in enumerate(recs):
                for r in rec:
                    outfile.write(self.id2dataset_id[ridx] + '\t' +
                                  unicode(self.id2dataset_id[r]) + '\n')

        with io.open(file_name + '_resolved.txt', 'w', encoding='utf-8')\
                as outfile:
            for ridx, rec in enumerate(recs):
                for r in rec:
                    outfile.write(self.id2title[ridx] + '\t' +
                                  self.id2title[r] + '\n')

    def get_similarity_matrix(self):
        raise NotImplementedError

    def get_recommendations(self):
        strategies = [
            TopNRecommendationStrategy,
            # TopNDivRandomRecommendationStrategy,
            # TopNDivDiversifyRecommendationStrategy,
            # TopNDivExpRelRecommendationStrategy,
        ]

        for strategy in strategies:
            s = strategy(self.similarity_matrix)
            print(s.label)
            for n in NUMBER_OF_RECOMMENDATIONS:
                print('   ', n)
                recs = s.get_recommendations(n=n)
                self.save_graph(recs, label=s.label, n=n)

    def save_recommendation_data(self, obj, label):
        class_name = str(self.__class__).strip("<>'").rsplit('.', 1)[-1]
        fname = os.path.join(self.recommendation_data_folder,
                             class_name + '_' + label + '.obj')
        # with open(fname, 'wb') as outfile:
        #     pickle.dump(obj, outfile, -1)
        np.save(fname, obj)

    def load_recommendation_data(self, label):
        class_name = str(self.__class__).strip("<>'").rsplit('.', 1)[-1]
        fname = os.path.join(self.recommendation_data_folder,
                             class_name + '_' + label + '.obj')
        obj = np.load(fname + '.npy')
        if not obj.shape:
            obj = obj.item()
        return obj


class RatingBasedRecommender(Recommender):
    def __init__(self, dataset, label='rb', load_cached=False, sparse=False):
        super(RatingBasedRecommender, self).__init__(
            dataset, label, load_cached
        )
        self.sparse = sparse
        self.user_types = [
            'min',
            'median',
            'max'
        ]
        self.example_users = self.get_example_users()
        self.user_rated = []
        self.user_predictions = []

    def get_recommendations(self):
        m = self.get_utility_matrix(centered=False)
        m = m.astype(float)
        m[m == 0] = np.nan
        um = recsys.UtilityMatrix(m)
        if self.dataset == 'movielens':
            k = 25
        elif self.dataset == 'bookcrossing':
            k = 25
        elif self.dataset == 'imdb':
            k = 20

        self.similarity_matrix = SimilarityMatrix(um.s_r)
        cfnn = recsys.CFNN(um, k=k)
        # self.user_predictions = [[] for u in self.example_users]
        # for idx, user_type in enumerate(self.user_types):
        #     u = self.example_users[idx]
        #     for i in range(m.shape[1]):
        #         p = cfnn.predict(u, i)
        #         if np.isfinite(p):
        #             self.user_predictions[idx].append(p)
        #         else:
        #             self.user_predictions[idx].append(-1)
        # self.user_rated = [set(np.where(~np.isnan(m[u, :]))[0])
        #                    for u in self.example_users
        # ]

        super(RatingBasedRecommender, self).get_recommendations()
        # s = TopNPersonalizedRecommendationStrategy(
        #     self.similarity_matrix,
        #     self.example_users,
        #     self.user_rated,
        #     self.user_predictions
        # )
        #
        # print(s.label)
        # for n in NUMBER_OF_RECOMMENDATIONS:
        #     print('   ', n)
        #     for idx, user_type in enumerate(self.user_types):
        #         recs = s.get_recommendations(n=n, user_type=idx)
        #         self.save_graph(recs, label=s.label + '_' + user_type, n=n)

    def get_utility_matrix(self, centered=False, load_cached=False):
        if self.load_cached or load_cached:
            str_sparse = '_sparse' if self.sparse else ''
            str_centered = '_centered' if centered else ''
            um = self.load_recommendation_data('um' + str_sparse + str_centered)
            return um

        path_ratings = os.path.join(self.dataset_folder, 'ratings.dat')

        # # load user ids
        # item_ids = set(map(str, self.df['dataset_id']))
        # item2matrix = {m: i for i, m in enumerate(self.df['dataset_id'])}
        # user_ids = set()
        # with io.open(path_ratings, encoding='latin-1') as infile:
        #     for line in infile:
        #         user, item = line.split('::')[:2]
        #         if item in item_ids:
        #             user_ids.add(int(user))
        #
        # user2matrix = {u: i for i, u in enumerate(sorted(user_ids))}
        # um = np.zeros((len(user_ids), len(item_ids)), dtype=np.int8)
        #
        # # load ratings
        # with io.open(path_ratings, encoding='latin-1') as infile:
        #     for line in infile:
        #         user, item, rat = line.split('::')[:3]
        #         user = int(user)
        #         rat = float(rat)
        #         if user in user_ids and item in item_ids:
        #             um[user2matrix[user], item2matrix[item]] = rat

        ratings = []
        with open(path_ratings) as infile:
            for line in infile:
                user_id, movie_id, rating = line.strip().split('::')[:3]
                if DATASET in ['movielens', 'imdb']:
                    ratings.append((int(movie_id), int(user_id), int(rating)))
                else:
                    ratings.append((movie_id, int(user_id), int(rating)))

        present_ids = set(self.df['dataset_id'])
        ratings = [t for t in ratings if str(t[0]) in present_ids]

        users = sorted(set([a[1] for a in ratings]))
        user2matrix = {user: i for user, i in zip(users, range(len(users)))}

        ttids = sorted(set([a[0] for a in ratings]))
        ttid2matrix = {ttid: i for ttid, i in zip(ttids, range(len(ttids)))}

        ratings = [(user2matrix[r[1]], ttid2matrix[r[0]], r[2])
                   for r in ratings]
        row_ind = [r[0] for r in ratings]
        col_ind = [r[1] for r in ratings]
        data = [r[2] for r in ratings]

        utility = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), dtype='int32')
        self.save_recommendation_data(utility, 'um_sparse')
        um = utility.toarray()
        self.save_recommendation_data(um, 'um')

        # center by subtracting the average ratings for items
        um_centered = um.astype(np.float32)
        um_centered[np.where(um_centered == 0)] = np.nan
        um_centered = um_centered - np.nanmean(um_centered, axis=0)[np.newaxis, :]
        um_centered[np.where(np.isnan(um_centered))] = 0
        utility_centered = scipy.sparse.csr_matrix(um_centered)
        self.save_recommendation_data(um_centered, 'um_centered')
        self.save_recommendation_data(utility_centered, 'um_centered_sparse')

        str_sparse = '_sparse' if self.sparse else ''
        str_centered = '_centered' if centered else ''
        um = self.load_recommendation_data('um' + str_sparse + str_centered)
        return um

    def get_similarity_matrix(self):
        if self.load_cached:
            data = self.load_recommendation_data('cosine-data')
            indices = self.load_recommendation_data('cosine-indices')
            indptr = self.load_recommendation_data('cosine-indptr')
            cosine = scipy.sparse.csr_matrix((data, indices, indptr))
            return SimilarityMatrix(cosine.toarray())
        um = self.get_utility_matrix(centered=True)

        print('computing similarities...')
        # transpose M because pdist calculates similarities between rows
        # similarity = scipy.spatial.distance.pdist(um_centered.T, 'cosine')

        # NEWEST AND FASTEST VERSION BELOW
        A = scipy.sparse.csr_matrix(um)

        print(1)
        # transpose, as the code below compares rows
        A = A.T

        print(2)
        # base similarity matrix (all dot products)
        similarity = A.dot(A.T)

        print(3)
        # squared magnitude of preference vectors (number of occurrences)
        square_mag = similarity.diagonal()

        print(4)
        # inverse squared magnitude
        inv_square_mag = 1 / square_mag

        print(5)
        # if it doesn't occur, set the inverse magnitude to 0 (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        print(6)
        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        print(7)
        # cosine similarity (elementwise multiply by inverse magnitudes)
        col_ind = range(len(inv_mag))
        row_ind = np.zeros(len(inv_mag))
        inv_mag2 = scipy.sparse.csr_matrix((inv_mag, (col_ind, row_ind)))

        print(8)
        cosine = similarity.multiply(inv_mag2)

        print(9)
        cosine = cosine.T.multiply(inv_mag2)

        print(10)
        cosine.setdiag(0)

        self.save_recommendation_data(cosine.data, 'cosine-data')
        self.save_recommendation_data(cosine.indices, 'cosine-indices')
        self.save_recommendation_data(cosine.indptr, 'cosine-indptr')
        return SimilarityMatrix(cosine.toarray())

    def get_example_users(self):
        um = self.get_utility_matrix()
        ratings = np.ravel(um.sum(axis=1))
        users = [
            np.where(ratings == min(ratings))[0][0],
            np.where(ratings == np.median(ratings))[0][0],
            np.where(ratings == max(ratings))[0][0],
        ]
        return users


class MatrixFactorizationRecommender(RatingBasedRecommender):
    def __init__(self, dataset, load_cached=False, sparse=False):
        super(MatrixFactorizationRecommender, self).__init__(
            dataset, 'rbmf', load_cached, sparse
        )

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

        s = TopNPersonalizedRecommendationStrategy(
            self.similarity_matrix,
            self.example_users,
            self.user_rated,
            self.user_predictions
        )

        sm = TopNPersonalizedMixedRecommendationStrategy(
            self.similarity_matrix,
            self.example_users,
            self.user_rated,
            self.user_predictions
        )

        print(s.label)
        for n in NUMBER_OF_RECOMMENDATIONS:
            print('   ', n)
            for idx, user_type in enumerate(self.user_types):
                recs = s.get_recommendations(n=n, user_type=idx)
                self.save_graph(recs, label=s.label + '_' + user_type, n=n)

        print(sm.label)
        for n in NUMBER_OF_RECOMMENDATIONS:
            print('   ', n)
            for idx, user_type in enumerate(self.user_types):
                recs = sm.get_recommendations(n=n, user_type=idx)
                self.save_graph(recs, label=sm.label + '_' + user_type, n=n)

        ss_n = 10
        for idx, user_type in enumerate(self.user_types):
            ss_recs = s.get_recommendations_selection_sizes(n=ss_n, user_type=idx)
            for ridx, recs in enumerate(ss_recs):
                label = s.label + '_' + user_type + '_ss_' + str(ridx)
                self.save_graph(recs, label=label, n=ss_n, selection_size=True)

        for idx, user_type in enumerate(self.user_types):
            ss_recs = sm.get_recommendations_selection_sizes(n=ss_n, user_type=idx)
            for ridx, recs in enumerate(ss_recs):
                label = sm.label + '_' + user_type + '_ss_' + str(ridx)
                self.save_graph(recs, label=label, n=ss_n, selection_size=True)

    def get_similarity_matrix(self):
        if self.load_cached:
            sim_mat = self.load_recommendation_data('sim_mat')
            self.user_rated, self.user_predictions =\
                self.load_recommendation_data('mf_predictions')
            return sim_mat
        print('loading utility matrix...')
        um = self.get_utility_matrix(centered=False, load_cached=True)

        print('factorizing...')
        q = self.factorize(um)

        # use the centered version for similarity computation
        q_centered = q.astype(float)
        q_centered[np.where(q_centered == 0)] = np.nan
        q_centered = q_centered - np.nanmean(q_centered, axis=0)[np.newaxis, :]
        q_centered[np.where(np.isnan(q_centered))] = 0

        # transpose M because pdist calculates similarities between rows
        # similarity = scipy.spatial.distance.pdist(q.T, 'correlation')
        similarity = scipy.spatial.distance.pdist(q_centered, 'cosine')

        # correlation is undefined for zero vectors --> set it to the max
        # max distance is 2 because the pearson correlation runs from -1...+1
        similarity[np.isnan(similarity)] = 2.0  # for correlation
        # similarity[np.isnan(similarity)] = 1.0  # for cosine
        similarity = scipy.spatial.distance.squareform(similarity)

        sim_mat = SimilarityMatrix(1 - similarity)
        self.save_recommendation_data(sim_mat, 'sim_mat')
        return sim_mat

    def factorize(self, m):
        if self.load_cached:
            f_q = self.load_recommendation_data('mf_data')
            self.user_rated, self.user_predictions =\
                self.load_recommendation_data('mf_predictions')
            return f_q

        # k should be smaller than #users and #items (2-300?)
        if self.sparse:
            um = recsys_sparse.UtilityMatrix(m, similarities=False)
        else:
            m = m.astype(float)
            m[m == 0] = np.nan
            um = recsys.UtilityMatrix(m)

        if self.dataset == 'movielens':
            # for MovieLens:
            #     k=15, nsteps=500, eta_type='bold_driver', regularize=True,
            #     eta=0.00001, init='random'
            kwargs = {
                'k': 15,
                'eta': 0.00001,
                'eta_type': 'bold_driver',
                'init': 'random',
                'regularize': True,
                'nsteps': 1000
            }

        elif self.dataset == 'bookcrossing':
            # for BookCrossing:
            #       k=5, nsteps=500, eta_type='increasing', regularize=True,
            #       eta=0.00001, init='random'
            kwargs = {
                'k': 30,
                'eta': 0.00001,
                'eta_type': 'bold_driver',
                'init': 'random',
                'regularize': True,
                'nsteps': 500
            }

        elif self.dataset == 'imdb':
            #     k=15, nsteps=500, eta_type='bold_driver', regularize=True,
            #     eta=0.00001, init='random'
            kwargs = {
                'k': 10,
                'eta': 0.000001,
                'eta_type': 'bold_driver',
                'init': 'random',
                'regularize': True,
                'nsteps': 1000
            }

        if self.sparse:
            f = recsys_sparse.Factors(um, **kwargs)
        else:
            f = recsys.Factors(um, **kwargs)

        self.user_predictions = [[] for u in self.example_users]
        for idx, user_type in enumerate(self.user_types):
            u = self.example_users[idx]
            for i in range(m.shape[1]):
                p = f.predict(u, i)
                if np.isfinite(p):
                    self.user_predictions[idx].append(p)
                else:
                    self.user_predictions[idx].append(-1)
        if self.sparse:
            self.user_rated = [set(np.where(~np.isnan(m[u, :].toarray()))[0])
                               for u in self.example_users
            ]
        else:
            self.user_rated = [set(np.where(~np.isnan(m[u, :]))[0])
                               for u in self.example_users
            ]
        self.save_recommendation_data(f.q, 'mf_data')
        self.save_recommendation_data(
            [self.user_rated, self.user_predictions],
            'mf_predictions'
        )
        return f.q


class InterpolationWeightRecommender(RatingBasedRecommender):
    def __init__(self, dataset, load_cached=False, sparse=False):
        super(InterpolationWeightRecommender, self).__init__(
            dataset, 'rbiw', load_cached, sparse
        )

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

        s = TopNPersonalizedRecommendationStrategy(
            self.similarity_matrix,
            self.example_users,
            self.user_rated,
            self.user_predictions
        )

        sm = TopNPersonalizedMixedRecommendationStrategy(
            self.similarity_matrix,
            self.example_users,
            self.user_rated,
            self.user_predictions
        )

        print(s.label)
        for n in NUMBER_OF_RECOMMENDATIONS:
            print('   ', n)
            for idx, user_type in enumerate(self.user_types):
                recs = s.get_recommendations(n=n, user_type=idx)
                self.save_graph(recs, label=s.label + '_' + user_type, n=n)

        print(sm.label)
        for n in NUMBER_OF_RECOMMENDATIONS:
            print('   ', n)
            for idx, user_type in enumerate(self.user_types):
                recs = sm.get_recommendations(n=n, user_type=idx)
                self.save_graph(recs, label=sm.label + '_' + user_type, n=n)

        ss_n = 10
        for idx, user_type in enumerate(self.user_types):
            ss_recs = s.get_recommendations_selection_sizes(n=ss_n, user_type=idx)
            for ridx, recs in enumerate(ss_recs):
                label = s.label + '_' + user_type + '_ss_' + str(ridx)
                self.save_graph(recs, label=label, n=ss_n, selection_size=True)

        for idx, user_type in enumerate(self.user_types):
            ss_recs = sm.get_recommendations_selection_sizes(n=ss_n, user_type=idx)
            for ridx, recs in enumerate(ss_recs):
                label = sm.label + '_' + user_type + '_ss_' + str(ridx)
                self.save_graph(recs, label=label, n=ss_n, selection_size=True)

    def get_coratings_all(self, um, mid, w):
        d = collections.defaultdict(int)
        for line in um:
            if line[mid] != 0:
                ratings = [r for r in np.nonzero(line)[0] if r != mid]
                for r in ratings:
                    d[r] += 1
        indices = np.arange(0, 3640)
        coratings = [d[i] for i in indices]
        titles = [self.id2title[idx] for idx in indices]
        similarities = [w[mid, i] for i in indices]
        df = pd.DataFrame(index=indices,
                          data=zip(titles, coratings, similarities),
                          columns=['title', 'coratings', 'similarity'])
        return df

    def get_coratings(self, mid, w, k, coratings_top_10):
        indices = np.arange(0, len(coratings_top_10))
        coratings = [coratings_top_10[mid][i] for i in indices]
        titles = [self.id2title[idx] for idx in indices]
        similarities = [w[mid, i] for i in indices]
        num_ratings = sum(coratings_top_10[mid].values())
        frac_coratings = [x / num_ratings for x in coratings]
        df = pd.DataFrame(index=indices,
                          data=zip(titles, coratings, frac_coratings, similarities),
                          columns=['title', 'coratings', 'frac_coratings', 'similarity'])
        return df

    def debug_cr(self, m, w, mid, top_n=20):
        print(self.id2title[mid])
        for item in np.argsort(1 - w[mid, :])[:top_n]:
            print('    %.3f %d %s %d' % (w[mid, item], m.coratings_r[mid, item], self.id2title[item], item))

    def get_similarity_matrix(self):
        w, k, beta, m = self.get_interpolation_weights()
        if self.load_cached:
            sim_mat = self.load_recommendation_data('sim_mat')
            return sim_mat

        # # DEBUG
        # w, k, beta, m, self.user_ratings = self.load_recommendation_data('iw_data')
        # """
        # 0   Toy Story (1995)
        # 2898 Toy Story 2 (1999)
        # 574 Aladding (1992)
        #
        # 802 Godfather, The (1972)
        # 1182 Akira (1988)
        # 2651 American Beauty (1999)
        #
        # """
        # pdb.set_trace()
        #  hugo = recsys_sparse.WeightedCFNNBiased(m=m, eta_type='bold_driver', k=k, eta=0.00001, regularize=True, init='random', nsteps=50, beta=beta)
        # hugo.w = w
        # hugo.test_error()
        #/DEBUG

        # df = self.get_coratings(mid=0, w=w, k=10, coratings_top_10=coratings)
        # print(df.sort_values('similarity'))
        # print(coratings[0][1140])

        # compute coratings
        # from recsys_sparse import UtilityMatrix
        # m_nan = np.copy(um.astype(float))
        # m_nan[m_nan == 0] = np.nan
        # umrs = UtilityMatrix(m_nan, beta=beta)
        # coratings = {i: collections.defaultdict(int) for i in range(um.shape[1])}
        # not_nan_indices = umrs.get_not_nan_indices(umrs.r)
        # idx_count = len(not_nan_indices)
        # for idx, (u, i) in enumerate(not_nan_indices):
        #     if ((idx+1) % 10000) == 0:
        #         print(idx+1, '/', idx_count, end='\r')
        #     s_u_i = umrs.similar_items(u, i, k, use_all=True)
        #     for ci in s_u_i:
        #         coratings[i][ci] += 1
        # self.save_recommendation_data(coratings, 'coratings')
        # # self.load_recommendataion_data('coratings')
        # coratings = m.coratings_r

        threshold = 1
        if self.dataset == 'movielens':
            threshold = 1
        elif self.dataset == 'bookcrossing':
            threshold = 1
        elif self.dataset == 'imdb':
            threshold = 1

        sims = np.zeros((m.coratings_r.shape[1], m.coratings_r.shape[1]))
        nnz = m.coratings_r.nnz
        cr_coo = m.coratings_r.tocoo()
        print('computing similarity thresholds...')
        for idx, (x, y, v) in enumerate(itertools.izip(cr_coo.row, cr_coo.col, cr_coo.data)):
            print('\r', idx, '/', nnz, end='')
            if v > threshold:  # confidence threshold
                sims[x, y] = w[x, y]
        print('\nthreshold =', threshold, '\n')
        sim_mat = SimilarityMatrix(sims)
        self.save_recommendation_data(sim_mat, 'sim_mat')
        return sim_mat

    def get_interpolation_weights(self):

        if self.dataset == 'movielens':
            kwargs = {
                'eta_type': 'bold_driver',
                'k': 15,
                'eta': 0.00005,
                'regularize': True,
                'init': 'sim',
                'nsteps': 50
            }

        elif self.dataset == 'bookcrossing':
            # kwargs = {
            #     'eta_type': 'bold_driver',
            #     'k': 20,
            #     'eta': 0.00001,
            #     'regularize': True,
            #     'init': 'zeros',
            #     'nsteps': 51,
            # }
            kwargs = {
                'eta_type': 'bold_driver',
                'k': 20,
                'eta': 0.0001,
                'regularize': True,
                'init': 'random_small',
                'nsteps': 21,
            }

        elif self.dataset == 'imdb':
            kwargs = {
                'eta_type': 'bold_driver',
                'k': 5,
                'eta': 0.0001,
                'regularize': True,
                'init': 'sim',
                'nsteps': 11,
            }

        if self.load_cached:
            w, k, beta, um = self.load_recommendation_data('iw_data')
            kwargs['w'] = w
            if self.sparse:
                wf = recsys_sparse.WeightedCFNNBiased(um, **kwargs)
            else:
                wf = recsys.WeightedCFNNBiased(um, **kwargs)
        else:
            # typical values for n lie in the range of 20-50 (Bell & Koren 2007)
            m = self.get_utility_matrix()
            m = m.astype(float)
            # m_nan = np.copy(m)
            # m_nan[m_nan == 0] = np.nan
            beta = 1  # for now, using beta=1 seems to work pretty well for both
            if self.dataset == 'imdb':
                beta = 10

            if self.sparse:
                um = recsys_sparse.UtilityMatrix(m, beta=beta)
            else:
                um = recsys.UtilityMatrix(m, beta=beta)

            if self.sparse:
                wf = recsys_sparse.WeightedCFNNBiased(um, **kwargs)
            else:
                wf = recsys.WeightedCFNNBiased(um, **kwargs)

            print('beta = ', beta)
            print('sparse = ', self.sparse)

        self.user_predictions = [[] for u in self.example_users]
        for idx, user_type in enumerate(self.user_types):
            u = self.example_users[idx]
            for i in range(um.r.shape[1]):
                p = wf.predict(u, i)
                if np.isfinite(p):
                    self.user_predictions[idx].append(p)
                else:
                    self.user_predictions[idx].append(-1)

        if self.sparse:
            self.user_rated = [set(np.where(~np.isnan(um.r[u, :].toarray()))[0])
                               for u in self.example_users
                               ]
        else:
            self.user_rated = [set(np.where(~np.isnan(um.r[u, :]))[0])
                               for u in self.example_users
                               ]

        self.save_recommendation_data(
            [wf.w, wf.k, beta, wf.m],
            'iw_data'
        )
        self.save_recommendation_data(
            [self.user_rated, self.user_predictions],
            'iw_predictions'
        )
        return wf.w, wf.k, beta, wf.m


class AssociationRuleRecommender(RatingBasedRecommender):
    def __init__(self, dataset, load_cached=False, sparse=False):
        super(AssociationRuleRecommender, self).__init__(
            dataset, 'rbar', load_cached=load_cached, sparse=sparse
        )

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

    def rating_stats(self, um):
        ratings = [(i, np.sum(um[:, i])) for i in range(um.shape[1])]
        print('ratings:')
        for r in sorted(ratings, key=operator.itemgetter(1), reverse=True)[:10]:
            print('   ', r[1], self.id2title[r[0]])

    def corating_stats(self, coratings, item_id=0):
        print('coratings for item %d %s:' % (item_id, self.id2title[item_id]))
        for r in sorted(coratings[item_id].items(), key=operator.itemgetter(1),
                        reverse=True)[:10]:
            print('   ', r[1], self.id2title[r[0]], '(', r[0], ')')

    def ar_simple(self, um, coratings, x, y):
        denominator = coratings[x][y]
        numerator = np.sum(um[:, x])
        if numerator > 0 and denominator > 0:
            sim = denominator / numerator
            # print(numerator, denominator, sim)
            # print(sim)
            return sim
        else:
            # print(numerator, denominator, '--ZERO--')
            print('--ZERO--')
            return -1

    def ar_complex(self, um, coratings, x, y):
        # ((x and y) * !x) / ((!x and y) * x)
        denominator = coratings[x][y] * (np.sum(um) - np.sum(um[:, x]))
        numerator = (sum(coratings[y].values()) - coratings[x][y]) * np.sum(um[:, x])
        if numerator > 0 and denominator > 0:
            sim = denominator / numerator
            # print(numerator, denominator, sim)
            # print(sim)
            return sim
        else:
            # print(numerator, denominator, '--ZERO--')
            print('--ZERO--')
            return -1

    def ar_both(self, um, coratings, x, y):
        simple = self.ar_simple(um, coratings, x, y)
        complex = self.ar_complex(um, coratings, x, y)
        print('s: %.4f, c: %.4f' % (simple, complex))

    def get_rating_stats(self):
        um = self.get_utility_matrix()
        um.data = np.ones(um.data.shape[0])
        counts = np.array(um.sum(axis=0))[0]

        # get possible coratings
        item_corating_counts = np.array(um.sum(axis=1)).T[0]
        cr_sum = sum((c * (c-1))/2 for c in item_corating_counts)
        cr_max = um.shape[0] * (um.shape[1] * (um.shape[1]-1)) / 2
        print('%.10f, %d' % (cr_sum/cr_max, cr_sum))
        # sys.exit()

        dataset_id2id = {v: k for k, v in self.id2dataset_id.items()}
        cts = []
        for did in self.df['dataset_id']:
            cts.append(counts[dataset_id2id[did]])
        self.df['rating_count'] = cts
        self.df.to_pickle(os.path.join(self.data_folder, 'item_stats.obj'))

    def get_similarity_matrix(self):
        if self.load_cached:
            sim_mat = self.load_recommendation_data('sim_mat_sparse')
            return SimilarityMatrix(sim_mat)

        print(1)
        um = self.get_utility_matrix()
        um.data = np.ones(um.data.shape[0])

        print(2)
        coratings = um.T.dot(um).toarray()
        np.fill_diagonal(coratings, 0)
        um = um.toarray()

        print(3)
        um_inv = np.copy(um)
        print(3.1)
        um_inv[um_inv == 0] = 2
        print(3.2)
        um_inv[um_inv == 1] = 0
        print(3.3)
        um_inv[um_inv == 2] = 1

        print(4)
        not_coratings = um_inv.T.dot(um)

        print(5)
        col_sum = um.sum(axis=0)
        not_col_sum = um.shape[0] - col_sum

        print(6)
        col_sums = np.matlib.repmat(col_sum, coratings.shape[0], 1)
        not_col_sums = np.matlib.repmat(not_col_sum, not_coratings.shape[0], 1)

        print(7)
        numerator = coratings * not_col_sums.T
        denominator = not_coratings * col_sums.T

        print(8)
        sims = numerator / denominator
        sims[np.isnan(sims)] = 0
        sims[np.isinf(sims)] = 0

        self.save_recommendation_data(sims, 'sim_mat_sparse')
        return SimilarityMatrix(sims)

    def corating_stats_sparse(self, coratings, item_id=0):
        print('coratings for item %d %s:' % (item_id, self.id2title[item_id]))
        for r in np.argsort(coratings[0])[-10:]:
            print('   ', int(coratings[item_id, r]), self.id2title[r], '(', r, ')')


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    from datetime import datetime
    start_time = datetime.now()

    GRAPH_SUFFIX = ''
    # DATASET = 'bookcrossing'
    DATASET = 'movielens'
    # DATASET = 'imdb'
    print('GRAPH_SUFFIX =', GRAPH_SUFFIX)
    print('DATASET =', DATASET)

    # r = ContentBasedRecommender(dataset=DATASET)
    # r = AssociationRuleRecommender(dataset=DATASET, load_cached=False, sparse=True)
    # r = RatingBasedRecommender(dataset=DATASET, load_cached=True, sparse=False)
    r = MatrixFactorizationRecommender(dataset=DATASET, load_cached=True, sparse=False)
    # r = InterpolationWeightRecommender(dataset=DATASET, load_cached=False, sparse=True)

    # r.get_recommendations()
    r.get_rating_stats()

    print('GRAPH_SUFFIX =', GRAPH_SUFFIX)
    print('DATASET =', DATASET)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))





