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
        self.df['dataset_id'] = self.df['dataset_id'].apply(lambda i: unicode(i))
        self.id2title = {
            t[0]: t[1] for t in zip(self.df.index, self.df['original_title'])
        }
        self.title2id = {v: k for k, v in self.id2title.items()}
        self.id2dataset_id = {
            t[0]: t[1] for t in zip(self.df.index, self.df['dataset_id'])
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

    def save_graph(self, recs, label, n):
        file_name = os.path.join(
            self.graph_folder,
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
        # with open(fname, 'rb') as infile:
        #     obj = pickle.load(infile)
        obj = np.load(fname + '.npy')
        if not obj.shape:
            obj = obj.item()
        return obj


class ContentBasedRecommender(Recommender):
    def __init__(self, dataset, load_cached=False):
        super(ContentBasedRecommender, self).__init__(dataset, 'cb',
                                                      load_cached)

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(ContentBasedRecommender, self).get_recommendations()

    def get_similarity_matrix(self):
        """get the TF-IDF similarity values of a given list of text"""
        import nltk
        data = self.df['wp_text']
        max_features = 50000
        simple = False

        class LemmaTokenizer(object):
            """
            lemmatizer (scikit-learn.org/dev/modules/feature_extraction.html
                          #customizing-the-vectorizer-classes)
            """
            def __init__(self):
                self.wnl = nltk.WordNetLemmatizer()

            def __call__(self, doc):
                return [self.wnl.lemmatize(t) for t in nltk.word_tokenize(doc)]

        path_stopw = os.path.join(DATA_BASE_FOLDER, 'stopwords.txt')
        stopw = [l.strip() for l in io.open(path_stopw, encoding='utf-8-sig')]

        if simple:
            cv = sklearn.feature_extraction.text.CountVectorizer()
        else:
            cv = sklearn.feature_extraction.text.CountVectorizer(
                stop_words=stopw,
                tokenizer=LemmaTokenizer(),
                max_features=max_features
            )
        counts = cv.fit_transform(data)

        v = sklearn.feature_extraction.text.TfidfTransformer()
        v = v.fit_transform(counts)
        v_dense = v.todense()
        similarity = np.array(v_dense * v_dense.T)  # cosine similarity
        return SimilarityMatrix(similarity)


class RatingBasedRecommender(Recommender):
    def __init__(self, dataset, label='rb', load_cached=False, sparse=False):
        super(RatingBasedRecommender, self).__init__(
            dataset, label, load_cached
        )
        self.sparse = sparse

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

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


class MatrixFactorizationRecommender(RatingBasedRecommender):
    def __init__(self, dataset, load_cached=False, sparse=False):
        super(MatrixFactorizationRecommender, self).__init__(
            dataset, 'rbmf', load_cached, sparse
        )

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

    def get_similarity_matrix(self):
        if self.load_cached:
            sim_mat = self.load_recommendation_data('sim_mat')
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
            # f = recsys.Factors(um, k=5, nsteps=150, eta_type='bold_driver',
            #                    regularize=True, eta=0.0001, init='random_small',
            #                    lamda=0.25, reset_params=True)
            kwargs = {
                'k': 5,
                'eta': 0.0001,
                'eta_type': 'bold_driver',
                'init': 'random_small',
                'regularize': True,
                'nsteps': 150,
                'lamda': 0.25,
                'reset_params': True
            }

        elif self.dataset == 'imdb':
            #     k=15, nsteps=500, eta_type='bold_driver', regularize=True,
            #     eta=0.00001, init='random'
            # f = recsys.Factors(um, k=5, eta=0.00001, eta_type='bold_driver',
            #                    init='random', regularize=True, nsteps=1000)
            kwargs = {
                'k': 10,
                'eta': 0.000001,
                'eta_type': 'bold_driver',
                'init': 'random_small',
                'regularize': True,
                'nsteps': 1000
            }

        if self.sparse:
            f = recsys_sparse.Factors(um, **kwargs)
        else:
            f = recsys.Factors(um, **kwargs)

        return f.q


class InterpolationWeightRecommender(RatingBasedRecommender):
    def __init__(self, dataset, load_cached=False, sparse=False):
        super(InterpolationWeightRecommender, self).__init__(
            dataset, 'rbiw', load_cached, sparse
        )

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

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

    def get_similarity_matrix(self):
        if self.load_cached:
            sim_mat = self.load_recommendation_data('sim_mat')
            return sim_mat
        w, k, beta, m = self.get_interpolation_weights()

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
        coratings = m.coratings_r

        threshold = 1
        if self.dataset == 'movielens':
            threshold = 1
        elif self.dataset == 'bookcrossing':
            threshold = 1
        elif self.dataset == 'movielens':
            threshold = 1

        sims = np.zeros((m.coratings_r.shape[1], m.coratings_r.shape[1]))
        nnz = m.coratings_r.nnz

        cr_coo = m.coratings_r.tocoo()
        for idx, (x, y, v) in enumerate(itertools.izip(cr_coo.row, cr_coo.col, cr_coo.data)):
            print('\r', idx, '/', nnz, end='')
            if v > threshold:  # confidence threshold
                sims[x, y] = w[x, y]
        print('threshold =', threshold, '\n')
        sim_mat = SimilarityMatrix(sims)
        self.save_recommendation_data(sim_mat, 'sim_mat')
        return sim_mat

    def get_interpolation_weights(self):
        if self.load_cached:
            w, k, beta, um = self.load_recommendation_data('iw_data')
            return w, k, beta, um

        # typical values for n lie in the range of 20-50 (Bell & Koren 2007)
        m = self.get_utility_matrix()
        m = m.astype(float)
        m_nan = np.copy(m)
        m_nan[m_nan == 0] = np.nan
        beta = 1  # for now, using beta=1 seems to work pretty well for both

        if self.sparse:
            um = recsys_sparse.UtilityMatrix(m, beta=beta)
        else:
            um = recsys.UtilityMatrix(m, beta=beta)

        if self.dataset == 'movielens':
            # for MovieLens:
            # beta = 1
            # um = recsys.UtilityMatrix(m_nan, beta=beta)
            # wf = recsys.WeightedCFNNBiased(um, eta_type='bold_driver', k=15,
            #                                eta=0.000001, regularize=True,
            #                                init='sim', nsteps=50)

            kwargs = {
                'eta_type': 'bold_driver',
                'k': 15,
                'eta': 0.00001,
                'regularize': True,
                'init': 'sim',
                'nsteps': 50
            }

            if self.sparse:
                wf = recsys_sparse.WeightedCFNNBiased(um, **kwargs)
            else:
                wf = recsys.WeightedCFNNBiased(um, **kwargs)

        elif self.dataset == 'bookcrossing':
            # for BookCrossing:
            #    beta = 1
            #    eta_type='bold_driver', k=20, eta=0.00001, regularize=True,
            #    init='zeros'
            kwargs = {
                'eta_type': 'bold_driver',
                'k': 20,
                'eta': 0.00001,
                'regularize': True,
                'init': 'zeros',
                'nsteps': 60,
            }

            if self.sparse:
                wf = recsys_sparse.WeightedCFNNBiased(um, **kwargs)
            else:
                wf = recsys.WeightedCFNNBiased(um, **kwargs)

        elif self.dataset == 'imdb':
            # for IMDb:
            # beta = 1
            # um = recsys.UtilityMatrix(m_nan, beta=beta)
            # wf = recsys.WeightedCFNNBiased(um, eta_type='bold_driver', k=15,
            #                                eta=0.000001, regularize=True,
            #                                init='sim', nsteps=50)

            kwargs = {
                'eta_type': 'bold_driver',
                'k': 5,
                'eta': 0.000075,
                'regularize': True,
                'init': 'random_small',
                'nsteps': 101,
                'reset_params': False,
                'lamda': 0.25,
            }

            if self.sparse:
                wf = recsys_sparse.WeightedCFNNBiased(um, **kwargs)
            else:
                wf = recsys.WeightedCFNNBiased(um, **kwargs)

        print('beta = ', beta)
        print('sparse = ', self.sparse)
        self.save_recommendation_data([wf.w, wf.k, beta, wf.m], 'iw_data')
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

    def get_similarity_matrix_old(self):
        if self.load_cached:
            sim_mat = self.load_recommendation_data('sim_mat')
            return sim_mat
        um = self.get_utility_matrix()
        um = np.where(um == 0, um, 1)  # set all ratings to 1
        # um = np.where(um >= 4, 1, 0)  # set all high ratings to 1
        ucount = um.shape[0]
        icount = um.shape[1]

        # coratings = {i: collections.defaultdict(int) for i in range(icount)}
        # for u in range(ucount):
        #     print('\r', u+1, '/', ucount, end='')
        #     items = np.nonzero(um[u, :])[0]
        #     for i in itertools.combinations(items, 2):
        #         coratings[i[0]][i[1]] += 1
        #         coratings[i[1]][i[0]] += 1
        # self.save_recommendation_data(coratings, 'coratings')
        coratings = self.load_recommendation_data('coratings')

        # not_coratings = {i: collections.defaultdict(int) for i in range(icount)}
        # for i in coratings.keys():
        #     print('\r', i+1, '/', len(coratings), end='')
        #     not_rated_i = set(np.where(um[:, i] == 0)[0])
        #     for j in coratings[i].keys():
        #         rated_j = set(np.where(um[:, j] == 1)[0])
        #         not_coratings[i][j] = len(not_rated_i & rated_j)
        # self.save_recommendation_data(not_coratings, 'not_coratings')
        not_coratings = self.load_recommendation_data('not_coratings')

        # # debug helpers
        # self.rating_stats(um)
        # self.corating_stats(coratings, item_id=0)
        # self.ar_simple(um, coratings, 0, 2849)
        # self.ar_complex(um, coratings, 0, 2849)
        # self.ar_both(um, coratings, 0, 2849)

        sims = np.zeros((icount, icount))
        for x in range(icount):
            is_x = np.sum(um[:, x])
            not_x = um.shape[0] - is_x
            for y in coratings[x]:
                # # (x and y) / x  simple version
                # denominator = coratings[x][y]
                # numerator = is_x

                # ((x and y) * !x) / ((!x and y) * x)  complex version
                denominator = coratings[x][y] * not_x
                numerator = not_coratings[x][y] * is_x

                if numerator > 0:
                    sims[x, y] = denominator / numerator

        pdb.set_trace()

        sim_mat = SimilarityMatrix(1 - sims)
        self.save_recommendation_data(sim_mat, 'sim_mat')
        return sim_mat

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
        # np.place(um_inv, um_inv == 0, 2)
        um_inv[um_inv == 0] = 2
        print(3.2)
        # np.place(um_inv, um_inv == 1, 0)
        um_inv[um_inv == 1] = 0
        print(3.3)
        # np.place(um_inv, um_inv == 2, 1)
        um_inv[um_inv == 2] = 1

        print(4)
        not_coratings = um_inv.T.dot(um)
        # icount = um.shape[1]
        # coratings_dense = self.load_recommendation_data('coratings')
        # tmp = scipy.sparse.dok_matrix((icount, icount))
        # for i in range(icount):
        #     print('\r', i+1, '/', icount, end='')
        #     for j in range(icount):
        #         tmp[i, j] = coratings_dense[i][j]
        # print()
        # tmp = tmp.toarray()
        # pdb.set_trace()

        # # debug helpers
        # self.rating_stats(um)
        # self.corating_stats(coratings, item_id=0)
        # self.ar_simple(um, coratings, 0, 2849)
        # self.ar_complex(um, coratings, 0, 2849)
        # self.ar_both(um, coratings, 0, 2849)

        print(5)
        col_sum = um.sum(axis=0)
        not_col_sum = um.shape[0] - col_sum

        # sims = np.zeros((icount, icount))
        # for x in range(icount):
        #     print('\r', x, end='')
        #     for y in range(icount):
        #         # # (x and y) / x  simple version
        #         # denominator = coratings[x, y]
        #         # numerator = col_sum[x]
        #
        #         # ((x and y) * !x) / ((!x and y) * x)  complex version
        #         denominator = coratings[x, y] * not_col_sum[x]
        #         numerator = not_coratings[x, y] * col_sum[x]
        #
        #         if numerator > 0:
        #             sims[x, y] = denominator / numerator

        print(6)
        col_sums = np.matlib.repmat(col_sum, coratings.shape[0], 1)
        not_col_sums = np.matlib.repmat(not_col_sum, not_coratings.shape[0], 1)

        print(7)
        denominator = coratings * not_col_sums.T
        numerator = not_coratings * col_sums.T

        print(8)
        sims = denominator / numerator
        sims[np.isnan(sims)] = 0
        sims[np.isinf(sims)] = 0

        # self.corating_stats_sparse(coratings, item_id=0)

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

    GRAPH_SUFFIX = '_r_5_5'
    SPARSE = True
    DATASET = 'imdb'
    print('GRAPH_SUFFIX =', GRAPH_SUFFIX)
    print('SPARSE =', SPARSE)
    print('DATASET =', DATASET)

    ## r = ContentBasedRecommender(dataset=DATASET)
    # r = RatingBasedRecommender(dataset=DATASET, load_cached=False)
    # r = AssociationRuleRecommender(dataset=DATASET, load_cached=False, sparse=SPARSE)
    # r = MatrixFactorizationRecommender(dataset=DATASET, load_cached=False, sparse=SPARSE)
    r = InterpolationWeightRecommender(dataset=DATASET, load_cached=False, sparse=SPARSE)

    r.get_recommendations()

    print('GRAPH_SUFFIX =', GRAPH_SUFFIX)
    print('SPARSE =', SPARSE)
    print('DATASET =', DATASET)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))





