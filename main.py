# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import io
import numpy as np
import cPickle as pickle
import nltk
import os
import pandas as pd
import pdb
import scipy.spatial.distance
import sklearn.feature_extraction.text
import sqlite3

import decorators
import recsys


np.random.seed(2014)
DEBUG = True
# DEBUG = False
DEBUG_SIZE = 500
DATA_BASE_FOLDER = 'data'
NUMBER_OF_RECOMMENDATIONS = [5, 10]
FRACTION_OF_DIVERSIFIED_RECOMMENDATIONS = 0.4  # should be 0.4 TODO make a list?
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
    def __init__(self, similarity_matrix, ):
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
        self.label = 'top_n'

    def get_recommendations(self, n):
        return self.get_top_n_recommendations(n).astype(int)


class TopNDivRandomRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivRandomRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = 'top_n_div_random'

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
        self.label = 'top_n_div_diversify'

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
        self.label = 'top_n_div_exprel'

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
    def __init__(self, dataset, label):
        print(label)
        self.label = label
        self.dataset = dataset
        self.data_folder = os.path.join(DATA_BASE_FOLDER, self.dataset)
        self.dataset_folder = os.path.join(self.data_folder, 'dataset')
        self.graph_folder = os.path.join(self.data_folder, 'graphs')
        self.db_file = os.path.join(self.data_folder, 'database.db')
        self.db_main_table = 'movies' if dataset == 'movielens' else 'books'
        if not os.path.exists(self.graph_folder):
            os.makedirs(self.graph_folder)

        data = self.query_db('SELECT * FROM ' + self.db_main_table)
        data = [(d[0], d[1], d[2], d[4], d[5], d[3]) for d in data]
        cols = ['movielens_id', 'cf_title', 'wp_title', 'original_title',
                'wp_id', 'wp_text']
        self.df = pd.DataFrame(data=data, columns=cols)
        self.id2original_title = {
            t[0]: t[1] for t in zip(self.df.index, self.df['original_title'])
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
            self.label + '_' + label + '_' + unicode(n)
        )
        with io.open(file_name + '.txt', 'w', encoding='utf-8') as outfile:
            for ridx, rec in enumerate(recs):
                for r in rec:
                    outfile.write(unicode(ridx) + '\t' + unicode(r) + '\n')

        with io.open(file_name + '_resolved.txt', 'w', encoding='utf-8')\
                as outfile:
            for ridx, rec in enumerate(recs):
                for r in rec:
                    outfile.write(self.id2original_title[ridx] + '\t' +
                                  self.id2original_title[r] + '\n')

    def get_similarity_matrix(self):
        raise NotImplementedError

    def get_recommendations(self):
        strategies = [
            TopNRecommendationStrategy,
            TopNDivRandomRecommendationStrategy,
            TopNDivDiversifyRecommendationStrategy,
            TopNDivExpRelRecommendationStrategy,
        ]

        for strategy in strategies:
            s = strategy(self.similarity_matrix)
            print(s.label)
            for n in NUMBER_OF_RECOMMENDATIONS:
                recs = s.get_recommendations(n=n)
                self.save_graph(recs, label=s.label, n=n)


class ContentBasedRecommender(Recommender):
    def __init__(self, dataset):
        super(ContentBasedRecommender, self).__init__(dataset, 'cb')

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(ContentBasedRecommender, self).get_recommendations()

    @decorators.Cached
    def get_similarity_matrix(self):
        """get the TF-IDF similarity values of a given list of text"""
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
    def __init__(self, dataset, label='rb'):
        super(RatingBasedRecommender, self).__init__(dataset, label)

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

    # @decorators.Cached # TODO
    def get_utility_matrix(self):
        # load user ids
        item_ids = set(map(str, self.df['movielens_id']))
        item2matrix = {m: i for i, m in enumerate(self.df['movielens_id'])}
        user_ids = set()
        path_ratings = os.path.join(self.dataset_folder, 'ratings.dat')
        with io.open(path_ratings, encoding='latin-1') as infile:
            for line in infile:
                user, item = line.split('::')[:2]
                if item in item_ids:
                    user_ids.add(int(user))
        # if DEBUG:
        #     user_ids = set(
        #         np.random.choice(list(user_ids), DEBUG_SIZE, replace=False)
        #     )
        user2matrix = {u: i for i, u in enumerate(sorted(user_ids))}
        matrix = np.zeros((len(user_ids), len(item_ids)))

        # load ratings
        with io.open(path_ratings, encoding='latin-1') as infile:
            for line in infile:
                user, item, rat = line.split('::')[:3]
                user = int(user)
                rat = float(rat)
                if user in user_ids and item in item_ids:
                    matrix[user2matrix[user], item2matrix[int(item)]] = rat
        return matrix.astype(int)

    # @decorators.Cached # TODO
    def get_similarity_matrix(self):
        um = self.get_utility_matrix()

        # transpose M because pdist calculates similarities between lines
        similarity = scipy.spatial.distance.pdist(um.T, 'correlation')
        # similarity = scipy.spatial.distance.pdist(um.T, 'cosine')

        # correlation is undefined for zero vectors --> set it to the max
        # max distance is 2 because the pearson correlation runs from -1...+1
        similarity[np.isnan(similarity)] = 2.0  # for correlation
        # similarity[np.isnan(similarity)] = 1.0  # for cosine
        similarity = scipy.spatial.distance.squareform(similarity)
        return SimilarityMatrix(1 - similarity)

    def get_training_matrix_indices(self, um, fraction=0.2):
        i, j = np.nonzero(um)
        rands = np.random.choice(
            len(i),
            np.floor(fraction * len(i)),
            replace=False
        )
        return i[rands], j[rands]


class MatrixFactorizationRecommender(RatingBasedRecommender):
    def __init__(self, dataset):
        super(MatrixFactorizationRecommender, self).__init__(dataset, 'rbmf')

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

    # @decorators.Cached # TODO
    def get_similarity_matrix(self):
        um = self.get_utility_matrix()
        q = self.factorize(um)

        # transpose M because pdist calculates similarities between lines
        similarity = scipy.spatial.distance.pdist(q, 'correlation')
        # similarity = scipy.spatial.distance.pdist(q, 'cosine')

        # correlation is undefined for zero vectors --> set it to the max
        # max distance is 2 because the pearson correlation runs from -1...+1
        similarity[np.isnan(similarity)] = 2.0  # for correlation
        # similarity[np.isnan(similarity)] = 1.0  # for cosine
        similarity = scipy.spatial.distance.squareform(similarity)
        return SimilarityMatrix(1 - similarity)

    # @profile
    # @decorators.Cached
    def factorize(self, m, k=100, eta=0.0000001, nsteps=500, tol=1e-5):
        # k should be smaller than #users and #items (2-300?)
        m = m.astype(float)
        um = recsys.UtilityMatrix(m, self.get_training_matrix_indices(m), k)
        f = recsys.Factors(um, k, regularize=True, nsteps=nsteps, tol=tol,
                           eta=eta)
        return f.q


class InterpolationWeightRecommender(RatingBasedRecommender):
    def __init__(self, dataset):
        super(InterpolationWeightRecommender, self).__init__(dataset, 'rbiw')

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

    # @decorators.Cached # TODO
    def get_similarity_matrix(self):
        um = self.get_utility_matrix()
        w = self.get_interpolation_weights(um)

        # transpose M because pdist calculates similarities between lines
        similarity = scipy.spatial.distance.pdist(w, 'correlation')
        # similarity = scipy.spatial.distance.pdist(q, 'cosine')

        # correlation is undefined for zero vectors --> set it to the max
        # max distance is 2 because the pearson correlation runs from -1...+1
        similarity[np.isnan(similarity)] = 2.0  # for correlation
        # similarity[np.isnan(similarity)] = 1.0  # for cosine
        similarity = scipy.spatial.distance.squareform(similarity)
        return SimilarityMatrix(1 - similarity)

    # @profile
    # @decorators.Cached
    def get_interpolation_weights(self, m, nsteps=500, eta=0.000015, n=5):
        # typical values for n lie in the range of 20-50 (Bell & Koren 2007)
        m = m.astype(float)
        um = recsys.UtilityMatrix(m, self.get_training_matrix_indices(m), n)
        wf = recsys.WeightedCFNN2(um, nsteps=nsteps, eta=eta)
        return wf.w


class Graph(object):
    def __init__(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def compute_stats(self):
        pass


if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()
    # cbr = ContentBasedRecommender(dataset='movielens'); cbr.get_recommendations()
    # rbr = RatingBasedRecommender(dataset='movielens'); rbr.get_recommendations()
    # mfrbr = MatrixFactorizationRecommender(dataset='movielens'); mfrbr.get_recommendations()
    iwrbr = InterpolationWeightRecommender(dataset='movielens'); iwrbr.get_recommendations()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))



