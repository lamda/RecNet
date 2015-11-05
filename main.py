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


# TODO: move to separate file
np.random.seed(2014)
DEBUG = True
DATA_BASE_FOLDER = 'data'
NUMBER_OF_RECOMMENDATIONS = [5, 10]
FRACTION_OF_DIVERSIFIED_RECOMMENDATIONS = 0.4
NUMBER_OF_POTENTIAL_RECOMMENDATIONS = 25  # should be 50?


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
        for col_idx in range(nd):
            div_column = np.zeros(self.sims.sims.shape[0])
            for idx in range(self.sims.sims.shape[0]):
                node_max, val_max = -1, -1
                for node in range(NUMBER_OF_POTENTIAL_RECOMMENDATIONS):
                    if node not in idx2sel[idx]:
                        val = sum(self.sims.sims[node, r]
                                  for r in idx2sel[idx])
                        if val > val_max:
                            val_max = val
                            node_max = node
                div_column[idx] = node_max
                idx2sel[idx].add(node_max)
            results.append(div_column)
        return np.array(results).T


class TopNDivExpRelRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivExpRelRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = 'top_n_div_exprel'

    def get_recommendations(self, n):
        return recs.astype(int)


class Recommender(object):
    def __init__(self, dataset):
        self.label = ''
        self.dataset = dataset
        self.data_folder = os.path.join(DATA_BASE_FOLDER, self.dataset)
        self.dataset_folder = os.path.join(self.data_folder, 'dataset')
        self.graph_folder = os.path.join(self.data_folder, 'graphs')
        self.db_file = os.path.join(self.data_folder, 'database.db')
        self.db_main_table = 'movies' if dataset == 'movielens' else 'books'
        if not os.path.exists(self.graph_folder):
            os.makedirs(self.graph_folder)

        data = self.query_db('SELECT * FROM ' + self.db_main_table)
        idx = [d[0] for d in data]
        data = [(d[1], d[2], d[4], d[5], d[3]) for d in data]
        cols = ['cf_title', 'wp_title', 'original_title', 'wp_id', 'wp_text']
        self.df = pd.DataFrame(data=data, columns=cols, index=idx)
        self.id2original_title = {
            t[0]: t[1] for t in zip(self.df.index, self.df['original_title'])
        }
        if DEBUG:
            self.df = self.df.iloc[:25]
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
                    outfile.write(self.id2original_title[ridx+1] + '\t' +
                                  self.id2original_title[r+1] + '\n')

    def get_similarity_matrix(self):
        raise NotImplementedError

    def get_recommendations(self):
        strategies = [
            # TopNRecommendationStrategy,
            # TopNDivRandomRecommendationStrategy,
            TopNDivDiversifyRecommendationStrategy,
            # TopNDivExpRelRecommendationStrategy,
        ]

        for strategy in strategies:
            for n in NUMBER_OF_RECOMMENDATIONS:
                s = strategy(self.similarity_matrix)
                recs = s.get_recommendations(n=n)
                self.save_graph(recs, label=s.label, n=n)


class ContentBasedRecommender(Recommender):
    def __init__(self, dataset):
        super(ContentBasedRecommender, self).__init__(dataset)
        self.label = 'cb'

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix(self.df['wp_text'])
        super(ContentBasedRecommender, self).get_recommendations()

    # @decorators.Cached
    def get_similarity_matrix(self, data, max_features=50000, simple=False):
        """get the TF-IDF similarity values of a given list of text"""

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
        similarity = v_dense * v_dense.T  # cosine similarity
        return SimilarityMatrix(similarity)


class RatingBasedRecommender(Recommender):
    def __init__(self, dataset):
        super(RatingBasedRecommender, self).__init__(dataset)
        self.label = 'rb'

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

    # @decorators.Cached
    def get_utility_matrix(self):
        # load user ids
        item_ids = set(map(str, self.df.index))
        item2matrix = {m: i for i, m in enumerate(self.df.index)}
        user_ids = set()
        path_ratings = os.path.join(self.dataset_folder, 'ratings.dat')
        with io.open(path_ratings, encoding='latin-1') as infile:
            for line in infile:
                user, item = line.split('::')[:2]
                if item in item_ids:
                    user_ids.add(int(user))
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
        return matrix

    def get_similarity_matrix(self):
        um = self.get_utility_matrix()

        # transpose M because pdist calculates similarities between lines
        similarity = scipy.spatial.distance.pdist(um.T, 'correlation')

        # correlation is undefined for zero vectors --> set it to the max
        # max distance is 2 because the pearson correlation runs from -1...+1
        similarity[np.isnan(similarity)] = 2.0
        similarity = scipy.spatial.distance.squareform(similarity)
        return SimilarityMatrix(1 - similarity)


class MatrixFactorizationRecommender(RatingBasedRecommender):
    def __init__(self):
        pass


class InterpolationWeightRecommender(RatingBasedRecommender):
    def __init__(self):
        pass


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
    # TODO: use     @decorators.Cached
    # cbr = ContentBasedRecommender(dataset='movielens')
    # cbr.get_recommendations()

    rbr = RatingBasedRecommender(dataset='movielens')
    rbr.get_recommendations()


