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
DEBUG = True
DATA_BASE_FOLDER = 'data'
NUMBER_OF_RECOMMENDATIONS = [5, 10, 15]
NUMBER_OF_POTENTIAL_RECOMMENDATIONS = 50


class SimilarityMatrix(object):
    def __init__(self, sims):
        self.sims = sims

    def get_top_n(self, n):
        return self.filter_matrix(self.sims, n=n)

    def filter_matrix(self, m, n=NUMBER_OF_POTENTIAL_RECOMMENDATIONS):
        """delete diagonal entries from a matrix and keep only the top 50"""
        m_filtered = np.zeros((m.shape[0], m.shape[1] - 1))
        for index, line in enumerate(m):
            m_filtered[index, :] = np.delete(line, index)
        return m_filtered.argsort()[:, :n]


class RecommendationStrategy(object):
    def __init__(self, similarity_matrix):
        self.sims = similarity_matrix
        self.label = ''

    def get_recommendations(self, n):
        raise NotImplementedError


class TopNRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNRecommendationStrategy, self).__init__(similarity_matrix)
        self.label = 'top_n'

    def get_recommendations(self, n):
        return self.sims.get_top_n(n)


class TopNDivRandomRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivRandomRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = 'top_n_div_random'


class TopNDivDiversifyRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivDiversifyRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = 'top_n_div_diversify'


class TopNDivExpRelRecommendationStrategy(RecommendationStrategy):
    def __init__(self, similarity_matrix):
        super(TopNDivExpRelRecommendationStrategy, self).__init__(
            similarity_matrix
        )
        self.label = 'top_n_div_exprel'


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
            TopNRecommendationStrategy,
            # TopNDivRandomRecommendationStrategy,
            # TopNDivDiversifyRecommendationStrategy,
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

    #@decorators.Cached
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
        M = np.zeros((len(user_ids), len(item_ids)))

        # load ratings
        with io.open(path_ratings, encoding='latin-1') as infile:
            for line in infile:
                user, item, rat = line.split('::')[:3]
                user = int(user)
                rat = float(rat)
                if user in user_ids and item in item_ids:
                    M[user2matrix[user], item2matrix[int(item)]] = rat
        return M

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


