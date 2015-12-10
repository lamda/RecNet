# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import io
import itertools
import numpy as np
import cPickle as pickle
# import nltk
import os
import pandas as pd
pd.set_option('display.width', 1000)
import pdb
import scipy.spatial.distance
import sklearn.feature_extraction.text
import sqlite3

import decorators
import recsys


# np.random.seed(2014)
# DEBUG = True
DEBUG = False
# DEBUG_SIZE = 255
DEBUG_SIZE = 750
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
        self.id2title = {
            t[0]: t[1] for t in zip(self.df.index, self.df['original_title'])
        }
        self.title2id = {v: k for k, v in self.id2title.items()}
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

        # use the centered version for similarity computation
        um_centered = np.copy(um.astype(float))
        um_centered[np.where(um_centered == 0)] = np.nan
        um_centered = um_centered - np.nanmean(um_centered, axis=0)[np.newaxis, :]
        um_centered[np.where(np.isnan(um_centered))] = 0

        # transpose M because pdist calculates similarities between rows
        similarity = scipy.spatial.distance.pdist(um_centered.T, 'cosine')

        # correlation is undefined for zero vectors --> set it to the max
        # max distance is 2 because the pearson correlation runs from -1...+1
        similarity[np.isnan(similarity)] = 2.0  # for correlation
        similarity = scipy.spatial.distance.squareform(similarity)
        return SimilarityMatrix(1 - similarity)


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
        # use the centered version for similarity computation
        q_centered = np.copy(q.astype(float).T)
        q_centered[np.where(q_centered == 0)] = np.nan
        q_centered = q_centered - np.nanmean(q_centered, axis=0)[np.newaxis, :]
        q_centered[np.where(np.isnan(q_centered))] = 0
        q_centered = q_centered.T

        # transpose M because pdist calculates similarities between rows
        # similarity = scipy.spatial.distance.pdist(q.T, 'correlation')
        similarity = scipy.spatial.distance.pdist(q_centered, 'cosine')

        # correlation is undefined for zero vectors --> set it to the max
        # max distance is 2 because the pearson correlation runs from -1...+1
        similarity[np.isnan(similarity)] = 2.0  # for correlation
        # similarity[np.isnan(similarity)] = 1.0  # for cosine
        similarity = scipy.spatial.distance.squareform(similarity)
        return SimilarityMatrix(1 - similarity)

    # @profile
    # @decorators.Cached
    def factorize(self, m, k=15, eta=0.000005, nsteps=1000):
        # k should be smaller than #users and #items (2-300?)
        m = m.astype(float)
        m[m == 0] = np.nan
        um = recsys.UtilityMatrix(m)
        # f = recsys.Factors(um, k, regularize=True, nsteps=nsteps, eta=eta)
        f = recsys.Factors(um, k=15, nsteps=1000, regularize=True, eta=0.00001, lamda=0.05, init_svd=False)
        return f.q


class InterpolationWeightRecommender(RatingBasedRecommender):
    def __init__(self, dataset):
        super(InterpolationWeightRecommender, self).__init__(dataset, 'rbiw')

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

    def get_coratings(self, um, mid, w, k=10):
        d = collections.defaultdict(int)
        nan_indices = set(um.rt_nan_indices)
        for u in range(um.rt.shape[0]):
            print(u, end='\r')
            if (u, mid) in nan_indices:
                continue
            s_u_i = um.similar_items(u, mid, k)
            for r in s_u_i:
                d[r] += 1
        indices = np.arange(0, um.rt.shape[1])
        coratings = [d[i] for i in indices]
        titles = [self.id2title[idx] for idx in indices]
        similarities = [w[mid, i] for i in indices]
        df = pd.DataFrame(index=indices,
                          data=zip(titles, coratings, similarities),
                          columns=['title', 'coratings', 'similarity'])
        return df

    # @decorators.Cached # TODO
    def get_similarity_matrix(self):
        um = self.get_utility_matrix()
        w = self.get_interpolation_weights(um)

        # from recsys import UtilityMatrix
        # with open('iw.obj', 'rb') as infile:
        #     print('DEBUG: loading IW matrix')
        #     w = pickle.load(infile)
        # with open('um.obj', 'rb') as infile:
        #     print('DEBUG: loading UM matrix')
        #     umrs = pickle.load(infile)
        # # # TODO: warum gibt es Ähnlichkeitswerte für Filme mit 0 Coratings?
        # df = self.get_coratings(umrs, 0, w, k=5)
        # print(df.sort_values('similarity'))
        # pdb.set_trace()
        # df2 = df[df['coratings'] > 0]; df2.sort_values('similarity')
        # print(np.sum(np.abs(w)))
        # # sys.exit()

        return SimilarityMatrix(w)

    # @profile
    # @decorators.Cached
    def get_interpolation_weights(self, m, nsteps=500, eta=0.00001, n=10):
        # typical values for n lie in the range of 20-50 (Bell & Koren 2007)
        m = m.astype(float)
        m_nan = np.copy(m)
        m_nan[m_nan == 0] = np.nan
        um = recsys.UtilityMatrix(m_nan)
        # with open('m.obj', 'wb') as outfile:
        #     pickle.dump(m, outfile)
        # sys.exit()
        # wf = recsys.WeightedCFNN(um, eta_type='constant', k=10, eta=0.0001, regularize=True, init_sim=True)
        wf = recsys.WeightedCFNN(um, eta_type='increasing', k=10, eta=0.000001, regularize=True, init='zeros')
        # wf = recsys.WeightedCFNN(um, eta_type='bold_driver', k=10, eta=0.0001, regularize=True, init_sim=False)
        # server oben: k=10, unten k=15
        with open('um.obj', 'wb') as outfile:
            pickle.dump(wf.m, outfile)
        with open('iw.obj', 'wb') as outfile:
            pickle.dump(wf.w, outfile)
        return wf.w


class AssociationRuleRecommender(RatingBasedRecommender):
    def __init__(self, dataset):
        super(AssociationRuleRecommender, self).__init__(dataset, 'rbar')

    def get_recommendations(self):
        self.similarity_matrix = self.get_similarity_matrix()
        super(RatingBasedRecommender, self).get_recommendations()

    def rating_stats(self, um):
        import operator
        ratings = [(i, np.sum(um[:, i])) for i in range(um.shape[1])]
        print('ratings:')
        for r in sorted(ratings, key=operator.itemgetter(1), reverse=True)[:10]:
            print('   ', r[1], self.id2title[r[0]])

    def corating_stats(self, coratings, item_id=0):
        import operator
        print('coratings for item %d %s:' % (item_id, self.id2title[item_id]))
        for r in sorted(coratings[item_id].items(), key=operator.itemgetter(1),
                        reverse=True)[:10]:
            print('   ', r[1], self.id2title[r[0]])

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

    def get_similarity_matrix(self):
        um = self.get_utility_matrix()
        um = np.where(um == 0, um, 1)  # set all ratings to 1
        # um = np.where(um >= 4, 1, 0)  # set all high ratings to 1
        ucount = um.shape[0]
        icount = um.shape[1]

        with open('coratings.obj', 'rb') as infile:
            coratings = pickle.load(infile)
        # pdb.set_trace()
        # self.rating_stats(um)
        # self.corating_stats(coratings, item_id=0)
        # self.ar_simple(um, coratings, 0, 2849)
        # self.ar_complex(um, coratings, 0, 2849)
        # self.ar_both(um, coratings, 0, 2849)

        # coratings = {i: collections.defaultdict(int) for i in range(icount)}
        # for u in range(ucount):
        #     print(u+1, '/', ucount, end='\r')
        #     items = np.nonzero(um[u, :])[0]
        #     for i in itertools.combinations(items, 2):
        #         coratings[i[0]][i[1]] += 1
        #         coratings[i[1]][i[0]] += 1
        # with open('coratings.obj', 'wb') as outfile:
        #     pickle.dump(coratings, outfile, -1)

        sims = np.zeros((icount, icount))
        sum_items = np.sum(um)
        sums_coratings = {x: sum(coratings[x].values()) for x in coratings}
        for x in range(icount):
            print(x+1, '/', icount, end='\r')
            not_x = (sum_items - np.sum(um[:, x]))
            is_x = np.sum(um[:, x])
            for y in coratings[x]:
                # # (x and y) / x  simple version
                # denominator = coratings[x][y]
                # numerator = is_x

            # ((x and y) * !x) / ((!x and y) * x)  complex version
            if (coratings[x][y] / is_x) > 0.25:  # confidence threshold
                denominator = coratings[x][y] * not_x
                numerator = (sums_coratings[y] - coratings[x][y]) * is_x
            else:
                denominator = numerator = 0

            if numerator > 0 and denominator > 0:
                    sims[x, y] = denominator / numerator

        return SimilarityMatrix(sims)


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
    # rbmf = MatrixFactorizationRecommender(dataset='movielens'); rbmf.get_recommendations()
    # rbiw = InterpolationWeightRecommender(dataset='movielens'); rbiw.get_recommendations()
    rbar = AssociationRuleRecommender(dataset='movielens'); rbar.get_recommendations()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    # values for MovieLens:
    #    CFNN, k = 10
    #    IW, k = 10, init = sim
    #    MF, k = 25, init = SVD
    #
    # values for BookCrossing:
    #    CFNN, k =
    #    IW, k = , init =
    #    MF, k = , init =




