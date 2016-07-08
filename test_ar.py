# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import itertools
import numpy as np
import numpy.matlib
np.matlib = numpy.matlib
import pdb
import scipy.sparse


def get_similarity_matrix_old(um):
    um = np.where(um == 0, um, 1)  # set all ratings to 1 ???
    # um = np.where(um >= 4, 1, 0)  # set all high ratings to 1
    ucount = um.shape[0]
    icount = um.shape[1]

    coratings = {i: collections.defaultdict(int) for i in range(icount)}
    for u in range(ucount):
        print('\r', u+1, '/', ucount, end='')
        items = np.nonzero(um[u, :])[0]
        for i in itertools.combinations(items, 2):
            coratings[i[0]][i[1]] += 1
            coratings[i[1]][i[0]] += 1
    # save_recommendation_data(coratings, 'coratings')
    # coratings = load_recommendation_data('coratings')

    not_coratings = {i: collections.defaultdict(int) for i in range(icount)}
    for i in coratings.keys():
        print('\r', i+1, '/', len(coratings), end='')
        not_rated_i = set(np.where(um[:, i] == 0)[0])
        for j in coratings[i].keys():
            rated_j = set(np.where(um[:, j] == 1)[0])
            not_coratings[i][j] = len(not_rated_i & rated_j)
    # # save_recommendation_data(not_coratings, 'not_coratings')
    # not_coratings = load_recommendation_data('not_coratings')

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

    return sims


def get_similarity_matrix(um):
    print(1)
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

    return sims

class Test(object):
    def __init__(self, path, targets_original):
        self.path = path
        self.targets_original = targets_original

    def compute_stats(self):
        STEPS_MAX = 50
        self.path_original = self.path[:]  # DEBUG
        self.stats = np.zeros(STEPS_MAX + 1)
        self.path = self.path[2:]
        if self.path[-2:] == ['*', '*']:
            self.path = self.path[:-2]
        diff = len(self.path) - 2 * self.path.count(u'*') - STEPS_MAX - 1
        if diff > 0:
            self.path = self.path[:-diff]

        path = ' '.join(self.path).split('*')
        path = [l.strip().split(' ') for l in path]
        path = [path[0]] + [p[1:] for p in path[1:]]

        del self.targets_original[0]
        val = 0
        len_sum = -1
        for p in path:
            self.stats[len_sum:len_sum+len(p)] = val
            len_sum += len(p)
            val += (1 / len(self.targets_original))

        if len_sum < len(self.stats):
            fill = self.stats[len_sum - 1]
            if path[-1] and path[-1][-1] in self.targets_original[len(path)-1]:
                fill = min(fill+1/3, 1.0)
            self.stats[len_sum:] = fill

        print(self.stats)
        pdb.set_trace()

if __name__ == '__main__':
    p = [
        u'006100345X', u'*',
        u'006100345X', u'0', u'0', u'0', u'0399134700', u'*',
        u'0399134700', u'0', u'0', u'0006512062', u'*',
        u'0006512062', u'0', u'0', u'0', u'0', u'0', u'0060509392'
     ]
    to = [
        ['006100345X'],
        [u'0060198702', u'006093736X', u'0061000027', u'0141001828',
         u'0156007754', u'0312084986', u'0345285859', u'0345433491',
         u'034544003X', u'0394545370', u'0399134409', u'0399134700',
         u'0425144062', u'044020352X', u'0440204429', u'0446343455',
         u'0449202496', u'0451205634', u'067091021X', u'0671455990',
         u'0767904133', u'0786014245', u'0836218515', u'0836220986',
         u'0842342702'],
        [u'0006512062', u'0060089555', u'0060155515', u'0060171928',
         u'0140185216', u'034542705X', u'0385304943', u'0394531809',
         u'0451204530', u'0553050672', u'0553209906', u'0684826127',
         u'0743427149', u'0743431030', u'0786866195'],
        [u'0060509392', u'0312187106', u'0312261594', u'0330376136',
         u'0345469674', u'0373250479', u'0385721234', u'039912764X',
         u'0425155404', u'0425183394', u'0446526614', u'0452281679',
         u'0553756850', u'0671025708', u'067179356X', u'0743202562',
         u'0767907809']
    ]
    t = Test(p, to)
    t.compute_stats()
    print(t.stats)
    pdb.set_trace()

    # um_dense = np.array([  # simple test case
    #     [5, 1, 0, 2, 2, 4, 3, 2],
    #     [1, 5, 2, 5, 5, 1, 1, 4],
    #     [2, 0, 3, 5, 4, 1, 2, 4],
    #     [4, 3, 5, 3, 0, 5, 3, 0],
    #     [2, 0, 1, 3, 0, 2, 5, 3],
    #     [4, 1, 0, 1, 0, 4, 3, 2],
    #     [4, 2, 1, 1, 0, 5, 4, 1],
    #     [5, 2, 2, 0, 2, 5, 4, 1],
    #     [4, 3, 3, 0, 0, 4, 3, 0]
    # ])
    #
    # um_sparse = scipy.sparse.csr_matrix(um_dense)

    # dataset = 'bookcrossing'
    # # dataset = 'movielens'
    # # dataset = 'imdb'
    # um_dense = np.load('data/' + dataset + '/recommendation_data/RatingBasedRecommender_um.obj.npy')
    # um_sparse = np.load('data/' + dataset + '/recommendation_data/RatingBasedRecommender_um_sparse.obj.npy').item()

    # sims_new = get_similarity_matrix(um_sparse)
    # # sims_old = get_similarity_matrix_old(um_dense)
    # pdb.set_trace()


