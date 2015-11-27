# -*- coding: utf-8 -*-

from __future__ import division, print_function

import cPickle as pickle
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb

np.set_printoptions(linewidth=225)
# np.seterr(all='raise')


class UtilityMatrix:
    def __init__(self, r, hidden=None):
        self.r = r  # rating matrix (=utility matrix)
        self.rt, self.hidden = self.get_training_data(hidden)
        self.mu = np.nanmean(self.rt)
        self.b_i = np.nanmean(self.rt, axis=0) - self.mu
        self.b_u = np.nanmean(self.rt, axis=1) - self.mu
        self.s = self.get_similarities()
        self.sirt_cache = {}
        self.rt_not_nan_indices = self.get_not_nan_indices(self.rt)
        self.rt_nan_indices = self.get_nan_indices(self.rt)
        # self.r_nan_indices = self.get_nan_indices(self.r)
        # self.r_not_nan_indices = self.get_not_nan_indices(self.r)

    def get_training_data(self, hidden, training_share=0.2):
        if hidden is None:  # take some of the input as training data
            i, j = np.where(~((self.r == 0) | (np.isnan(self.r))))
            rands = np.random.choice(
                len(i),
                np.floor(training_share * len(i)),
                replace=False
            )
            hidden = np.vstack((i[rands], j[rands]))
        rt = np.copy(self.r)
        rt[hidden[0], hidden[1]] = np.nan
        return rt, hidden

    def get_similarities(self):
        rc = self.rt - np.nanmean(self.rt, axis=0)
        rc[np.isnan(rc)] = 0.0
        # ignore division errors, set the resulting nans to zero
        with np.errstate(all='ignore'):
            s = np.corrcoef(rc.T)
        s[np.isnan(s)] = 0.0

        return s

    def similar_items(self, u, i, k):
        try:
            return self.sirt_cache[(u, i)]
        except KeyError:
            r_u = self.rt[u, :]  # user ratings
            s_i = np.copy(self.s[i, :])  # item similarity
            # s_i[s_i < 0.0] = np.nan  # mask only to similar items
            s_i[i] = np.nan  # mask the item
            s_i[np.isnan(r_u)] = np.nan  # mask to items rated by the user
            nn = np.isnan(s_i).sum()  # how many invalid items

            s_i_sorted = np.argsort(s_i)
            s_i_k = tuple(s_i_sorted[-k - nn:-nn])
            self.sirt_cache[(u, i)] = s_i_k
            return s_i_k

    def get_not_nan_indices(self, m):
        nnan = np.where(~np.isnan(m))
        nnan_indices = zip(nnan[0], nnan[1])
        return nnan_indices

    def get_nan_indices(self, m):
        ynan = np.where(np.isnan(m))
        ynan_indices = zip(ynan[0], ynan[1])
        return ynan_indices


class Recommender:
    def __init__(self, m):
        self.m = m
        self.rmse = []

    def predict(self, u, i):
        raise NotImplementedError

    def training_error(self):
        sse = 0.0
        for u, i in self.m.rt_not_nan_indices:
            err = self.m.rt[u, i] - self.predict(u, i)
            sse += err ** 2

        # return np.sqrt(sse) / len(self.m.rt_not_nan_indices)
        return np.sqrt(sse / len(self.m.rt_not_nan_indices))

    def test_error(self):
        sse = 0.0
        for u, i in self.m.hidden.T:
            err = self.m.r[u, i] - self.predict(u, i)
            sse += err ** 2

        # return np.sqrt(sse) / self.m.hidden.shape[1]
        return np.sqrt(sse / self.m.hidden.shape[1])

    def plot_rmse(self, title='', suffix=None):
        plt.plot(range(len(self.rmse)), self.rmse)
        plt.xlabel("iteration")
        plt.ylabel("RMSE")
        plt.title(title + ' | ' + '%.4f' % self.rmse[-1] +
                  ' | ' + datetime.datetime.now().strftime("%H:%M:%S"))
        plt.savefig('rmse' + ('_' + suffix if suffix is not None else '') +
                    '.png')


class CFNN(Recommender):
    def __init__(self, m, k):
        Recommender.__init__(self, m)
        self.w = self.m.s
        self.k = k
        self.normalize = True
        print('k =', k)

    def predict(self, u, i):
        # predict a item-based CF rating based on the training data
        b_xi = self.m.mu + self.m.b_u[u] + self.m.b_i[i]
        if np.isnan(b_xi):
            if np.isnan(self.m.b_u[u]) and np.isnan(self.m.b_i[i]):
                return self.m.mu
            elif np.isnan(self.m.b_u[u]):
                return self.m.mu + self.m.b_i[i]
            else:
                return self.m.mu + self.m.b_u[u]

        n_u_i = self.m.similar_items(u, i, self.k)
        r = 0
        for j in n_u_i:
            diff = self.m.r[u, j] - (self.m.mu + self.m.b_u[u] + self.m.b_i[j])
            r += self.w[i, j] * diff
        if self.normalize:
            if r != 0:
                s = sum(self.w[i, j] for j in n_u_i)
                if not np.isfinite(r/sum(self.w[i, j] for j in n_u_i)) or np.isnan(r/sum(self.w[i, j] for j in n_u_i)):
                    pdb.set_trace()
                r /= s
        return b_xi + r


class Factors(Recommender):

    def __init__(self, m, k, nsteps=500, eta=0.000004, regularize=False,
                 newton=False, tol=1e-5, lamda=0.05, init_svd=True):
        Recommender.__init__(self, m)
        self.k = k
        self.nsteps = nsteps
        self.eta = eta
        self.regularize = regularize
        self.newton = newton
        self.tol = tol
        self.lamda = lamda

        if init_svd:
            # init by Singular Value Decomposition
            # m = np.copy(self.m.rt)
            m = self.m.rt
            m[np.where(np.isnan(m))] = 0
            ps, ss, vs = np.linalg.svd(m)
            self.p = ps[:, :self.k]
            self.q = np.dot(np.diag(ss[:self.k]), vs[:self.k, :]).T
        else:
            # init randomly
            self.eta *= 15  # use a higher eta for random initialization
            self.p = np.random.random((self.m.rt.shape[0], self.k))
            self.q = np.random.random((self.m.rt.shape[1], self.k))

        p_init = np.copy(self.p)
        q_init = np.copy(self.q)

        print('init_svd =', init_svd)
        print('k =', k)
        print('lamda =', self.lamda)
        print('eta = ', self.eta)

        self.factorize()

        print('init_svd =', init_svd)
        print('k =', k)
        print('lamda =', self.lamda)
        print('eta = ', self.eta)

        diff = np.linalg.norm(p_init - self.p) + np.linalg.norm(q_init - self.q)

        self.plot_rmse('%.4f' % diff, suffix='svd' if init_svd else 'random')
        print('test error: %.4f' % self.test_error())

    def predict(self, u, i):
        p_u = self.p[u, :]
        q_i = self.q[i, :]
        return np.dot(p_u, q_i.T)

    def factorize(self):
        for m in range(self.nsteps):
            masked = np.ma.array(self.m.rt, mask=np.isnan(self.m.rt))
            delta_p = np.ma.dot(np.dot(self.p, self.q.T) - masked, self.q)
            delta_q = np.ma.dot((np.dot(self.p, self.q.T) - masked).T, self.p)

            if self.regularize:
                delta_p += self.lamda * self.p
                delta_q += self.lamda * self.q

            self.p -= 2 * self.eta * delta_p
            self.q -= 2 * self.eta * delta_q

            self.rmse.append(self.training_error())
            print(m, 'eta = %.8f, rmse = %.8f' % (self.eta, self.rmse[-1]))
            if len(self.rmse) > 1:
                if abs(self.rmse[-1] - self.rmse[-2]) < self.tol:
                    break
                if self.rmse[-1] > self.rmse[-2]:
                    print('RMSE getting larger')
                    self.eta *= 0.5
                else:
                    self.eta *= 1.05

    def factorize_iterate(self):
        for m in range(self.nsteps):
            print(m, end='\r')
            delta_p = np.zeros((self.m.rt.shape[0], self.k))
            delta_q = np.zeros((self.m.rt.shape[1], self.k))

            for u, i in self.m.rt_not_nan_indices:
                error = np.dot(self.p[u, :], self.q[i, :]) - self.m.rt[u, i]
                for k in range(self.k):
                    delta_p[u, k] = error * self.p[u, k]
                    delta_q[i, k] = error * self.q[i, k]

            self.p -= 2 * self.eta * delta_p
            self.q -= 2 * self.eta * delta_q

            self.rmse.append(self.training_error())
            print(self.rmse[-1])
            if len(self.rmse) > 1:
                if abs(self.rmse[-1] - self.rmse[-2]) < self.tol:
                    break


class WeightedCFNN(CFNN):
    def __init__(self, m, k, nsteps=500, eta=0.00075, regularize=False,
                 init_sim=True, tol=1e-5):
        Recommender.__init__(self, m)
        self.k = k
        self.nsteps = nsteps
        self.eta = eta
        self.tol = tol
        self.regularize = regularize
        self.lamda = 0.05
        self.normalize = False
        if init_sim:
            self.w = np.copy(self.m.s)
        else:
            self.w = np.random.random((self.m.rt.shape[1], self.m.rt.shape[1]))
            # self.eta *= 5
        w_init = np.copy(self.w)

        print('init_sim =', init_sim)
        print('k =', k)
        print('lamda =', self.lamda)
        print('eta = ', self.eta)

        self.interpolate_weights()

        print('init_sim =', init_sim)
        print('k =', k)
        print('lamda =', self.lamda)
        print('eta = ', self.eta)

        diff = np.linalg.norm(w_init - self.w)

        self.plot_rmse('%.4f' % diff, suffix='svd' if init_sim else 'random')
        print('test error: %.4f' % self.test_error())

    # @profile
    def interpolate_weights(self):
        icount = self.m.rt.shape[1]

        rt_nan_indices = set(self.m.rt_nan_indices)
        ucount = self.m.rt.shape[0]
        m = self.m
        for step in xrange(self.nsteps):
            print(step, end='\r')
            delta_w_i_j = np.zeros((icount, icount))
            for i in xrange(icount):
                for u in xrange(ucount):
                    if (u, i) in rt_nan_indices:
                        continue
                    s_u_i = m.similar_items(u, i, self.k)
                    error = sum(self.w[i, k] * m.rt[u, k] for k in s_u_i) \
                        - m.rt[u, i]
                    for j in s_u_i:
                        delta_w_i_j[i, j] += error * m.rt[u, j]
                        if self.regularize:
                            delta_w_i_j[i, j] += self.lamda * self.w[i, j]
            self.w -= 2 * self.eta * delta_w_i_j
            self.rmse.append(self.training_error())
            print(step, 'eta = %.8f, rmse = %.8f' % (self.eta, self.rmse[-1]))
            if len(self.rmse) > 1:
                if abs(self.rmse[-1] - self.rmse[-2]) < self.tol:
                    break
                if self.rmse[-1] > self.rmse[-2]:
                    print('RMSE getting larger')
                    self.eta *= 0.5
                else:
                    self.eta *= 1.05


def read_movie_lens_data():
    import csv
    csvfile = open('u1.test', 'r')
    reader = csv.reader(csvfile, delimiter='\t')
    ratings = {}
    movies = set()
    for row in reader:
        user = row[0]
        movie = row[1]
        rating = row[2]
        if user not in ratings:
            ratings[user] = [(movie, rating)]
        else:
            ratings[user].append((movie, rating))
        movies.add(movie)

    m = list(movies)
    r = np.zeros([len(ratings), len(movies)])
    r.fill(np.nan)
    i = 0
    for user in ratings:
        uratings = ratings[user]
        for rating in uratings:
            r[i, m.index(rating[0])] = rating[1]
        i += 1

    return r


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    with open('m.obj', 'rb') as infile:  # complete MovieLens matrix
        m = pickle.load(infile).astype(float)
    m[m == 0] = np.nan

    # with open('m255.obj', 'rb') as infile: # sample of 255 from MovieLens
    #     m = pickle.load(infile).astype(float)
    # m[m == 0] = np.nan

    # m = read_movie_lens_data() # Denis's MovieLens sample

    # m = np.array([  # simple test case
    #     [5, 1, np.NAN, 2, 2, 4, 3, 2],
    #     [1, 5, 2, 5, 5, 1, 1, 4],
    #     [2, np.NAN, 3, 5, 4, 1, 2, 4],
    #     [4, 3, 5, 3, np.NAN, 5, 3, np.NAN],
    #     [2, np.NAN, 1, 3, np.NAN, 2, 5, 3],
    #     [4, 1, np.NAN, 1, np.NAN, 4, 3, 2],
    #     [4, 2, 1, 1, np.NAN, 5, 4, 1],
    #     [5, 2, 2, np.NAN, 2, 5, 4, 1],
    #     [4, 3, 3, np.NAN, np.NAN, 4, 3, np.NAN]
    # ])
    # hidden = np.array([
    #     [6, 2, 0, 2, 2, 5, 3, 0, 1, 1],
    #     [1, 2, 0, 4, 5, 3, 2, 3, 0, 4]
    # ])
    # um = UtilityMatrix(m, hidden=hidden)

    um = UtilityMatrix(m)
    # cfnn = CFNN(um, k=20)
    # f = Factors(um, k=5, eta=0.00001, regularize=True, init_svd=False)
    w = WeightedCFNN(um, k=15, eta=0.000001, regularize=True, init_sim=False)

    end_time = datetime.datetime.now()
    print('Duration: {}'.format(end_time - start_time))


