import numpy as np
import matplotlib.pylab as pylab
import math
import abc
import csv
import pdb

FIXED_RANDOM = False
if FIXED_RANDOM:
    np.random.seed(2014)
np.set_printoptions(linewidth=225)
debug_ = False


class UtilityMatrix:

    def __init__(self, r, hidden, k):
        self.r = r  # rating matrix (=utility matrix)
        self.hidden = hidden  # parts of rating matrix to hold out for training
        self.k = k  # number of latent factors

        self.mu_i = np.nanmean(r, axis=0)
        self.rc = r - self.mu_i
        self.rc[np.isnan(self.rc)] = 0.0
        self.s = np.corrcoef(self.rc.T)
        self.s[np.isnan(self.s)] = 0.0
        self.rt = self.training_data()
        self.rt_nnan_indices = self.get_rt_nnan_indices()

    def similar_items(self, u, i):
        r_u = self.r[u, :]  # user ratings
        s_i = np.copy(self.s[i, :])  # item similarity
        s_i[s_i < 0.0] = np.nan  # mask only to similar items
        s_i[i] = np.nan  # mask the item
        s_i[np.isnan(r_u)] = np.nan  # mask to items rated by the user
        nn = np.isnan(s_i).sum()  # how many invalid items

        s_i_sorted = np.argsort(s_i)
        s_i_k = s_i_sorted[-self.k-nn:-nn]
        return s_i_k

    def training_data(self):
        rt = np.copy(self.r)
        rt[self.hidden[0], self.hidden[1]] = np.nan
        return rt

    def get_rt_nnan_indices(self):
        nnan = np.where(~np.isnan(self.rt))
        nnan_indices = zip(nnan[0], nnan[1])
        np.random.shuffle(nnan_indices)
        return nnan_indices


class Recommender:

    __metaclass__ = abc.ABCMeta

    def __init__(self, m):
        self.m = m

    @abc.abstractmethod
    def predict(self, u, i):
        return

    def test_error(self):
        count = len(self.m.hidden)
        sse = 0.0
        for u, i in self.m.hidden.T:
            r_u_i = self.predict(u, i)
            err = self.m.r[u, i] - r_u_i
            sse += err ** 2

            if debug_:
                print "user:", u
                print "item:", i
                print "rating:", r_u_i
                print "error:", err
                print "--------------"

        rmse = (1.0 / count) * math.sqrt(sse)
        if debug_:
            print "rmse test data: ", rmse
            print "--------------"
        return rmse

    def training_error(self):
        ucount = self.m.rt.shape[0]
        icount = self.m.rt.shape[1]
        count = ucount * icount - np.isnan(self.m.rt).sum()
        sse = 0.0
        for u in range(ucount):
            for i in range(icount):
                if not np.isnan(self.m.rt[u, i]):
                    r_u_i = self.predict(u, i)
                    err = self.m.r[u, i] - r_u_i
                    sse += err ** 2

                    if debug_:
                        print "user:", u
                        print "item:", i
                        print "rating:", r_u_i
                        print "error:", err
                        print "--------------"

        rmse = (1.0 / count) * math.sqrt(sse)
        if debug_:
            print "rmse training data: ", rmse
            print "--------------"
        return rmse


class CFNN(Recommender):

    def __init__(self, m):
        Recommender.__init__(self, m)
        self.w = self.m.s
        self.normalize = True

    def predict(self, u, i):
        r = 0.0
        z = 0.0
        n_u_i = self.m.similar_items(u, i)
        if debug_:
            print "similar items:", n_u_i
        for j in n_u_i:
            r += self.m.rc[u, j] * self.w[i, j]
            if self.normalize:
                z += self.w[i, j]

        if r == 0:
            return self.m.mu_i[i]
        if self.normalize:
            return r / z + self.m.mu_i[i]
        return r + self.m.mu_i[i]


class WeightedCFNN(CFNN):

    def __init__(self, m, nsteps=500, eta=0.0002):
        Recommender.__init__(self, m)
        self.normalize = False
        self.nsteps = nsteps
        self.eta = eta
        self.interpolate_weights()

    def interpolate_weights(self):
        ucount = self.m.rt.shape[0]
        icount = self.m.rt.shape[1]
        self.w = np.copy(self.m.s)
        #self.w = np.random.random((icount, icount))

        js = []
        for i in range(icount):
                for u in range(ucount):
                    if not np.isnan(self.m.rt[u, i]):
                        n_u_i = self.m.similar_items(u, i)
                        for l in n_u_i:
                            js.append(l)
        js = list(set(js))

        self.rmse = []
        for m in range(self.nsteps):
            print m
            delta_w_i_j = np.zeros([icount, icount])
            for i in range(icount):
                for j in js:
                    for u in range(ucount):
                        if not np.isnan(self.m.rt[u, i]):
                            n_u_i = self.m.similar_items(u, i)
                            tmp = 0.0
                            for l in n_u_i:
                                tmp += self.w[i, l] * self.m.rc[u, l]
                            tmp -= self.m.rc[u, i]
                            if not np.isnan(self.m.rt[u, j]):
                                delta_w_i_j[i, j] += tmp * self.m.rc[u, j]

            self.w -= 2.0 * self.eta * delta_w_i_j

            self.rmse.append(self.training_error())
            if len(self.rmse) > 1:
                if abs(self.rmse[-1] - self.rmse[-2]) < 1e-5:
                    break


class Factors(Recommender):

    def __init__(self, m, k, nsteps=500, eta=0.02, regularize=False, newton=False, tol=1e-5):
        Recommender.__init__(self, m)
        self.k = k
        self.nsteps = nsteps
        self.eta = eta
        self.regularize = regularize
        self.newton = newton
        self.tol = tol
        self.factorize()

    def predict(self, u, i):
        p_u = self.p[u, :]
        q_i = self.q[i, :]
        return np.dot(p_u, q_i.T) + self.m.mu_i[i]

    def factorize(self):
        ucount = self.m.rt.shape[0]
        icount = self.m.rt.shape[1]

        # init randomly
        # self.p = np.random.random((ucount, self.k))
        # self.q = np.random.random((icount, self.k))

        # init by SVD
        m = np.copy(self.m.rt)
        m[np.where(np.isnan(m))] = 0
        ps, ss, vs = np.linalg.svd(m)
        qs = np.dot(np.diag(ss), vs)
        self.p = ps[:, :self.k]
        self.q = qs[:self.k, :].T

        lamb = 0.02

        self.rmse = []
        for m in range(self.nsteps):
            print m,

            masked = np.ma.array(self.m.rt - self.m.mu_i, mask=np.isnan(self.m.rt))
            delta_p = np.ma.dot(np.ma.dot(self.p, self.q.T) - masked, self.q)
            delta_q = np.ma.dot((np.ma.dot(self.p, self.q.T) - masked).T, self.p)

            if self.regularize:
                delta_p += lamb * self.p
                delta_q += lamb * self.q

            if self.newton:
                hess_i_p = np.linalg.inv(np.dot(self.q.T, self.q))
                hess_i_q = np.linalg.inv(np.dot(self.p.T, self.p))
                self.p -= 10 * self.eta * np.dot(delta_p, hess_i_p)
                self.q -= 10 * self.eta * np.dot(delta_q, hess_i_q)
            else:
                self.p -= 2.0 * self.eta * delta_p
                self.q -= 2.0 * self.eta * delta_q
            self.rmse.append(self.training_error())
            print self.rmse[-1]
            if len(self.rmse) > 1:
                if abs(self.rmse[-1] - self.rmse[-2]) < self.tol:
                    break

        if debug_:
            print self.p
            print self.q


class FactorsSGD(Factors):

    def __init__(self, m, k, nsteps=500, eta=0.02, regularize=False, newton=False, tol=1e-5):
        Factors.__init__(self, m, k, nsteps, eta, regularize, newton, tol)

    def factorize(self):
        ucount = self.m.rt.shape[0]
        icount = self.m.rt.shape[1]

        # init randomly
        # self.p = np.random.random((ucount, self.k))
        # self.q = np.random.random((icount, self.k))

        # init by SVD
        m = np.copy(self.m.rt)
        m[np.where(np.isnan(m))] = 0
        ps, ss, vs = np.linalg.svd(m)
        qs = np.dot(np.diag(ss), vs)
        self.p = ps[:, :self.k]
        self.q = qs[:self.k, :].T

        lamb = 0.02

        self.rmse = []
        for m in range(self.nsteps):
            print m

            counter = 0
            for u, i in self.m.rt_nnan_indices:
                for y in range(self.k):
                    error = np.dot(self.p[u, :], self.q.T[:, i]) - self.m.rc[u, i]
                    delta_p = error * self.q.T[y, i]
                    delta_q = error * self.p[u, y]
                    if self.regularize:
                        delta_p -= lamb * self.p[u, y]
                        delta_q -= lamb * self.q.T[y, i]
                    self.p[u, y] -= 2.0 * self.eta * delta_p
                    self.q[i, y] -= 2.0 * self.eta * delta_q

                if counter % 100 == 0:
                    self.rmse.append(self.training_error())
                counter += 1
            if len(self.rmse) > 1:
                if abs(self.rmse[-1] - self.rmse[-2]) < self.tol:
                    break

        if debug_:
            print self.p
            print self.q


class FactorsSGD2(Factors):

    def __init__(self, m, k, nsteps=500, eta=0.02, regularize=False, newton=False):
        Factors.__init__(self, m, k, nsteps, eta, regularize, newton)

    def factorize(self):
        ucount = self.m.rt.shape[0]
        icount = self.m.rt.shape[1]
        if FIXED_RANDOM:
            np.random.seed(2014)
        self.p = np.random.random((ucount, self.k))
        if FIXED_RANDOM:
            np.random.seed(2014)
        self.q = np.random.random((icount, self.k))

        lamb = 0.02

        self.rmse = []
        for m in range(self.nsteps):
            print m
            counter = 0
            for u, i in self.m.rt_nnan_indices:
                for y in range(self.k):
                    error = np.dot(self.p[u, :], self.q.T[:, i]) - self.m.rc[u, i]
                    delta_p = error * self.q.T[y, i]
                    delta_q = error * self.p[u, y]
                    if self.regularize:
                        delta_p -= lamb * self.p[u, y]
                        delta_q -= lamb * self.q.T[y, i]
                    self.p[u, y] -= 2.0 * self.eta * delta_p
                    self.q[i, y] -= 2.0 * self.eta * delta_q

                if counter % 100 == 0:
                    self.rmse.append(self.training_error())
                counter += 1
            if len(self.rmse) > 1:
                if abs(self.rmse[-1] - self.rmse[-2]) < self.tol:
                    break

        if debug_:
            print self.p
            print self.q


class Plotter(object):
    def __init__(self):
        self.fig = pylab.figure()

    def add_plot(self, rmse, label):
        pylab.plot(range(len(rmse)), rmse, label=label)

    def finish(self):
        pylab.legend()
        pylab.xlabel("iteration")
        pylab.ylabel("RMSE")
        pylab.show()


def plot_rmse(rmse, f, title):
    fig = pylab.figure(f)
    pylab.plot(range(len(rmse)), rmse)
    pylab.xlabel("iteration")
    pylab.ylabel("RMSE")
    pylab.title(title)
    fig.show()


def print_test_error(recommender, title):
    rmse = recommender.test_error()
    print title, " test rmse:", rmse


def read_movie_lens_data():
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


def movie_lens():
    data = read_movie_lens_data()
    um = UtilityMatrix(data, np.array([[458, 458], [209, 211]]), 2)
    print "training data"
    print um.rt
    print um.s

    cfnn = CFNN(um)
    factors = Factors(um, 2, eta=0.002)
    factors1 = Factors(um, 2, eta=0.002, regularize=True)
    factors2 = Factors(um, 2, eta=0.002, newton=True)
    factors3 = FactorsSGD(um, 2, eta=0.002)

    print "cfnn similarities"
    print cfnn.w
    print "factors p"
    print factors.p
    print "factors q.t"
    print factors.q.T
    print "factors1 p"
    print factors1.p
    print "factors1 q.t"
    print factors1.q.T
    print "factors2 p"
    print factors2.p
    print "factors2 q.t"
    print factors2.q.T
    print "factors3 p"
    print factors3.p
    print "factors3 q.t"
    print factors3.q.T

    print_test_error(cfnn, 'cfnn')
    print_test_error(factors, 'factors')
    print_test_error(factors1, 'factors1')
    print_test_error(factors2, 'factors2')
    print_test_error(factors3, 'factors3')
    print "-----------------"
    rmse = cfnn.training_error()
    print "cfnn training rmse:", rmse
    print "factors training rmse:", len(factors.rmse), factors.rmse[-1]
    print "factors1 training rmse:", len(factors1.rmse), factors1.rmse[-1]
    print "factors2 training rmse:", len(factors2.rmse), factors2.rmse[-1]
    print "factors3 training rmse:", len(factors3.rmse), factors3.rmse[-1]

    plot_rmse(factors.rmse, 1, 'RMSE (Matrix Factorization)')
    plot_rmse(factors1.rmse, 2, 'RMSE (Matrix Factorization) Regularized')
    plot_rmse(factors2.rmse, 3, 'RMSE (Matrix Factorization) Newton')
    plot_rmse(factors3.rmse, 4, 'RMSE (Matrix Factorization) SGD')
    pylab.show()


def toy():
    um = UtilityMatrix(
        np.array([
            [5, 1, np.NAN, 2, 2, 4, 3, 2],
            [1, 5, 2, 5, 5, 1, 1, 4],
            [2, np.NAN, 3, 5, 4, 1, 2, 4],
            [4, 3, 5, 3, np.NAN, 5, 3, np.NAN],
            [4, 1, np.NAN, 1, np.NAN, 4, 3, 2],
            [4, 2, 1, 1, np.NAN, 5, 4, 1],
            [5, 2, 2, np.NAN, 2, 5, 4, 1],
            [4, 3, 3, np.NAN, np.NAN, 4, 3, np.NAN]
        ]),
        np.array([
            [0, 4],
            [3, 1]
        ]),
        2
    )
    print "training data"
    print um.rt
    print um.s

    cfnn = CFNN(um)
    wcfnn = WeightedCFNN(um)
    factors = Factors(um, 2)

    print "cfnn similarities"
    print cfnn.w
    print "wcfnn weights"
    print wcfnn.w
    print "factors p"
    print factors.p
    print "factors q.t"
    print factors.q.T

    print_test_error(cfnn, 'cfnn')
    print_test_error(wcfnn, 'wcfnn')
    print_test_error(factors, 'factors')
    print "-----------------"
    rmse = cfnn.training_error()
    print "cfnn training rmse:", rmse
    print "wcfnn training rmse:", len(wcfnn.rmse), wcfnn.rmse[-1]
    print "factors training rmse:", len(factors.rmse), factors.rmse[-1]

    plot_rmse(wcfnn.rmse, 1, 'RMSE (Weighted CFNN)')
    plot_rmse(factors.rmse, 2, 'RMSE (Matrix Factorization)')
    pylab.show()


def toy2():
    um = UtilityMatrix(
        np.array([
            [5, 1, np.NAN, 2, 2, 4, 3, 2],
            [1, 5, 2, 5, 5, 1, 1, 4],
            [2, np.NAN, 3, 5, 4, 1, 2, 4],
            [4, 3, 5, 3, np.NAN, 5, 3, np.NAN],
            [4, 1, np.NAN, 1, np.NAN, 4, 3, 2],
            [4, 2, 1, 1, np.NAN, 5, 4, 1],
            [5, 2, 2, np.NAN, 2, 5, 4, 1],
            [4, 3, 3, np.NAN, np.NAN, 4, 3, np.NAN]
        ]),
        np.array([
            [0, 4],
            [3, 1]
        ]),
        2
    )
    ptr = Plotter()
    for approach, label in [
        (Factors(um, 2), 'Gradient Descent'),
        (FactorsSGD(um, 2), 'Stochastic Gradient Descent (original)'),
        (FactorsSGD2(um, 2), 'Gradient Descent (modified)'),
    ]:
        ptr.add_plot(approach.rmse, label)
    ptr.finish()


def movie_lens2():
    data = read_movie_lens_data()
    um = UtilityMatrix(data, np.array([[458, 458], [209, 211]]), 2)

    ptr = Plotter()
    for approach, label in [
        (Factors(um, 2, eta=0.002), 'Gradient Descent'),
        (FactorsSGD(um, 2, eta=0.002), 'Stochastic Gradient Descent (original)'),
        (FactorsSGD2(um, 2, eta=0.002), 'Gradient Descent (modified)'),
    ]:
        ptr.add_plot(approach.rmse, label)
    ptr.finish()

if __name__ == '__main__':
    # toy()
    # toy2()
    # movie_lens()
    movie_lens2()



