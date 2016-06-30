# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import cPickle as pickle
import io
import HTMLParser
import numpy as np
import os
import pandas as pd
import pdb
import random
import sklearn.cluster
import sklearn.feature_extraction.text
import sqlite3
import sys

DATA_BASE_FOLDER = 'data'


pd.set_option('display.width', 1000)


class ItemCollection(object):
    def __init__(self, dataset):
        print(dataset)
        self.dataset = dataset
        self.data_folder = os.path.join(DATA_BASE_FOLDER, self.dataset)
        self.dataset_folder = os.path.join(self.data_folder, 'dataset')
        self.graph_folder = os.path.join(self.data_folder, 'graphs')
        self.matrices_folder = os.path.join(self.data_folder, 'matrices')
        for folder in [self.graph_folder, self.matrices_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.db_file = os.path.join(self.data_folder, 'database_new.db')
        self.db_main_table = 'books' if dataset == 'bookcrossing' else 'movies'

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        stmt = """SELECT id FROM """ + self.db_main_table
        cursor.execute(stmt)
        ids = cursor.fetchall()
        try:
            self.ids = map(str, sorted(map(lambda x: int(x[0]), ids)))
        except ValueError:
            self.ids = map(str, sorted(map(lambda x: x[0], ids)))

    def write_clusters_title_matrix(self, random_based=True):
        print('write_clusters_title_matrix()')
        file_suffix = '_random' if random_based else ''
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        limit = ""

        # get the categories
        stmt = """SELECT id, name FROM categories"""
        cursor.execute(stmt)
        result = cursor.fetchall()
        id2cat = {t[0]: t[1] for t in result}

        # get the titles
        stmt = """SELECT id, original_title FROM """ + \
               self.db_main_table + """ ORDER BY id ASC""" + limit
        cursor.execute(stmt)
        result = cursor.fetchall()
        # replace the year in the title by the decade (e.g., 1995 --> 1990)
        title2id = {t[1][:-2] + '0': t[0] for t in result}
        # id2titleshort = {t[0]: t[1][:-7] for t in result}
        id2titleshort = {t[0]: t[1] for t in result}
        # titleshort2id = {t[1][:-7]: t[0] for t in result}
        titleshort2id = {t[1]: t[0] for t in result}
        id2year = {t[0]: t[1][-5:-2] + '0' for t in result}

        # get the item-category relations
        stmt = """SELECT item_id, cat_id FROM item_cat ORDER BY id ASC"""
        cursor.execute(stmt)
        cats = cursor.fetchall()
        # t = set(type(c[0]) for c in cats)
        cats = [(a, b) for (a, b) in cats if a in id2year]

        # get the complete strings (title, category and decade)
        # data = collections.defaultdict(str)
        # for t, id in title2id.items():
        #     data[id] = t
        # for id, c in cats:
        #     data[id] += ' ' + id2cat[c]
        #
        # data = sorted(data.items(), key=operator.itemgetter(0))
        # data = [t[1] for t in data]
        #
        # # strip (),.! a.k.a.
        # # data_new = []
        # # for d in data:
        # #     d_new = d
        # #     for repl in ['(', ')', ',', 'a.k.a.', ':']:
        # #         d_new = d_new.replace(repl, '')
        # #     data_new.append(d_new)
        # # data = data_new
        stmt = """SELECT id, wp_text FROM """ + \
               self.db_main_table + """ ORDER BY id ASC """ + limit
        cursor.execute(stmt)
        result = cursor.fetchall()
        h = HTMLParser.HTMLParser()
        data = [h.unescape(r[1].replace('\n', ' ')) for r in result]

        # strip (),.! a.k.a.
        data_new = []
        for d in data:
            d_new = d
            for repl in ['(', ')', ',', 'a.k.a.', ':']:
                d_new = d_new.replace(repl, '')
            data_new.append(d_new)
        data = data_new

        # categorize/cluster items
        clusters = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        ids = self.ids
        try:
            ids = sorted(map(int, ids))
        except ValueError:
            ids = sorted(ids)

        def write_cluster(clusters, fname):
            f = os.path.join(self.data_folder, fname + file_suffix + '.txt')
            fr = os.path.join(self.data_folder, fname + file_suffix + '_resolved.txt')
            with io.open(f, encoding='utf-8', mode='w') as outfile, \
                    io.open(fr, encoding='utf-8', mode='w') as outfile_res:
                lens = []
                years = sorted(clusters.keys())
                for y in years:
                    # outfile.write(y + u'\n')
                    outfile_res.write(y + u'\n')
                    for c in sorted(clusters[y].keys()):
                        if not clusters[y][c]:
                            continue
                        # outfile.write(c + u'\t')
                        outfile_res.write(u' '.join(c) + u' (' +
                                          unicode(len(clusters[y][c])) +
                                          u')' + u'\t')
                        lens.append(len(clusters[y][c]))
                        titles = [id2titleshort[t] for t in clusters[y][c]]
                        if titles:
                            line = u'\t'.join(map(unicode, clusters[y][c]))
                            line_res = u'\t'.join(titles)
                        else:
                            line, line_res = u'\t', u'\t'
                        outfile.write(line + u'\n')
                        outfile_res.write(line_res + u'\n')
                    outfile.write(u'\n')
                    outfile_res.write(u'\n')
                for l in sorted(lens):
                    outfile_res.write(unicode(l) + '\t')

        def write_cluster_ratingbased(clusters, fname):
            fname += file_suffix
            f = os.path.join(self.data_folder, fname + '.txt')
            fr = os.path.join(self.data_folder, fname + '_resolved.txt')
            with io.open(f, encoding='utf-8', mode='w') as outfile, \
                    io.open(fr, encoding='utf-8', mode='w') as outfile_res:
                lens = []
                for cluster in clusters:
                    outfile_res.write('(' + unicode(len(cluster)) + ')\t')
                    lens.append(len(cluster))
                    titles = [id2titleshort[t] for t in cluster]
                    line = u'\t'.join(map(unicode, cluster))
                    line_res = u'\t'.join(titles)
                    outfile.write(line + u'\n')
                    outfile_res.write(line_res + u'\n')
                    outfile.write(u'\n')
                    outfile_res.write(u'\n')
                for l in sorted(lens):
                    outfile_res.write(unicode(l) + '\t')

        mission_limit = 1145
        pairs = set()
        # numpairs = len(ids) * (len(ids) - 1) if len(ids) < 35 else mission_limit
        numpairs = mission_limit

        if random_based:
            # select missions randomly
            while len(pairs) < numpairs:
                pairs.add(tuple(random.sample(ids, 2)))

            # cluster by genre and year
            # select clusters based on similarity
            item2cats = collections.defaultdict(list)
            erroneous = set()
            for id, c in cats:
                if id in id2year and len(item2cats[id]) < 3:  # TODO
                    item2cats[id].append(id2cat[c])
                # else:
                #     erroneous.add((id, id2titleshort[id], tuple(item2cats[id])))
            for err in erroneous:
                print(err)
            # pdb.set_trace()
            # Why are there so many excluded (<3) and what happens to them later?
            for k in item2cats:
                item2cats[k] = frozenset(item2cats[k])
            for id, c in item2cats.items():
                year = id2year[id]
                clusters[year][c].add(id)
            write_cluster(clusters, 'clusters')

            years = sorted(clusters.keys())
            cluster_items = []
            for y in years:
                for c in sorted(clusters[y].keys()):
                    if not clusters[y][c]:
                        continue
                    tids = sorted([t for t in clusters[y][c]])
                    cluster_items.append(tids)

            cd = {}
            for c in cluster_items:
                for n in c:
                    cd[n] = c
            selected_clusters = [c for c in cluster_items if 4 <= len(c) <= 30]

            perm = []
            while len(perm) < mission_limit:
                m = random.sample(selected_clusters, 4)
                if m not in perm:
                    perm.append(m)

        else:
            # select missions based on corated items
            # get coratings
            fname = 'RatingBasedRecommender_um_sparse.obj.npy'
            fpath = os.path.join('data', dataset, 'recommendation_data', fname)
            um = np.load(fpath)
            if not um.shape:
                um = um.item()
            um.data = np.ones(um.data.shape[0])
            coratings = um.T.dot(um)
            coratings.setdiag(0)

            cs = np.cumsum(coratings.toarray())
            cs /= cs[-1]

            id2dataset_id = {idx: id_val for idx, id_val in enumerate(ids)}

            # compute missions
            while len(pairs) < numpairs:
                # pairs.add(tuple(random.sample(ids, 2)))
                idx = cs.searchsorted(np.random.random(), 'right')
                pairs.add(np.unravel_index(idx, coratings.shape))

            pairs = [(id2dataset_id[t[0]], id2dataset_id[t[1]]) for t in pairs]

            # cluster rating-based with k-means
            # select clusters based on similarity
            kmeans_init = 'random' if dataset == 'movielens' else 'kmeans-++'
            km = sklearn.cluster.KMeans(
                n_clusters=int(len(ids)/3),
                init=kmeans_init,
                n_jobs=-4,
                max_iter=500,
                precompute_distances=True,
                n_init=100,
                verbose=False,
            )
            km.fit(um.T)
            # with open('kmeans.obj', 'wb') as outfile:
            #     pickle.dump(km, outfile, -1)
            # with open('kmeans.obj', 'rb') as infile:
            #     km = pickle.load(infile)
            labels = km.predict(um.T)
            clusters = [[] for _ in range(max(labels)+1)]
            for idx, val in enumerate(labels):
                clusters[val].append(id2dataset_id[idx])
            write_cluster_ratingbased(clusters, 'clusters')
            selected_clusters = [c for c in clusters if 4 <= len(c) <= 30]
            selected_cluster_ids = [cidx for cidx, c in enumerate(clusters)
                                    if 4 <= len(c) <= 30]
            cd = {}
            for c in clusters:
                for n in c:
                    cd[n] = c

            perm = []
            for cidx in selected_cluster_ids:
                cc = km.cluster_centers_[cidx]
                cc_tiled = np.tile(cc, (km.cluster_centers_.shape[0], 1))
                dists = np.sqrt(np.sum(
                    (km.cluster_centers_ - cc_tiled) ** 2, axis=1)
                )
                mission = np.argsort(dists)[:4]
                perm.append([clusters[i] for i in mission])
                if len(perm) >= mission_limit:
                    break

        # write missions
        fpath = os.path.join(self.data_folder, 'missions' + file_suffix + '.txt')
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            for p in pairs:
                outfile.write(unicode(p[0]) + u'\t' + unicode(p[1]) + u'\n')

        if_missions = []
        for cluster in selected_clusters:
            for s_node in cluster:
                if_missions.append([s_node] + cluster)
        if_selected_missions = random.sample(if_missions, min(mission_limit, len(if_missions)))

        fpath = os.path.join(self.data_folder, 'missions_if' + file_suffix + '.txt')
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            for mission in if_selected_missions:
                outfile.write(u'\t'.join(map(unicode, mission)) + u'\n')

        fpath = os.path.join(self.data_folder, 'missions_bp' + file_suffix + '.txt')
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            for p in perm:
                for ind, c in enumerate(p):
                    if ind == 0:
                        outfile.write(unicode(c[0]) + u'\t')
                    else:
                        outfile.write(u'*\t')
                    for node in c:
                        outfile.write(unicode(node) + u'\t')
                    outfile.write(u'\n')

        fpath = os.path.join(self.data_folder, 'item2matrix.txt')
        with open(fpath, 'w') as outfile:
            item2matrix = {str(m): i for i, m in enumerate(ids)}
            for k in sorted(item2matrix.keys()):
                outfile.write(k + '\t' + str(item2matrix[k]) + '\n')

        m, title_matrix = self.get_tfidf_cluster_matrix(ids, data, cd)
        mpath = os.path.join(self.data_folder, 'matrices')
        if not os.path.isdir(mpath):
            os.makedirs(mpath)
        np.save(os.path.join(mpath, 'title_matrix' + file_suffix), m)
        np.save(os.path.join(mpath, 'title_matrix_c' + file_suffix), title_matrix)

    def get_tfidf_cluster_matrix(self, ids, data, cd, simple=False):
        """
        build the TF-IDF similarity matrix of the titles
        build the clustered version of that matrix
            --> similarities to cluster centroids
        """
        movie2matrix = {m: i for i, m in enumerate(ids)}

        title_matrix = self.get_tf_idf_similarity(data, simple=simple)
        print('building cluster matrix')
        m = np.zeros((len(ids), len(ids)))
        for row, i in enumerate(ids):
            print('\r', row + 1, '/', len(ids), end='')
            for col, j in enumerate(ids):
                if i in cd[j]:
                    m[row, col] = 1.00
                else:
                    m[row, col] = sum(title_matrix[row, movie2matrix[k]]
                                      for k in cd[j]) / len(cd[j])
        print()
        return [m, title_matrix]

    def get_tf_idf_similarity(self, data, max_features=50000,
                              simple=False):
        import nltk
        print('counting...')

        class LemmaTokenizer(object):
            """
            lemmatizer (scikit-learn.org/dev/modules/feature_extraction.html
                          #customizing-the-vectorizer-classes)
            """

            def __init__(self):
                self.wnl = nltk.WordNetLemmatizer()

            def __call__(self, doc):
                return [self.wnl.lemmatize(t) for t in
                        nltk.word_tokenize(doc)]

        path_stopw = os.path.join(DATA_BASE_FOLDER, 'stopwords.txt')
        stopw = [l.strip() for l in
                 io.open(path_stopw, encoding='utf-8-sig')]

        if simple:
            cv = sklearn.feature_extraction.text.CountVectorizer()
        else:
            cv = sklearn.feature_extraction.text.CountVectorizer(
                stop_words=stopw,
                tokenizer=LemmaTokenizer(),
                max_features=max_features
            )
        counts = cv.fit_transform(data)

        print('getting TF-IDF')
        v = sklearn.feature_extraction.text.TfidfTransformer()
        v = v.fit_transform(counts)
        v_dense = v.todense()
        similarity = np.array(v_dense * v_dense.T)  # cosine similarity
        return similarity

    def write_network_neighbors_matrix(self):
        print('write_network_neighbors_matrix()')

        path = os.path.join(self.data_folder, 'graphs')
        graphs = set(f[:f.rfind('.')] + '.txt'
                     for f in os.listdir(path) if 'resolved' not in f)
        graphs = [
            'rb_10.txt',
            'rb_10_div_diversify.txt',
            'rb_10_div_exprel.txt',
            'rb_10_div_random.txt',
            'rb_15.txt',
            'rb_15_div_diversify.txt',
            'rb_15_div_exprel.txt',
            'rb_15_div_random.txt',
            'rb_20.txt',
            'rb_20_div_diversify.txt',
            'rb_20_div_exprel.txt',
            'rb_20_div_random.txt',
            'rb_5.txt',
            'rb_5_div_diversify.txt',
            'rb_5_div_exprel.txt',
            'rb_5_div_random.txt',
            'rbar_10.txt',
            'rbar_10_div_diversify.txt',
            'rbar_10_div_exprel.txt',
            'rbar_10_div_random.txt',
            'rbar_15.txt',
            'rbar_15_div_diversify.txt',
            'rbar_15_div_exprel.txt',
            'rbar_15_div_random.txt',
            'rbar_20.txt',
            'rbar_20_div_diversify.txt',
            'rbar_20_div_exprel.txt',
            'rbar_20_div_random.txt',
            'rbar_5.txt',
            'rbar_5_div_diversify.txt',
            'rbar_5_div_exprel.txt',
            'rbar_5_div_random.txt',
            'rbiw_10.txt',
            'rbiw_10_div_diversify.txt',
            'rbiw_10_div_exprel.txt',
            'rbiw_10_div_random.txt',
            'rbiw_15.txt',
            'rbiw_15_div_diversify.txt',
            'rbiw_15_div_exprel.txt',
            'rbiw_15_div_random.txt',
            'rbiw_20.txt',
            'rbiw_20_div_diversify.txt',
            'rbiw_20_div_exprel.txt',
            'rbiw_20_div_random.txt',
            'rbiw_5.txt',
            'rbiw_5_div_diversify.txt',
            'rbiw_5_div_exprel.txt',
            'rbiw_5_div_random.txt',
            'rbmf_10.txt',
            'rbmf_10_div_diversify.txt',
            'rbmf_10_div_exprel.txt',
            'rbmf_10_div_random.txt',
            'rbmf_15.txt',
            'rbmf_15_div_diversify.txt',
            'rbmf_15_div_exprel.txt',
            'rbmf_15_div_random.txt',
            'rbmf_20.txt',
            'rbmf_20_div_diversify.txt',
            'rbmf_20_div_exprel.txt',
            'rbmf_20_div_random.txt',
            'rbmf_5.txt',
            'rbmf_5_div_diversify.txt',
            'rbmf_5_div_exprel.txt',
            'rbmf_5_div_random.txt',
        ]
        for index, g in enumerate(graphs):
            print(index + 1, '/', len(graphs), ': ', g)
            fg = os.path.join(self.data_folder, 'graphs', g)
            fc = os.path.join(self.data_folder, 'clusters.txt')
            fi = os.path.join(self.data_folder, 'item2matrix.txt')
            m_cos, m_clus = self.calculate_clustered_cosine_matrix(fg, fc, fi)
            dirpath = os.path.join(self.data_folder, 'matrices')
            if not os.path.isdir(dirpath):
                os.makedirs(dirpath)
            np.save(os.path.join(dirpath, g.split('.')[0]), m_cos)
            np.save(os.path.join(dirpath, g.split('.')[0] + '_c'), m_clus)

    def calculate_clustered_cosine_matrix(self, graph_file, cluster_file,
                                          item2matrix_file, matrix_cos=None):
        item2matrix = {}
        with io.open(item2matrix_file, encoding='utf-8-sig') as infile:
            for line in infile:
                if not line.strip():
                    continue
                l, r = line.strip().split('\t')
                item2matrix[l] = int(r)

        cd = {}
        with io.open(cluster_file, encoding='utf-8-sig') as infile:
            for line in infile:
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                for p in parts:
                    cd[p] = parts
        # item2matrix = {k: v for k, v in item2matrix.items() if k in cd}
        matrix_clus = np.ones((len(item2matrix), len(item2matrix)))
        if matrix_cos is None:
            matrix_cos = self.calculate_cosine_matrix(graph_file)

        cd_new = {}
        for k, v in cd.items():
            cd_new[item2matrix[k]] = [item2matrix[m] for m in v]
        cd = cd_new
        clusters = []
        for k, v in cd.items():
            if v not in clusters:
                clusters.append(v)

        for index, n in enumerate(clusters):
            for jindex, m in enumerate(clusters):
                if m == n:
                    continue
                for a in n:
                    v = sum(matrix_cos[a, b] for b in m) / len(m)
                    for b in m:
                        matrix_clus[a, b] = v
        return [matrix_cos, matrix_clus]

    def calculate_cosine_matrix(self, graph_file):
        import networkx as nx

        def load_graph(graph_file):
            """ load a graph from a tab-separated edge list
            """
            graph = nx.DiGraph()
            with io.open(graph_file, encoding='utf-8-sig') as in_file:
                for line in in_file:
                    u, v = line.strip().split('\t')
                    graph.add_edge(u, v)
            return graph

        graph = load_graph(graph_file)
        adj = nx.adjacency_matrix(graph, sorted(graph.nodes())).todense()
        adj = np.array(adj)
        num = np.dot(adj, adj)
        sc = np.sum(adj, 0)
        sr = np.sum(adj, 1)
        denom = np.sqrt(np.outer(sr, sc))
        with np.errstate(all='ignore'):  # suppresses division by zero warnings
            cs = num / denom
        cs[np.isnan(cs)] = 0.0
        return cs

    def calculate_cosine_matrix_inoutlinks(self, inlinks, outlinks):
            items = set(inlinks.keys()) | set(outlinks.keys())
            for i in items:
                inlinks[i] = set(inlinks[i])
                outlinks[i] = set(outlinks[i])
            ids = set(inlinks.keys()) | set(outlinks.keys())
            ids = map(unicode, sorted(map(int, ids)))
            cs = np.zeros((len(ids), len(ids)))
            for ind, i in enumerate(ids):
                if (ind + 1) % 100:
                    print('\r', ind + 1, '/', len(ids), end='')
                for jnd, j in enumerate(ids):
                    denom = np.sqrt(len(outlinks[i]) * len(inlinks[j]))
                    if not denom:
                        cs[ind, jnd] = 0.0
                    num = len(outlinks[i] & inlinks[j])
                    if num:
                        cs[ind, jnd] = num / denom
                    else:
                        cs[ind, jnd] = 0.0
            return cs


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in [
        'bookcrossing',
        'movielens',
        'imdb',
    ]:
        print('dataset not supported')
        sys.exit()
    dataset = sys.argv[1]

    ic = ItemCollection(dataset=dataset)
    ic.write_clusters_title_matrix(random_based=True)
    ic.write_clusters_title_matrix(random_based=False)
    # ic.write_network_neighbors_matrix()
