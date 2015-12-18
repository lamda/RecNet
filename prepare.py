# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import io
import numpy as np
import cPickle as pickle
import operator
import os
import pandas as pd
import pdb
import random
import sklearn.feature_extraction.text
import sqlite3

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

        self.db_file = os.path.join(self.data_folder, 'database.db')
        self.db_main_table = 'movies' if dataset == 'movielens' else 'books'

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        stmt = """SELECT id FROM """ + self.db_main_table
        cursor.execute(stmt)
        ids = cursor.fetchall()
        try:
            self.ids = map(str, sorted(map(lambda x: int(x[0]), ids)))
        except ValueError:
            self.ids = map(str, sorted(map(lambda x: x[0], ids)))

    def write_clusters_title_matrix(self):
        print('write_clusters_title_matrix()')
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
        stmt = """SELECT item_id, cat_id FROM item_cat ORDER BY item_id ASC"""
        cursor.execute(stmt)
        cats = cursor.fetchall()
        # t = set(type(c[0]) for c in cats)
        cats = [(a, b) for (a, b) in cats if a in id2year]

        # get the complete strings (title, category and decade)
        data = collections.defaultdict(str)
        for t, id in title2id.items():
            data[id] = t
        for id, c in cats:
            data[id] += ' ' + id2cat[c]

        data = sorted(data.items(), key=operator.itemgetter(0))
        data = [t[1] for t in data]

        # strip (),.! a.k.a.
        data_new = []
        for d in data:
            d_new = d
            for repl in ['(', ')', ',', 'a.k.a.', ':']:
                d_new = d_new.replace(repl, '')
            data_new.append(d_new)
        data = data_new

        # categorize/cluster items by release date and genre
        clusters = collections.defaultdict(
            lambda: collections.defaultdict(set))
        ids = self.ids
        try:
            ids = sorted(map(int, ids))
        except ValueError:
            ids = sorted(ids)

        def write_cluster(clusters, id2cat, fname):
            f = os.path.join(self.data_folder, fname + '.txt')
            fr = os.path.join(self.data_folder, fname + '_resolved.txt')
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
                        outfile_res.write(c + u' (' +
                                          unicode(len(clusters[y][c])) +
                                          u')' + u'\t')
                        lens.append(len(clusters[y][c]))
                        tids = [titleshort2id[t] for t in clusters[y][c]]
                        if tids:
                            line = u'\t'.join(map(unicode, tids))
                            line_res = u'\t'.join(clusters[y][c])
                        else:
                            line, line_res = u'\t', u'\t'
                        outfile.write(line + u'\n')
                        outfile_res.write(line_res + u'\n')
                    outfile.write(u'\n')
                    outfile_res.write(u'\n')
                for l in sorted(lens):
                    outfile_res.write(unicode(l) + '\t')

        item2cats = collections.defaultdict(list)
        for id, c in cats:
            if id in id2year:
                item2cats[id].append(id2cat[c])
        for k in item2cats:
            item2cats[k] = ' '.join(item2cats[k])
        for id, c in item2cats.items():
            year = id2year[id]
            clusters[year][item2cats[id]].add(id2titleshort[id])
        write_cluster(clusters, id2cat, 'clusters')

        years = sorted(clusters.keys())
        cluster_items = []
        for y in years:
            for c in sorted(clusters[y].keys()):
                if not clusters[y][c]:
                    continue
                tids = sorted([titleshort2id[t] for t in clusters[y][c]])
                cluster_items.append(tids)

        cd = {}
        for c in cluster_items:
            for n in c:
                cd[n] = c

        selected_clusters = [c for c in cluster_items if 3 <= len(c) <= 30]
        mission_limit = 1200
        pairs = set()
        numpairs = len(ids) * (len(ids) - 1) if len(ids) < 35 else mission_limit
        while len(pairs) < numpairs:
            pairs.add(tuple(random.sample(ids, 2)))

        fpath = os.path.join(self.data_folder, 'missions.txt')
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            for p in pairs:
                outfile.write(unicode(p[0]) + u'\t' + unicode(p[1]) + u'\n')

        if_missions = []
        for cluster in selected_clusters:
            for s_node in cluster:
                if_missions.append([s_node] + cluster)
        if_selected_missions = random.sample(if_missions, mission_limit)

        fpath = os.path.join(self.data_folder, 'missions_if.txt')
        with io.open(fpath, 'w', encoding='utf-8') as outfile:
            for mission in if_selected_missions:
                outfile.write(u'\t'.join(map(unicode, mission)) + u'\n')

        perm = []
        while len(perm) < mission_limit:
            m = random.sample(selected_clusters, 4)
            if m not in perm:
                perm.append(m)

        fpath = os.path.join(self.data_folder, 'missions_bp.txt')
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

        [m, title_matrix] = self.get_tfidf_cluster_matrix(ids, data, cd)
        mpath = os.path.join(self.data_folder, 'matrices')
        if not os.path.isdir(mpath):
            os.makedirs(mpath)
        np.save(os.path.join(mpath, 'title_matrix'), m)
        np.save(os.path.join(mpath, 'title_matrix_c'), title_matrix)

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
        graph_types = set(f[:f.rfind('_')]
                          for f in os.listdir(path)
                          if 'resolved' not in f)
        graphs = [g + '_' + unicode(N) + '.txt' for N in [5, 10]
                  for g in graph_types]
        graphs = graphs[31:]
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

    def write_wp_neighbors_matrix(self):
        print('write_wp_neighbors_matrix()')
        # import wikidump
        # conn = sqlite3.connect(self.db_file)
        # cursor = conn.cursor()
        # stmt = """SELECT id, wp_id FROM """ + self.db_main_table
        # cursor.execute(stmt)
        # ids = cursor.fetchall()
        # id2wp_id = {unicode(i[0]): unicode(i[1]) for i in ids}
        # ids = id2wp_id.values()
        # inlinks, outlinks = wikidump.wiki.get_links(ids)  # 10.632.934 lines
        # with open('wikidump/id_links_' + self.dataset + '.obj',
        #           'wb') as outfile:
        #     pickle.dump([inlinks, outlinks], outfile, -1)
        with open('wikidump/id_links_' + self.dataset + '.obj',
                  'rb') as infile:
            [inlinks, outlinks] = pickle.load(infile)
        g = 'wp_neighbors'
        m_cos = self.calculate_cosine_matrix_inoutlinks(inlinks, outlinks)
        fc = os.path.join(self.data_folder, 'clusters.txt')
        fi = os.path.join(self.data_folder, 'item2matrix.txt')
        m_clus = self.calculate_clustered_cosine_matrix(None, fc, fi, m_cos)[1]
        np.save(os.path.join(self.data_folder, 'matrices', g.split('.')[0]), m_cos)
        np.save(os.path.join(self.data_folder, 'matrices', g.split('.')[0] + '_c'),
                m_clus)

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
    ic = ItemCollection(dataset='movielens')
    # ic.write_clusters_title_matrix()
    # ic.write_network_neighbors_matrix()
    ic.write_wp_neighbors_matrix()
