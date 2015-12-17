# -*- coding: utf-8 -*-


from __future__ import division, print_function
import graph_tool.all as gt
import io
import networkx as nx
import numpy as np
import os
import pdb
import unittest

from navigator import Mission, Strategy, SimilarityMatrix, Navigator, DataSet


class Test(unittest.TestCase):
    def get_graph(self, data_set='test_wiki'):
        # uses the graph from http://en.wikipedia.org/wiki/File:6n-graf.svg
        # fname = 'data/test/' + data_set + '/graphs/graph_5.txt'
        # if data_set == 'wiki' and not os.path.exists(fname):
        #     edges = [(1, 2), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5), (4, 6)]
        #     with io.open(fname, 'w', encoding='utf-8') as outfile:
        #         for edge in edges:
        #             outfile.write(unicode(edge[0]) + '\t' +
        #                           unicode(edge[1]) + '\n')
        #             outfile.write(unicode(edge[1]) + '\t' +
        #                           unicode(edge[0]) + '\n')

        from navigator import load_graph
        g = load_graph(os.path.join('data', data_set, 'graphs', 'graph_5.gt'))
        return g

    def get_mission(self, data_set='test_wiki'):
        fname = os.path.join('data', data_set, 'missions.txt')
        if data_set == 'wiki' and not os.path.exists(fname):
            missions = [[4, 1]]
            with io.open(fname, 'w', encoding='utf-8') as outfile:
                for miss in missions:
                    outfile.write('\t'.join(map(unicode, miss)) + '\n')

        with io.open(fname, encoding='utf-8') as infile:
            targets = []
            start = None
            for line in infile:
                if not line.strip():
                    continue
                t = line.strip().split('\t')
                if not start:
                    start = t[0]
                    t = t[1:]
                targets.append(t)
            miss = Mission(start, targets)
        return miss

    def get_matrix(self, data_set='test_wiki'):
        """Generate/read a simple similarity matrix.

        Similarity is defined as
        1 - (abs(n - m) / (n + m)),
        where n and m are integers (and also the labels of two graph nodes)
        """
        def save_text(fname, matrix, item2matrix):
            snodes = [int(n) for n in item2matrix.keys()]
            snodes = map(unicode, sorted(snodes))
            with io.open(fname, 'w', encoding='utf-8') as outfile:
                outfile.write(u'\t' + u'\t\t'.join(snodes) + u'\n')
                for n in snodes:
                    outfile.write(n)
                    for m in snodes:
                        val = u'%0.2f' % matrix[item2matrix[n], item2matrix[m]]
                        outfile.write(u'\t' + val)
                    outfile.write(u'\n')

        fname = 'data/' + data_set + '/matrices/title_matrix.npy'
        fname2 = 'data/' + data_set + '/matrices/graph_5.npy'
        fname3 = 'data/' + data_set + '/item2matrix.txt'
        if not os.path.exists(fname):
            if data_set == 'paper':
                clusters = [[1, 2, 3], [11, 12, 13, 14, 15],
                            [21, 22, 23, 24, 25], [31, 32, 33]]
                cd = {}
                for c in clusters:
                    for n in c:
                        cd[n] = c
            graph = self.get_graph(data_set)
            item2matrix = {}
            with io.open(fname3, 'w', encoding='utf-8') as outfile:
                snodes = [int(n) for n in graph.vertices()]
                snodes = map(unicode, sorted(snodes))
                for i, node in enumerate(snodes):
                    outfile.write(unicode(node) + '\t' + unicode(i) + '\n')
                    item2matrix[unicode(node)] = unicode(i)
            matrix = np.zeros((graph.num_vertices(), graph.num_vertices()))
            for i, n in enumerate(graph.vertices()):
                for j, m in enumerate(graph.vertices()):
                    if data_set == 'paper':
                        if int(n) in cd[int(m)]:
                            val = 1.00
                        else:
                            c_value = sum(cd[int(m)]) / len(cd[int(m)])
                            val = 1 - (abs(int(n)-c_value) / (int(n)+c_value))
                    else:
                        val = 1 - (abs(int(n)-int(m)) / (int(n)+int(m) + 0.1))
                    matrix[item2matrix[graph.vp['name'][n]], item2matrix[graph.vp['name'][m]]] = val
            np.save(fname, matrix)
            np.save(fname2, matrix)
            save_text(fname + '.txt', matrix, item2matrix)
        matrix = SimilarityMatrix(fname3, fname)
        return matrix

    def test_mission(self):
        miss = self.get_mission()
        self.assertFalse(miss.targets is miss.targets_original)
        self.assertEqual(miss.targets, miss.targets_original)

    def test_strategy_random(self):
        s = Strategy()
        graph = self.get_graph()
        miss = self.get_mission()

        successors = [s.find_next(graph, 'random', miss, u'4', u'3')
                      for n in xrange(100)]
        # assert that the list does not contain all the same items
        self.assertNotEqual(len(successors), successors.count(successors[0]))

        # changed to not include this feature in navigator.py
        # assert that the list does contain all the same items in this case
        # (target among the neighbors)
        # successors = [s.find_next(graph, 'random', miss, u'2', u'4')
        #               for n in xrange(100)]
        # self.assertEqual(len(successors), successors.count(successors[0]))

    def test_strategy_matrix(self):
        s = Strategy()
        graph = self.get_graph()
        miss = self.get_mission()
        matrix = self.get_matrix()
        self.assertEqual(s.find_next(graph, 'title', miss, u'4', None, matrix), u'3')
        self.assertEqual(s.find_next(graph, 'title', miss, u'3', None, matrix), u'2')
        self.assertEqual(s.find_next(graph, 'title', miss, u'6', None, matrix), u'4')

    def test_navigator(self):
        wiki = DataSet('wiki', 'data/test/wiki/', ['graph'], n_vals=[5])
        paper = DataSet('paper', 'data/test/paper/', ['graph'], n_vals=[5])
        nav = Navigator([wiki, paper])
        nav.run()
        for d in nav.data_sets:
            with io.open(d.folder + '/paths_compare.txt', encoding='utf-8')\
                    as infile:
                paths_compare = infile.read()
            with io.open(d.folder + '/paths.txt', encoding='utf-8') as infile:
                paths = infile.read()
            # compare the resulting paths
            # exclude the random paths from the comparison
            self.assertEqual(paths_compare[paths_compare.find('--------titl'):],
                             paths[paths.find('--------titl'):])

if __name__ == '__main__':
    def test_all():
        t = Test('test_mission')
        t.test_mission()
        t.test_strategy_random()
        t.test_strategy_matrix()
        # t.test_navigator()

    test_all()
