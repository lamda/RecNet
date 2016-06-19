# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import cPickle as pickle
import graph_tool.all as gt
import io
import itertools
import numpy as np
import os
import pdb
import random
import re
import shutil


class Graph(object):
    def __init__(self, dataset, fname='', graph=None, N=None, use_sample=False,
                 refresh=False, suffix=''):
        print(dataset, fname, N, 'use_sample =', use_sample,
              'refresh =', refresh)
        self.graph_folder = os.path.join('data', dataset, 'graphs')
        self.matrix_folder = 'matrix'
        self.stats_folder = os.path.join('data', dataset, 'stats')
        if not os.path.exists(self.stats_folder):
            os.makedirs(self.stats_folder)
        self.use_sample = use_sample
        self.graph_name = fname if not use_sample else fname + '_sample'
        self.graph_file_path = os.path.join(self.graph_folder,
                                            self.graph_name + '.txt')
        self.N = N
        self.gt_file_path = os.path.join(
            self.graph_folder,
            fname + suffix + '.gt'
        )
        self.stats_file_path = os.path.join(
            self.stats_folder,
            self.graph_name + '.obj'
        )
        self.graph = gt.Graph(directed=True)
        self.names = self.graph.new_vertex_property('string')
        lbd_add = lambda: self.graph.add_vertex()
        self.name2node = collections.defaultdict(lbd_add)

    def load_graph(self, graph=None, refresh=False):
        if graph is not None:
            self.graph = graph
            print('graph set directly')
        elif refresh:
            self.load_from_adjacency_list()
            self.save()
            print('graph loaded from adjacency list')
        else:
            try:
                self.load_from_file()
                print('graph loaded from .gt file')
            except IOError:
                self.load_from_adjacency_list()
                self.save()
                print('graph loaded from adjacency list')

    def load_from_file(self):
        self.graph = gt.load_graph(self.gt_file_path, fmt='gt')

    def get_all_nodes_from_adjacency_list(self):
        nodes = set()
        with io.open(self.graph_file_path, encoding='utf-8') as infile:
            for line in infile:
                node, nb = line.strip().split('\t')
                nodes.add(node)
                nodes.add(nb)
        return nodes

    def get_recommenders_from_adjacency_list(self):
        recommenders = set()
        with io.open(self.graph_file_path, encoding='utf-8') as infile:
            for index, line in enumerate(infile):
                recommenders.add(line.strip().split('\t')[0])
        return recommenders

    def load_nodes_from_adjacency_list(self):
        nodes = self.get_all_nodes_from_adjacency_list()
        for node in sorted(nodes):
            v = self.name2node[node]
            self.names[v] = node
        self.graph.vp['name'] = self.names

    def load_from_adjacency_list(self):
        self.load_nodes_from_adjacency_list()
        edges = []
        with io.open(self.graph_file_path, encoding='utf-8') as infile:
            nb_count = collections.defaultdict(int)
            for index, line in enumerate(infile):
                print(index + 1, end='\r')
                node, nb = line.strip().split('\t')
                if self.N is not None and nb_count[node] > (self.N - 1):
                    continue
                v = self.graph.vertex_index[self.name2node[node]]
                nb_count[node] += 1
                edges += [(v, self.graph.vertex_index[self.name2node[nb]])]
        self.graph.add_edge_list(edges)

    def save(self):
        self.graph.save(self.gt_file_path, fmt='gt')

    def compute_stats(self):
        print('computing stats...')
        stats = {}
        data = self.basic_stats()
        stats['graph_size'], stats['recommenders'], stats[
            'outdegree_av'] = data
        # stats['cc'] = self.clustering_coefficient()
        # stats['cp_size'], stats['cp_count'] = self.largest_component()
        stats['bow_tie'] = self.bow_tie()
        stats['bow_tie_changes'] = self.compute_bowtie_changes()
        if self.N in [5, 20]:
            stats['ecc_max'], stats['ecc_median'] = self.eccentricity()

        print('saving...')
        with open(self.stats_file_path, 'wb') as outfile:
            pickle.dump(stats, outfile, -1)
        print()

    def update_stats(self):
        with open(self.stats_file_path, 'rb') as infile:
            stats = pickle.load(infile)

        # data = self.basic_stats()
        # stats['graph_size'], stats['recommenders'], stats['outdegree_av'] = data
        # print(stats['cp_size'], stats['cp_size'] * stats['graph_size'] / 100,
        #       0.01 * stats['cp_size'] * stats['graph_size'] / 100)
        # print(100 * stats['recommenders'] / stats['graph_size'])
        # stats['cp_size'], stats['cp_count'] = self.largest_component()
        # stats['lc_ecc'] = self.eccentricity()
        # stats['cp_size'], stats['cp_count'] = self.largest_component()
        # print('SCC size:', stats['cp_size'] * self.graph.num_vertices())
        stats['bow_tie'] = self.bow_tie()
        stats['bow_tie_changes'] = self.compute_bowtie_changes()

        print('saving...')
        with open(self.stats_file_path, 'wb') as outfile:
            pickle.dump(stats, outfile, -1)
        print()

    def basic_stats(self):
        print('basic_stats():')
        graph_size = self.graph.num_vertices()
        recommenders = len(self.get_recommenders_from_adjacency_list())
        pm = self.graph.degree_property_map('out')
        outdegree_av = float(np.mean(pm.a[pm.a != 0]))
        print('    ', graph_size, 'nodes in graph')
        print('    ', recommenders, 'recommenders in graph')
        print('     %.2f average out-degree' % outdegree_av)
        return graph_size, recommenders, outdegree_av

    def clustering_coefficient(self, minimal_neighbors=2):
        print('clustering_coefficient()')
        clustering_coefficient = 0
        neighbors = {int(node): set([int(n) for n in node.out_neighbours()])
                     for node in self.graph.vertices()}
        for idx, node in enumerate(self.graph.vertices()):
            node = int(node)
            if len(neighbors[node]) < minimal_neighbors:
                continue
            edges = sum(len(neighbors[int(n)] & neighbors[node])
                        for n in neighbors[node])
            cc = edges / (len(neighbors[node]) * (len(neighbors[node]) - 1))
            clustering_coefficient += cc
        return clustering_coefficient / self.graph.num_vertices()

    def largest_component(self):
        print('largest_component()')
        component, histogram = gt.label_components(self.graph)
        return [
            100 * max(histogram) / self.graph.num_vertices(),
            len(histogram),
        ]

    def bow_tie(self):
        print('bow tie')

        component, histogram = gt.label_components(self.graph)
        label_of_largest_component = np.argmax(histogram)
        largest_component = (component.a == label_of_largest_component)
        lcp = gt.GraphView(self.graph, vfilt=largest_component)

        # Core, In and Out
        all_nodes = set(int(n) for n in self.graph.vertices())
        scc = set([int(n) for n in lcp.vertices()])
        scc_node = random.sample(scc, 1)[0]
        graph_reversed = gt.GraphView(self.graph, reversed=True)

        outc = np.nonzero(gt.label_out_component(self.graph, scc_node).a)[0]
        inc = np.nonzero(gt.label_out_component(graph_reversed, scc_node).a)[0]
        outc = set(outc) - scc
        inc = set(inc) - scc

        # Tubes, Tendrils and Other
        wcc_view = gt.GraphView(self.graph, directed=False)
        # wcc = gt.label_largest_component(self.graph, scc, directed=False).a
        # wcc = set(np.nonzero(wcc)[0])
        wcc = set(np.nonzero(gt.label_out_component(wcc_view, scc_node).a)[0])
        tube = set()
        out_tendril = set()
        in_tendril = set()
        other = all_nodes - wcc
        remainder = wcc - inc - outc - scc

        for idx, r in enumerate(remainder):
            print(idx + 1, '/', len(remainder), end='\r')
            predecessors = set(
                np.nonzero(gt.label_out_component(graph_reversed, r).a)[0])
            successors = set(
                np.nonzero(gt.label_out_component(self.graph, r).a)[0])
            if any(p in inc for p in predecessors):
                if any(s in outc for s in successors):
                    tube.add(r)
                else:
                    in_tendril.add(r)
            elif any(s in outc for s in successors):
                out_tendril.add(r)
            else:
                other.add(r)

        vp_bowtie = self.graph.new_vertex_property('string')
        for component, label in [
            (inc, 'IN'),
            (scc, 'SCC'),
            (outc, 'OUT'),
            (in_tendril, 'TL_IN'),
            (out_tendril, 'TL_OUT'),
            (tube, 'TUBE'),
            (other, 'OTHER')
        ]:
            for node in component:
                vp_bowtie[self.graph.vertex(node)] = label
        self.graph.vp['bowtie'] = vp_bowtie
        self.save()

        bow_tie = [inc, scc, outc, in_tendril, out_tendril, tube, other]
        bow_tie = [100 * len(x) / self.graph.num_vertices() for x in bow_tie]
        return bow_tie

    def compute_bowtie_changes(self):
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
        comp2num = {l: i for l, i in zip(labels, range(len(labels)))}
        if self.N == 1:
            return None
        elif 1 < self.N <= 5:
            prev_N = self.N - 1
        else:
            prev_N = self.N - 5
        prev_gt_file_path = self.gt_file_path.split('_')[0] +\
            '_' + unicode(prev_N) + '.gt'
        prev_graph = gt.load_graph(prev_gt_file_path, fmt='gt')

        changes = np.zeros((len(labels), len(labels)))
        for node in self.graph.vertices():
            c1 = comp2num[self.graph.vp['bowtie'][node]]
            try:
                c2 = comp2num[prev_graph.vp['bowtie'][node]]
            except KeyError:
                c2 = comp2num['OTHER']
            changes[c1, c2] += 1
        changes /= prev_graph.num_vertices()
        return changes

    def eccentricity(self, use_sample=False):
        component, histogram = gt.label_components(self.graph)
        label_of_largest_component = np.argmax(histogram)
        largest_component = (component.a == label_of_largest_component)
        graph_copy = self.graph.copy()
        lcp = gt.GraphView(graph_copy, vfilt=largest_component)
        lcp.purge_vertices()
        lcp.clear_filters()

        print('eccentricity() for lcp of', lcp.num_vertices(), 'vertices')
        ecc_max = collections.defaultdict(int)
        ecc_median = collections.defaultdict(int)
        vertices = [int(v) for v in lcp.vertices()]
        if use_sample:
            sample_size = int(0.15 * lcp.num_vertices())
            if sample_size == 0:
                sample_size = lcp.num_vertices()
            sample = random.sample(vertices, sample_size)
            vertices = sample
        for idx, node in enumerate(vertices):
            print(idx + 1, '/', len(vertices), end='\r')
            dist = gt.shortest_distance(lcp, source=node).a
            ecc_max[max(dist)] += 1
            ecc_median[int(np.median(dist))] += 1
        ecc_max = [ecc_max[i] for i in range(max(ecc_max.keys()) + 2)]
        ecc_median = [ecc_median[i] for i in range(max(ecc_median.keys()) + 2)]
        return ecc_max, ecc_median


def extract_recommendations():
    for dataset in [
        'movielens',
        'bookcrossing',
    ]:
        folder = os.path.join('data', dataset, 'graphs')
        for rec_type in [
            'rbmf',
            'rbiw'
        ]:
            for suffix in [
                '',
                '_resolved',
            ]:
                fpath_in = os.path.join(folder, rec_type + '_5' +
                                        suffix + '.txt')
                with io.open(fpath_in, encoding='utf-8') as infile:
                    data = infile.readlines()
                for N in range(1, 5):
                    fpath = os.path.join(folder, rec_type + '_' + str(N) +
                                         suffix + '.txt')
                    with io.open(fpath, 'w', encoding='utf-8') as outfile:
                        for idx, line in enumerate(data):
                            if (idx % 5) < N:
                                outfile.write(line)


def rename_selected():
    dataset = 'imdb'
    old_dir = os.path.join('data', dataset, 'graphs_selected')
    new_dir = os.path.join('data', dataset, 'graphs')
    files = [f for f in os.listdir(old_dir) if f.endswith('.txt')]
    try:
        os.makedirs(new_dir)
    except OSError:
        pass
    for f in files:
        print(f)
        match = re.search(r'([a-z]+_\d+)(_[a-z]|_[0-9]+)*(_resolved)?(\.txt)', f)
        if match is None:
            pdb.set_trace()
        groups = [g for g in match.group(1, 3, 4) if g]
        f_new = ''.join(groups)
        shutil.copyfile(os.path.join(old_dir, f), os.path.join(new_dir, f_new))


if __name__ == '__main__':
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    # extract_recommendations()
    rename_selected()
    # sys.exit()

    datasets = [
        # 'movielens',
        # 'bookcrossing',
        'imdb',
    ]
    rec_types = [
        # 'cb',
        # 'rbar',
        # 'rb',
        # 'rbiw',
        'rbmf',
    ]
    div_types = [
        '',
        # '_div_random',
        # '_div_diversify',
        # '_div_exprel'
    ]
    Ns = [
        1,
        2,
        3,
        4,
        5,
        10,
        15,
        20
    ]
    bowtie_stats = {}
    for dataset in datasets:
        for rec_type in rec_types:
            for N in Ns:
                for div_type in div_types:
                    fname = rec_type + '_' + unicode(N) + div_type
                    g = Graph(dataset=dataset, fname=fname, N=N,
                              use_sample=False, refresh=False)
                    g.load_graph()
                    g.compute_stats()
                    # g.update_stats()
