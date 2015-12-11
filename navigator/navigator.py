# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import os
import copy
import io
import itertools
import operator
import random
import cPickle as pickle
import pdb

import matplotlib
matplotlib.use('Agg')
matplotlib.rc('pdf', fonttype=42)
import graph_tool.all as gt
import numpy as np
import matplotlib.pyplot as plt

do_debug = False
if do_debug:
    with open('ttid2title.obj', 'rb') as infile:
        ttid2title = pickle.load(infile)
else:
    ttid2title = collections.defaultdict(int)


def print_debug(*text):
    """wrapper for the print(function that can be turned on and off"""
    if do_debug:
        for t in text:
            print(t, end=' ')
        print()


def load_graph(fpath):
    """load a graph from a tab-separated edge list file"""
    graph = gt.load_graph(fpath, fmt='gt')
    name2node = {graph.vp['name'][node]: node for node in graph.vertices()}
    graph.name2node = name2node
    return graph


class SimilarityMatrix(object):
    """
    Class representing a similarity matrix for graph node similarities.

    The class holds a numpy matrix, the indices of which are masked by a dict,
    to allow the indexing with graph node descriptors in any format (e.g.,
     string).
    """
    def __init__(self, item2matrix_file, matrix_file):
        item2matrix = dict()
        with io.open(item2matrix_file, encoding='utf-8') as infile:
            for line in infile:
                u, v = line.strip().split('\t')
                item2matrix[u] = int(v)
        self.item2matrix = item2matrix

        with open(matrix_file, 'rb') as infile:
            self.cluster2sims = pickle.load(infile)

    def __getitem__(self, key):
        item, targets = key[0], key[1]
        s = self.cluster2sims[targets][self.item2matrix[item], 0]
        return s


class MissionCollection(object):
    """This class represents a collection (set) of missions, i.e., navigation
    problems the simulation should undertake."""
    def __init__(self, missions):
        self.missions = missions
        self.stats = None
        self.stretch = None

    def __iter__(self):
        return iter(self.missions)

    def compute_stats(self):
        """compute statistics for the missions
        call after all missions have been simulated"""
        if self.stats is not None:
            return
        self.stats = np.zeros(Navigator.steps_max + 1)
        for m in self.missions:
            m.compute_stats()
            self.stats += m.stats
        self.stats /= len(self.missions)


class Mission(object):
    """This class represents a Point-to-Point Search mission"""
    # mission types
    missions = [u'Information Foraging']

    def __init__(self, start, targets):
        self.steps = 0
        self.path = []
        self.start = start
        self.visited = set()
        self.targets = targets
        self.targets_original = [frozenset(t) for t in copy.deepcopy(targets)]
        self.stats = None

    def add(self, node):
        """add the current node to the path"""
        self.steps += 1
        self.path.append(node)
        self.visited.add(node)
        if node in self.targets[0]:
            self.targets[0].remove(node)

    def is_active(self):
        """check if the mission is still within its limits"""
        if self.steps > Navigator.steps_max or not self.targets[0]:
            return False
        return True

    def reset(self):
        """reset the visited nodes and go to the next target set"""
        self.visited = set()
        del self.targets[0]

    def compute_stats(self):
        self.stats = np.zeros(Navigator.steps_max + 1)
        try:
            ind = self.path.index(self.targets_original[0][0])
            self.stats[ind:] = 1.0
        except ValueError:
            pass


class IFMission(Mission):
    """This class represents an Information Foraging mission"""
    def __init__(self, start, targets):
        super(IFMission, self).__init__(start, targets)

    def compute_stats(self):
        self.stats = np.zeros(Navigator.steps_max + 1)
        targets = list(copy.deepcopy(self.targets_original[0]))
        curr = 0
        for i, n in enumerate(self.path[:len(self.stats)]):
            if n in targets:
                targets.remove(n)
                curr += 1
            self.stats[i] = curr
        if i < len(self.stats):
            self.stats[i:] = curr


class BPMission(Mission):
    """This class represents a Berrypicking mission"""
    def __init__(self, start, targets):
        super(BPMission, self).__init__(start, targets)

    def add(self, node):
        self.steps += 1
        self.path.append(node)
        self.visited.add(node)
        if node in self.targets[0]:
            self.targets[0] = []

    def is_active(self):
        if self.steps > Navigator.steps_max or not self.targets[0]:
            return False
        return True

    def compute_stats(self):
        self.stats = np.zeros(Navigator.steps_max + 1)
        self.path = self.path[2:]
        if self.path[-2:] == ['*', '*']:
            self.path = self.path[:-2]
        diff = len(self.path) - 2*self.path.count(u'*') - Navigator.steps_max-1
        if diff > 0:
            self.path = self.path[:-diff]
        path = ' '.join(self.path).split('*')
        path = [l.strip().split(' ') for l in path]
        path = [path[0]] + [p[1:] for p in path[1:]]
        path = path[:len(self.stats)]
        curr = 0
        pset = 0
        del self.targets_original[0]
        for p in path:
            self.stats[pset:pset+len(p)] = curr
            pset += len(p)
            curr += (1 / len(self.targets_original))
        ind = sum(len(p) for p in path) - 1
        if ind < len(self.stats):
            self.stats[ind:] = self.stats[ind-1]


class Strategy(object):
    """This class represents a strategy for choosing the next hop.
    During missions, thhe find_next method is called to select the next node
    """
    strategies = [
        # u'random',
        u'title',
        # u'optimal'
    ]

    def __init__(self):
        pass

    @staticmethod
    def find_next(graph, strategy, mission, node, parent_node=None,
                  matrix=None):
        """Select the next node to go to in a navigation mission"""
        print_debug('strategy =', strategy)
        graph_node = graph.name2node[node]
        nodes = [graph.vp['name'][n] for n in graph_node.out_neighbours()]
        if strategy == 'random':
            print('random strategy')
            if parent_node and parent_node not in nodes:
                nodes.append(parent_node)
            return random.choice(nodes)
        neighbor_targets = [n for n in nodes if n in mission.targets[0]]
        if neighbor_targets:
            print_debug('target in neighbors')
            return neighbor_targets[0]

        nodes = [n for n in nodes if n not in mission.visited]
        candidates = {n: matrix[n, mission.targets_original[0]] for n in nodes}
        if not candidates:
            chosen_node = None  # abort search
            print_debug('aborting search')
        else:
            chosen_node = max(candidates.iteritems(),
                              key=operator.itemgetter(1))[0]
            if do_debug:
                print_debug('candidates are:')
                for k, v in candidates.items():
                    print_debug(k, ttid2title[k], ':', v)
        if chosen_node == parent_node:
            print_debug('backtracking to node', parent_node)
            return None
        if do_debug:
            print_debug('going to ', ttid2title[chosen_node] if chosen_node else 'None')
            pdb.set_trace()
        return chosen_node


class DataSet(object):
    mission_folder = 'missions'

    def __init__(self, label, rec_types, div_types, Ns):
        self.label = label
        self.folder_graphs = os.path.join('..', label, 'graphs')
        self.rec_types = rec_types
        self.Ns = Ns
        self.graphs = {}
        for rec_type in self.rec_types:
            self.graphs[rec_type] = [
                os.path.join(
                    self.folder_graphs,
                    rec_type + '_top_n_' + d + str(N) + '.gt')
                for N in self.Ns
                for d in div_types
            ]
        self.sim_matrix = SimilarityMatrix(
            os.path.join('..', self.label, 'item2matrix.txt'),
                                           'cluster2sims.obj')

        # Structure: self.matrices[rec_type][graph][strategy]
        # Structure: self.missions[rec_type][graph][strategy][scenario]
        self.matrices = {}
        self.missions = {}
        for rec_type in self.rec_types:
            self.matrices[rec_type] = {}
            self.missions[rec_type] = {}
            for graph in self.graphs[rec_type]:
                self.matrices[rec_type][graph] = {}
                self.missions[rec_type][graph] = {}
                # self.matrices[rec_type][graph]['random'] = None
                # self.matrices[rec_type][graph]['optimal'] = None
                self.matrices[rec_type][graph]['title'] = self.sim_matrix
                for strategy in Strategy.strategies:
                    self.missions[rec_type][graph][strategy] = {}
                    mpath = os.path.join(
                        DataSet.mission_folder,
                        'missions_' + graph.split(os.sep)[-1].split('.')[0] +
                        '.txt'
                    )
                    m_if = self.load_missions(IFMission, mpath)
                    missions = {u'Information Foraging': m_if}
                    for m in missions:
                        mc = copy.deepcopy(missions[m])
                        self.missions[rec_type][graph][strategy][m] = mc

    def load_missions(self, mission_class, mission_file):
        with io.open(mission_file, encoding='utf-8') as infile:
            missions = []
            for line in infile:
                parts = line.strip().split('\t')
                start = parts[0]
                targets = [parts[1:]]
                missions.append(mission_class(start, targets))
            m = MissionCollection(missions)
        return m


class Navigator(object):
    steps_max = 50

    def __init__(self, data_sets):
        self.data_sets = data_sets

    def run(self):
        """run the simulations for optimal, random and the three types of
         background knowledge
        """
        print('    strategies...')
        matrix_c = None
        # run for all but the optimal version
        for data_set in self.data_sets:
            for rec_type in data_set.graphs:
                for graph in data_set.graphs[rec_type]:
                    print('        ', graph)
                    gt_graph = load_graph(graph)
                    for strategy in Strategy.strategies:
                        if strategy == 'optimal':
                            continue
                        # print('            ', strategy)
                        print_debug(strategy)
                        matrix_c = data_set.matrices[rec_type][graph][strategy]
                        for miss in data_set.missions[rec_type][graph][strategy]:
                            for m in data_set.missions[rec_type][graph][strategy][miss]:
                                for ti in xrange(len(m.targets_original)):
                                    start = m.path[-2] if m.path else m.start
                                    print_debug('++++' * 16, 'mission', ti, '/',
                                          len(m.targets_original))
                                    print_debug(m.targets_original[ti])
                                    self.navigate(gt_graph, strategy, m, start,
                                                  None, matrix_c)
                                    if not (ti + 1) == len(m.targets_original):
                                        m.path.append(u'*')
                                    m.reset()

        # run the simulations for the optimal solution
        # print('    optimal...')
        # for data_set in self.data_sets:
        #     for rec_type in data_set.graphs:
        #         for graph in data_set.graphs[rec_type]:
        #             print('        ', graph)
        #             gt_graph = load_graph(graph)
        #             for miss in data_set.missions[rec_type][graph]['optimal']:
        #                 dist = collections.defaultdict(int)
        #                 for m in data_set.missions[rec_type][graph]['optimal'][miss]:
        #                     for ti in xrange(len(m.targets_original)):
        #                         start = m.path[-2] if m.path else m.start
        #                         print_debug('++++' * 16, 'mission', ti, '/',
        #                               len(m.targets_original))
        #                         print_debug(m.targets_original[ti])
        #                         if miss == u'Greedy Search':
        #                             # TODO: compute shortest path with graph-tool
        #                             if m.targets_original[ti][0] in sp[start]:
        #                                 s = sp[start][m.targets_original[ti][0]]
        #                                 dist[s] += 1
        #                             else:
        #                                 dist[-1] += 1
        #                         self.optimal_path(gt_graph, m, start, sp)
        #                         if not (ti + 1) == len(m.targets_original):
        #                             m.path.append(u'*')
        #                         m.reset()

        # write the results to a file
        self.write_paths()

    def optimal_path(self, graph, mission, start, sp):
        """write a fake path to the mission, that is of the correct length
        if more evaluations such as the set of visited nodes are needed,
        this needs to be extended
        """
        mission.add(start)
        while mission.targets[0] and mission.is_active():
            ds = [sp[start][t] for t in mission.targets[0] if t in sp[start]]
            if not ds:
                mission.add(u'-1')  # target not connected --> fill with dummies
                continue
            for i in range(min(ds) - 1):
                mission.add(u'0')
            ind = ds.index(min(ds))
            start = mission.targets[0][ind]
            mission.add(mission.targets[0][ind])

    def navigate(self, graph, strategy, mission, node, parent_node=None,
                 matrix=None):
        if do_debug:
            print_debug('-' * 32 + '\n')
            print_debug('navigate called with', node, ttid2title[node],
                  '(parent: ',
                  ttid2title[parent_node] if parent_node else 'None', ')')
            print_debug('targets:')
            for m in mission.targets[0]:
                print_debug('   ', m, ttid2title[m])
            pdb.set_trace()
        if not mission.is_active() or node == -1 and not parent_node:
            print_debug('aborting')
            return
        mission.add(node)

        # recurse
        while mission.is_active():
            out_node = Strategy.find_next(graph, strategy, mission, node,
                                          parent_node, matrix)
            print_debug('choosing node', out_node)
            if not out_node:  # backtracking
                if parent_node:
                    mission.add(parent_node)
                return
            self.navigate(graph, strategy, mission, out_node, node, matrix)

    def write_paths(self):
        for data_set in self.data_sets:
            with io.open('paths.txt', 'w', encoding='utf-8') as outfile:
                outfile.write(u'----' * 16 + u'\n')
                for rec_type in data_set.graphs:
                    outfile.write(u'----' * 16 + u' ' + rec_type + u'\n')
                    for graph in data_set.graphs[rec_type]:
                        outfile.write(u'----' * 8 + u' ' + graph + u'\n')
                        for strategy in Strategy.strategies:
                            outfile.write(u'----' * 4 + strategy + u'\n')
                            for miss in [u'Information Foraging']:
                                outfile.write(u'----' * 2 + miss + u'\n')
                                stras = data_set.missions[rec_type][graph][strategy][miss]
                                for m in stras:
                                    outfile.write(u'\t'.join(m.path) + u'\n')


class PlotData(object):
    def __init__(self):
        self.missions = {}


class Evaluator(object):
    """Class responsible for calculating stats and plotting the results"""
    def __init__(self):
        try:
            with open('data_sets_new.obj', 'rb') as infile:
                print('loading...')
                self.data_sets = pickle.load(infile)
            print('loaded')
        except (IOError, EOFError):
            print('loading failed... computing from scratch')
            print('loading complete dataset...')
            with open('data_sets.obj', 'rb') as infile:
                self.data_sets = pickle.load(infile)
            print('loaded')
            self.compute()

        if not os.path.isdir('plots/'):
            os.makedirs('plots/')
        self.sc2abb = {u'Information Foraging': u'if'}
        self.colors = ['#FFA500', '#FF0000', '#0000FF', '#05FF05', '#000000']
        self.hatches = ['', 'xxx', '///', '---']

    def compute(self):
        print('computing...')
        data_sets_new = []
        for data_set in self.data_sets:
            pt = PlotData()
            pt.label = data_set.label
            pt.folder_graphs = data_set.folder_graphs
            for i, rec_type in enumerate(data_set.missions):
                pt.missions[rec_type] = {}
                for j, g in enumerate(div_types):
                    graph = data_set.folder_graphs + '/' + rec_type + \
                            g + '.gt'
                    pt.missions[rec_type][graph] = {}
                    for strategy in ['title']:
                        pt.missions[rec_type][graph][strategy] = {}
                        for scenario in [u'Information Foraging']:
                            print_debug(rec_type, graph, strategy, scenario)
                            m = data_set.missions[rec_type][graph][strategy][scenario]
                            m.compute_stats()
                            pt.missions[rec_type][graph][strategy][scenario] = m.stats
            data_sets_new.append(pt)
        print('saving to disk...')
        with open('data_sets_new.obj', 'wb') as outfile:
            pickle.dump(data_sets_new, outfile, -1)
        self.data_sets = data_sets_new

    def plot_bar(self):
        print('plot_bar()')

        # plot the legend in a separate plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        patches = [ax.bar([0], [0]) for i in range(4)]
        for pidx, p in enumerate(patches):
            p[0].set_fill(False)
            p[0].set_edgecolor(self.colors[pidx])
            p[0].set_hatch(self.hatches[pidx])
        figlegend = plt.figure(figsize=(7.75, 0.465))
        figlegend.legend(patches, ['No Diversification', 'ExpRel', 'Diversify', 'Random'], ncol=4)
        fig.subplots_adjust(left=0.19, bottom=0.06, right=0.91, top=0.92,
                            wspace=0.34, hspace=0.32)
        # plt.show()
        figlegend.savefig('plots/nav_legend.pdf')

        # plot the scenarios
        bars = None
        for sind, scenario in enumerate([u'Information Foraging']):
            print_debug('\n', scenario)
            for dind, data_set in enumerate(self.data_sets):
                fig, axes = plt.subplots(1, len(div_types), figsize=(13, 3.25),
                                         squeeze=False)
                for nidx, rec_type in enumerate(rec_types):
                    print_debug(data_set.label, rec_type)
                    ax = axes[0, nidx]
                    bar_vals = []
                    for didx, div_type in enumerate(div_types):
                        print_debug('    ', rec_type)
                        graph = data_set.folder_graphs + '/' + rec_type +\
                                div_type + '.gt'
                        stats = data_set.missions[rec_type][graph]['title'][scenario]
                        bar_vals.append(stats[-1])
                    x = np.arange(len(div_types))
                    bars = ax.bar(x, bar_vals)
                    for bidx, bar in enumerate(bars):
                        bar.set_fill(False)
                        bar.set_hatch(self.hatches[bidx])
                        bar.set_edgecolor(self.colors[bidx])

                    ax.set_title(rec_type[:2] + ' (' + rec_type[3:] + ')')
                    ax.set_ylabel('Found nodes')
                    ax.set_ylim(0, 5)
                    ax.set_xlim([-0.25, None])
                    ax.set_xticks([])

                fig.subplots_adjust(left=0.06, bottom=0.14, right=0.98,
                                    top=0.88, wspace=0.38, hspace=0.32)
                # plt.show()
                plt.savefig('plots/nav_success_rate.pdf')


rec_types = [
    'cb',
    'rb',
    'rbmf',
    'rbar'
]

div_types = [
    '',
    'div_random_',
    'div_diversify_',
    'div_exprel_'
]

Ns = [
    5,
    10
]


if __name__ == '__main__':
    movies = DataSet('movielens', rec_types, div_types, Ns)
    # nav = Navigator([movies])
    # print('running...')
    # nav.run()
    # with open('data_sets.obj', 'wb') as outfile:
    #     pickle.dump([movies], outfile, -1)
    # try:
    #     os.remove('data_sets_new.obj')
    #     print('deleted')
    # except OSError:
    #     pass

    evaluator = Evaluator()
    evaluator.plot_bar()

