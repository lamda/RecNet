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
import matplotlib.pyplot as plt
try:
    import graph_tool.all as gt
except ImportError:
    pass  # useful to import stuff from this script in other scripts
import numpy as np
# import matplotlib.pyplot as plt


def debug(*text):
    """wrapper for the print function that can be turned on and off"""
    if False:
        print(' '.join(str(t) for t in text))


def load_graph(fpath):
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
        self.matrix = np.load(matrix_file)

    def __getitem__(self, key):
        return self.matrix[self.item2matrix[key[0]], self.item2matrix[key[1]]]


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
            self.stats += 100 * m.stats
        self.stats /= len(self.missions)


class Mission(object):
    """This class represents a Point-to-Point Search mission"""
    # mission types
    missions = [u'Greedy Search', u'Berrypicking', u'Information Foraging']

    def __init__(self, start, targets):
        self.steps = 0
        self.path = []
        self.start = start
        self.visited = set()
        self.targets = targets
        self.targets_original = copy.deepcopy(targets)
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
        targets = copy.deepcopy(self.targets_original[0])
        i = targets.index(self.path[0])
        del targets[i]
        curr = 0
        for i, n in enumerate(self.path[:len(self.stats)]):
            if n in targets:
                targets.remove(n)
                curr += (1 / len(self.targets_original[0]))
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
        u'random',
        u'title',
        u'neighbors',
        # u'wp_neighbors',  # not working anymore for now
        u'optimal'
    ]

    def __init__(self):
        pass

    @staticmethod
    def find_next(graph, strategy, mission, node, parent_node=None,
                  matrix=None):
        """Select the next node to go to in a navigation mission"""
        debug('strategy =', strategy)
        node_gt = graph.name2node[node]
        nodes_gt = [n for n in node_gt.out_neighbours()]
        nodes = [graph.vp['name'][n] for n in nodes_gt]
        debug('nodes =', nodes)
        if strategy == 'random':
            if parent_node is not None and parent_node not in nodes:
                nodes.append(parent_node)
            return random.choice(nodes)
        neighbor_targets = [n for n in nodes if n in mission.targets[0]]
        if neighbor_targets:
            debug('target in neighbors')
            return neighbor_targets[0]

        nodes = [n for n in nodes if n not in mission.visited]
        try:
            candidates = {n: matrix[n, mission.targets[0][0]] for n in nodes}
        except KeyError:
            pdb.set_trace()
        if not candidates:
            chosen_node = None  # abort search
        else:
            chosen_node = max(candidates.iteritems(),
                              key=operator.itemgetter(1))[0]
            debug('candidates are:')
            for k, v in candidates.items():
                debug(k, ':', v)
        if chosen_node == parent_node:
            debug('backtracking to node', parent_node)
            return None
        debug('going to ', chosen_node)
        return chosen_node


class DataSet(object):
    def __init__(self, label, rec_types, div_types):
        self.label = label
        self.base_folder = os.path.join('..', 'data', self.label)
        self.folder_graphs = os.path.join(self.base_folder, 'graphs')
        self.folder_matrices = os.path.join(self.base_folder, 'matrices')
        self.n_vals = n_vals
        self.rec_types = rec_types
        self.graphs = {}
        for rec_type in self.rec_types:
            self.graphs[rec_type] = [
                os.path.join(
                    self.folder_graphs,
                    rec_type + '_' + unicode(N) + d + '.gt')
                for N in self.n_vals
                for d in div_types
            ]
            self.compute_shortest_path_lengths(self.graphs[rec_type])

        m_ptp = self.load_missions(Mission, u'missions.txt')
        m_if = self.load_missions(IFMission, u'missions_if.txt')
        m_bp = self.load_missions(BPMission, u'missions_bp.txt')
        missions = {u'Greedy Search': m_ptp, u'Information Foraging': m_if,
                    u'Berrypicking': m_bp}

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
                self.matrices[rec_type][graph]['random'] = [None, None]
                self.matrices[rec_type][graph]['optimal'] = [None, None]
                self.matrices[rec_type][graph]['title'] =\
                    [os.path.join(self.folder_matrices, 'title_matrix.npy'),
                     os.path.join(self.folder_matrices, 'title_matrix_c.npy')]
                self.matrices[rec_type][graph]['wp_neighbors'] =\
                    [os.path.join(self.folder_matrices, 'wp_neighbors.npy'),
                     os.path.join(self.folder_matrices, 'wp_neighbors_c.npy')]
                self.matrices[rec_type][graph]['neighbors'] =\
                    [os.path.join(self.folder_matrices, rec_type + '_' + str(N) + '.npy'),
                     os.path.join(self.folder_matrices, rec_type + '_' + str(N) + '_c.npy')]
                for strategy in Strategy.strategies:
                    self.missions[rec_type][graph][strategy] = {}
                    for m in missions:
                        mc = copy.deepcopy(missions[m])
                        self.missions[rec_type][graph][strategy][m] = mc

    def compute_shortest_path_lengths(self, graph_files):
        for i, gfile in enumerate(graph_files):
            print(gfile, i + 1, '/', len(graph_files))
            sp_file = gfile.rsplit('.', 1)[0] + '.npy'
            if os.path.exists(sp_file):
                print('    file exists!')
                continue
            print('    computing...')
            graph = load_graph(gfile)
            vertices = [n for n in graph.vertices()]
            dist = gt.shortest_distance(graph)
            d_max = np.iinfo(np.int32).max  # graph-tool uses this to mean inf
            sp = {}
            for vidx, vertex in enumerate(vertices):
                print('       ', vidx+1, '/', len(vertices), end='\r')
                dists = zip(vertices, dist[vertex].a)
                dists = {graph.vp['name'][v]: d for v, d in dists if d < d_max}
                sp[graph.vp['name'][vertex]] = dists
            with open(sp_file, 'wb') as outfile:
                pickle.dump(sp, outfile, -1)
        print('done computing path lengths')

    def load_missions(self, mission_class, mission_file):
        fpath = os.path.join(self.base_folder, mission_file)
        with io.open(fpath, encoding='utf-8') as infile:
            missions = []
            start = None
            targets = []
            for line in infile:
                parts = line.strip().split('\t')
                if line[0] != '*':
                    if start:
                        missions.append(mission_class(start, targets))
                    start = parts[0]
                    targets = [parts[1:]]
                else:
                    parts = line.strip().split('\t')
                    targets.append(parts[1:])
            missions.append(mission_class(start, targets))
            m = MissionCollection(missions)
        return m


class Navigator(object):
    steps_max = 50

    def __init__(self, data_set):
        self.data_set = data_set

    def run(self):
        """run the simulations for optimal, random and the three types of
         background knowledge
        """
        print('    strategies...')
        matrix_file = ''
        # run for all but the optimal version
        item2matrix = os.path.join(self.data_set.base_folder, 'item2matrix.txt')
        for rec_type in self.data_set.graphs:
            for graph in self.data_set.graphs[rec_type]:
                print('        ', graph)
                gt_graph = load_graph(graph)
                for strategy in Strategy.strategies:
                    if strategy == 'optimal':
                        continue
                    debug(strategy)
                    m_new = self.data_set.matrices[rec_type][graph][strategy][0]
                    m_newc = self.data_set.matrices[rec_type][graph][strategy][1]
                    if not m_new:
                        matrix_s, matrix_c = None, None
                    elif matrix_file != m_new:
                        matrix_s = SimilarityMatrix(item2matrix, m_new)
                        matrix_c = SimilarityMatrix(item2matrix, m_newc)
                        matrix_file = m_new
                    for miss in self.data_set.missions[rec_type][graph][strategy]:
                        if miss in ['Information Foraging', 'Berrypicking']:
                            matrix = matrix_c
                        else:
                            matrix = matrix_s
                        for m in self.data_set.missions[rec_type][graph][strategy][miss]:
                            for ti in xrange(len(m.targets_original)):
                                start = m.path[-2] if m.path else m.start
                                debug('++++' * 16, 'mission', ti, '/',
                                      len(m.targets_original))
                                debug(m.targets_original[ti])
                                self.navigate(gt_graph, strategy, m, start,
                                              None, matrix)
                                if not (ti + 1) == len(m.targets_original):
                                    m.path.append(u'*')
                                m.reset()

        # run the simulations for the optimal solution
        print('    optimal...')
        for rec_type in self.data_set.graphs:
            for graph in self.data_set.graphs[rec_type]:
                print('        ', graph)
                gt_graph = load_graph(graph)
                sp_file = graph.rsplit('.', 1)[0] + '.npy'
                with open(sp_file, 'rb') as infile:
                    sp = pickle.load(infile)
                for miss in self.data_set.missions[rec_type][graph]['optimal']:
                    dist = collections.defaultdict(int)
                    for m in self.data_set.missions[rec_type][graph]['optimal'][miss]:
                        for ti in xrange(len(m.targets_original)):
                            start = m.path[-2] if m.path else m.start
                            debug('++++' * 16, 'mission', ti, '/',
                                  len(m.targets_original))
                            debug(m.targets_original[ti])
                            if miss == u'Greedy Search':
                                if m.targets_original[ti][0] in sp[start]:
                                    s = sp[start][m.targets_original[ti][0]]
                                    dist[s] += 1
                                else:
                                    dist[-1] += 1
                            self.optimal_path(gt_graph, m, start, sp)
                            if not (ti + 1) == len(m.targets_original):
                                m.path.append(u'*')
                            m.reset()

        # write the results to a file
        self.write_paths()
        self.save()

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
        debug('-' * 32 + '\n')
        debug('navigate called with', node, '(parent: ', parent_node, ')')
        debug(mission.targets[0])
        if not mission.is_active() or node == -1 and not parent_node:
            debug('aborting')
            return
        mission.add(node)

        # recurse
        while mission.is_active():
            out_node = Strategy.find_next(graph, strategy, mission, node,
                                          parent_node, matrix)
            debug('choosing node', out_node)
            if not out_node:  # backtracking
                if parent_node:
                    mission.add(parent_node)
                return
            self.navigate(graph, strategy, mission, out_node, node, matrix)

    def write_paths(self):
        fpath = os.path.join(self.data_set.base_folder, 'paths.txt')
        with open(fpath, 'w') as outfile:
            outfile.write('----' * 16 + ' ' + self.data_set.base_folder + '\n')
            for rec_type in self.data_set.graphs:
                outfile.write('----' * 16 + ' ' + rec_type + '\n')
                for graph in self.data_set.graphs[rec_type]:
                    outfile.write('----' * 8 + ' ' + graph + '\n')
                    for strategy in Strategy.strategies:
                        outfile.write('----' * 4 + strategy + '\n')
                        for miss in ['Greedy Search',
                                     'Berrypicking',
                                     'Information Foraging']:
                            outfile.write('----' * 2 + miss + '\n')
                            stras = self.data_set.missions[rec_type][graph][strategy][miss]
                            for m in stras:
                                outfile.write('\t'.join(m.path) + '\n')

    def save(self):
        with open('data_sets_' + self.data_set.label + '.obj', 'wb') as outfile:
            pickle.dump([self.data_set], outfile, -1)


class PlotData(object):
    def __init__(self):
        self.missions = {}


class Evaluator(object):
    """Class responsible for calculating stats and plotting the results"""
    def __init__(self, datasets):
        global div_types
        self.data_sets = []
        for dataset in datasets:
            try:
                with open('data_sets_' + dataset + '_new.obj', 'rb') as infile:
                    print('loading...')
                    self.data_sets.append(pickle.load(infile)[0])
                print('loaded')
            except (IOError, EOFError):
                print('loading failed... computing from scratch (%s)' % dataset)
                with open('data_sets_' + dataset + '.obj', 'rb') as infile:
                    data_set = pickle.load(infile)[0]
                data_set_new = self.compute(label=dataset, data_set=data_set)
                self.data_sets.append(data_set_new)
                print('saving to disk...')
                with open('data_sets_' + dataset + '_new.obj', 'wb') as outfile:
                    pickle.dump([data_set_new], outfile, -1)

        if not os.path.isdir('plots'):
            os.makedirs('plots')
        self.sc2abb = {u'Greedy Search': u'ptp',
                       u'Information Foraging': u'if',
                       u'Berrypicking': u'bp'}
        self.colors = ['#FFA500', '#FF0000', '#0000FF', '#05FF05', '#000000']
        self.hatches = ['----', '/', 'xxx', '///', '---']
        self.linestyles = ['-', '--', ':', '-.']
        self.graphs = {
            'RB': ['rb_' + str(c) for c in n_vals],
            'MF': ['rbmf_' + str(c) for c in n_vals],
            'AR': ['rbar_' + str(c) for c in n_vals],
            'IW': ['rbiw_' + str(c) for c in n_vals],
        }
        self.graph_labels = {
            'CF': ['CF (' + str(c) + ')' for c in n_vals],
            'MF': ['MF (' + str(c) + ')' for c in n_vals],
            'AR': ['AR (' + str(c) + ')' for c in n_vals],
            'IW': ['IW (' + str(c) + ')' for c in n_vals],
        }
        self.graph_order = ['AR', 'CF', 'IW', 'MF']
        self.rec_type2label = {
            'rb': 'CF',
            'rbmf': 'MF',
            'rbar': 'AR',
            'rbiw': 'IW',
        }
        self.label2rec_type = {v: k for k, v in self.rec_type2label.items()}
        self.plot_file_types = [
            '.png',
            # '.pdf',
        ]

    def compute(self, label, data_set):
        print('computing...')
        print('    ', label)
        pt = PlotData()
        pt.label = data_set.label
        pt.folder_graphs = data_set.folder_graphs
        for i, rec_type in enumerate(data_set.missions):
            pt.missions[rec_type] = {}
            for dtype in div_types:
                for j, g in enumerate(n_vals):
                    graph = data_set.folder_graphs + '/' + rec_type + '_' + unicode(g) + dtype + '.gt'
                    pt.missions[rec_type][graph] = {}
                    for strategy in Strategy.strategies:
                        pt.missions[rec_type][graph][strategy] = {}
                        for scenario in Mission.missions:
                            debug(rec_type, graph, strategy, scenario)
                            m = data_set.missions[rec_type][graph][strategy][scenario]
                            m.compute_stats()
                            pt.missions[rec_type][graph][strategy][scenario] = m.stats
        return pt

    def plot(self):
        print('plot()')
        for data_set in self.data_sets:
            for scenario in Mission.missions:
                fig, axes = plt.subplots(len(data_set.missions),
                                         len(data_set.missions['cf_cosine']),
                                         figsize=(14, 14))
                for i, rec_type in enumerate(rec_types):
                    for j, g in enumerate((5, 10, 15, 20)):
                        graph = data_set.folder_graphs + rec_type + '_' + \
                            unicode(g) + '.txt'
                        for strategy in Strategy.strategies:
                            debug(rec_type, graph, strategy, scenario)
                            stats = data_set.missions[rec_type][graph][strategy][scenario]
                            ppl.plot(axes[i, j],
                                     np.arange(Navigator.steps_max + 1),
                                     stats, label=strategy, linewidth=1)
                            axes[i, j].set_ylim(0, 100)
                            axes[i, j].set_xlim(0, Navigator.steps_max * 1.1)
                            label = rec_type + ', Top' + str(g)
                            axes[i, j].set_title(label, size=10)
                fig.subplots_adjust(left=0.06, bottom=0.05, right=0.95,
                                    top=0.98, wspace=0.30, hspace=0.30)
                axes[0, 0].legend(loc=0, prop={'size': 6})

                for i in range(axes.shape[0]):
                    axes[i, 0].set_ylabel('Success Ratio')

                for j in range(axes.shape[1]):
                    axes[-1, j].set_xlabel('#Hops')

                plt.savefig('plots/' + data_set.label + '_' +
                            self.sc2abb[scenario] + '.pdf')

    def plot_aggregated(self):
        print('plot_aggregated()')
        colors = [['#66C2A5', '#46AF8E', '#34836A', '#235847'],
                  ['#FC8D62', '#FB6023', '#DC4204', '#A03003'],
                  ['#8DA0CB', '#657EB8', '#47609A', '#334670']]
        styles = ['-', ':', '-.', '--', '-', ':', '-.', '--']
        fig, axes = plt.subplots(len(self.data_sets), len(self.sc2abb),
                                 figsize=(18, 7))
        for dind, data_set in enumerate(self.data_sets):
            for sind, scenario in enumerate(Mission.missions):
                debug(data_set.label, scenario)
                ax = axes[dind, sind]
                ax.set_title(scenario)
                for i, rec_type in enumerate(rec_types):
                    debug('    ', rec_type)
                    # for j, g in enumerate((5, 10, 15, 20)):
                    for j, g in enumerate((5, 20)):
                        c_max = None
                        val_max = -1
                        graph = data_set.folder_graphs + rec_type + '_' + \
                            unicode(g) + u'.txt'
                        for k, strategy in enumerate(Strategy.strategies):
                            if strategy in [u'random', u'optimal']:
                                continue
                            debug('        ', strategy, rec_type, g)
                            stats = data_set.missions[rec_type][graph][strategy][scenario]
                            auc = sum(stats)
                            if auc > val_max:
                                val_max = auc
                                c_max = stats
                        ls = styles[i]
                        lab = rec_type
                        if g == 5:
                            lab += u' '
                        lab += unicode(g)
                        x = np.arange(Navigator.steps_max + 1)
                        cidx = 0 if i < len(rec_types)/2 else 1
                        ppl.plot(ax, x, c_max, label=lab, linewidth=2,
                                 # linestyle=ls, color=colors[i][j])
                                 linestyle=ls, color=colors[cidx][j])

                ax.set_xlabel('#Hops')
                ax.set_ylabel('Success Ratio')
                ax.set_ylim(0, 70)
                ax.set_xlim(0, Navigator.steps_max * 1.1)
        for row in range(axes.shape[0]):
            t_x = (axes[row][0].get_ylim()[0] + axes[row][0].get_ylim()[1]) / 2
            label = [u'MovieLens', u'BookCrossing'][row]
            axes[row][0].text(-55, t_x, label, size='x-large')
        leg = plt.legend(bbox_to_anchor=(2.6, 1.25), loc='center right')
        leg.get_frame().set_linewidth(0.0)
        fig.subplots_adjust(left=0.2, bottom=0.08, right=0.75, top=0.93,
                            wspace=0.31, hspace=0.42)
        # plt.show()
        # plt.savefig('plots/navigation_aggregated_' + unicode(g) + '.pdf')
        plt.savefig('plots/navigation_aggregated.pdf')

    def plot_bar(self):
        print('plot_bar()')
        # plot the legend in a separate plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # patches = [ax.bar([0], [0]) for i in range(4)]
        # for pidx, p in enumerate(patches):
        #     p[0].set_fill(False)
        #     p[0].set_edgecolor(self.colors[pidx])
        #     p[0].set_hatch(self.hatches[pidx])
        # figlegend = plt.figure(figsize=(7.75, 0.465))
        # figlegend.legend(patches, ['No Diversification', 'ExpRel', 'Diversify', 'Random'], ncol=4)
        # fig.subplots_adjust(left=0.19, bottom=0.06, right=0.91, top=0.92,
        #                     wspace=0.34, hspace=0.32)
        # # plt.show()
        # figlegend.savefig('plots/nav_legend.pdf')

        # plot the scenarios
        for scenario in Mission.missions:
            for data_set in self.data_sets:
                fig, ax = plt.subplots(1, figsize=(6, 3))
                bar_vals = []
                for graph_type in self.graph_order:
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals):
                        g = data_set.folder_graphs + '/' + rec_type +\
                                '_' + str(N) + '.gt'
                        bar_vals.append(data_set.missions[rec_type][g]['title'][scenario][-1])
                        print(graph_type, rec_type, N, bar_vals[-1])
                x_vals = [1, 2, 4, 5, 7, 8, 10, 11]
                bars = ax.bar(x_vals, bar_vals, align='center')

                # Beautification
                for bidx, bar in enumerate(bars):
                    bar.set_fill(False)
                    bar.set_hatch(self.hatches[bidx % 2])
                    bar.set_edgecolor(self.colors[int(bidx/2)])

                ax.set_xlim(0.25, 3 * len(self.graphs))
                ax.set_xticks([x - 0.25 for x in x_vals])
                for tic in ax.xaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                labels = [g for k in self.graph_order for g in self.graph_labels[k]]
                ax.set_xticklabels(labels, rotation='-50', ha='left')

                ax.set_ylim(0, 100)
                ylabel = 'Found Nodes (%)'
                ax.set_ylabel(ylabel)

                plt.tight_layout()
                fname = data_set.label + '_' + scenario.lower().replace(' ', '_')
                fpath = os.path.join('plots', fname)
                for ftype in self.plot_file_types:
                    plt.savefig(fpath + ftype)
                plt.close()


    def plot_sample(self):
        """plot and save an example evaluation showing all types of background
        knowledge used in the simulations
        """
        print(u'plot_sample()')
        data_set = self.data_sets[1]
        scenario = u'Greedy Search'
        titles = [u'Collaborative Filtering', u'Content-based']
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for i, rec_type in enumerate(data_set.missions):
            graph = data_set.folder_graphs + rec_type + '_' + str(15) + u'.txt'
            for strategy in Strategy.strategies:
                m = data_set.missions[rec_type][graph][strategy][scenario]
                m.compute_stats()
                ppl.plot(axes[i], np.arange(Navigator.steps_max + 1),
                         m.stats, label=strategy, linewidth=2)
                axes[i].set_xlabel(u'#Hops')
                axes[i].set_ylabel(u'Success Ratio')
                axes[i].set_ylim(0, 85)
                axes[i].set_xlim(0, Navigator.steps_max * 1.01)
                axes[i].set_title(titles[i])
            ppl.legend(axes[i], loc=0)


        # plt.suptitle(u'Greedy Search on the BookCrossing for N=15',
        #              size='xx-large', x=0.5)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.9)

        plt.savefig('plots/sample.png')
        plt.savefig('plots/sample.pdf')


rec_types = [
    # 'cb',
    'rb',
    'rbmf',
    'rbar',
    'rbiw',
]

div_types = [
    '',
    '_div_random',
    '_div_diversify',
    '_div_exprel'
]

n_vals = [
    5,
    # 10,
    # 15,
    20
]


if __name__ == '__main__':
    # for dataset in [
    #     # 'movielens',
    #     'bookcrossing',
    # ]:
    #     dataset = DataSet(dataset, rec_types, div_types)
    #     nav = Navigator(dataset)
    #     print('running...')
    #     nav.run()

    evaluator = Evaluator(datasets=['movielens', 'bookcrossing'])
    evaluator.plot_bar()

