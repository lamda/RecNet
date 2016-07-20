# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import copy
import io
import operator
import prettyplotlib as ppl
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
import sys


def debug(*text):
    """wrapper for the print function that can be turned on and off"""
    if False:
    # if True:
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
        self.stats = np.zeros(STEPS_MAX + 1)
        for m in self.missions:
            m.compute_stats()
            self.stats += 100 * m.stats
        self.stats /= len(self.missions)


class Mission(object):
    """This class represents a Point-to-Point Search mission"""
    # mission types
    missions = [
        u'Greedy Search',
        u'Greedy Search (Random)',
        u'Berrypicking',
        u'Berrypicking (Random)',
        u'Information Foraging',
        u'Information Foraging (Random)'
    ]

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
        if self.steps > STEPS_MAX or not self.targets[0]:
            return False
        return True

    def reset(self):
        """reset the visited nodes and go to the next target set"""
        self.visited = set()
        del self.targets[0]

    def compute_stats(self):
        self.stats = np.zeros(STEPS_MAX + 1)
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
        self.stats = np.zeros(STEPS_MAX + 1)
        targets = copy.deepcopy(self.targets_original[0])
        i = targets.index(self.path[0])
        del targets[i]
        curr = 0
        for i, n in enumerate(self.path[:len(self.stats)]):
            if n in targets:
                targets.remove(n)
                curr += (1 / 3)  # new: normalize by no. of clusters instead
            self.stats[i] = curr
        if i < len(self.stats):
            self.stats[i:] = curr
        self.stats = np.array([min(i, 1.0) for i in self.stats])


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
        if self.steps > STEPS_MAX or not self.targets[0]:
            return False
        return True

    def compute_stats(self):
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


class Strategy(object):
    """This class represents a strategy for choosing the next hop.
    During missions, the find_next method is called to select the next node
    """
    strategies = [
        u'random',
        u'title',
        # u'title_stochastic',
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
        except TypeError, e:
            print(e)
            pdb.set_trace()
        if not candidates:
            chosen_node = None  # abort search
        else:
            if strategy == 'title_stochastic' and random.random() <= 0.05:
                chosen_node = random.choice(candidates.keys())
                debug('randomly selecting node', chosen_node)
                return chosen_node
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
    def __init__(self, label, rec_types, pers_recs, personalization_types):
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
                    rec_type + '_' + unicode(N) + '.gt')
                for N in self.n_vals
            ]
            if rec_type in pers_recs:
                self.graphs[rec_type] += [
                    os.path.join(
                        self.folder_graphs,
                        rec_type + '_' + unicode(N) + p + '.gt')
                    for N in self.n_vals
                    for p in personalization_types
                ]
            self.compute_shortest_path_lengths(self.graphs[rec_type])

        missions = {
            u'Greedy Search': self.load_missions(Mission, u'missions.txt'),
            u'Greedy Search (Random)': self.load_missions(Mission, u'missions_random.txt'),
            u'Information Foraging': self.load_missions(IFMission, u'missions_if.txt'),
            u'Information Foraging (Random)': self.load_missions(IFMission, u'missions_if_random.txt'),
            u'Berrypicking': self.load_missions(BPMission, u'missions_bp.txt'),
            u'Berrypicking (Random)': self.load_missions(BPMission, u'missions_bp_random.txt'),
        }

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
                self.matrices[rec_type][graph]['title_stochastic'] =\
                    self.matrices[rec_type][graph]['title']
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
    def __init__(self, data_set):
        self.data_set = data_set

    def run(self):
        """run the simulations for all strategies (optimal, random and informed)
        """
        print('    strategies...')
        matrix_file = ''
        matrix_s, matrix_c = None, None
        # run for all but the optimal version
        item2matrix = os.path.join(self.data_set.base_folder, 'item2matrix.txt')
        for rec_type in self.data_set.graphs:
            for graph in self.data_set.graphs[rec_type]:
                print('        ', graph)
                gt_graph = load_graph(graph)
                for strategy in Strategy.strategies:
                    if strategy == 'optimal':
                        continue
                    print('            ', strategy)
                    m_new = self.data_set.matrices[rec_type][graph][strategy][0]
                    m_newc = self.data_set.matrices[rec_type][graph][strategy][1]
                    debug('             ----', m_new)
                    debug('             ----', m_newc)
                    if not m_new:
                        debug('             ---- not m_new')
                        matrix_s, matrix_c, matrix_file = None, None, None
                    elif matrix_file != m_new:
                        matrix_s = SimilarityMatrix(item2matrix, m_new)
                        matrix_c = SimilarityMatrix(item2matrix, m_newc)
                        matrix_file = m_new
                        debug('            ---- matrix_file != m_new')
                    # for miss in self.data_set.missions[rec_type][graph][strategy]:
                    for miss in Mission.missions:
                        print('                ', miss)
                        if 'Information Foraging' in miss or 'Berrypicking' in miss:
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
                                if ti > 0 and len(m.targets_original[ti]) == len(m.targets[0]):
                                    # print('breaking...')
                                    m.reset()
                                    break
                                if not (ti + 1) == len(m.targets_original):
                                    m.path.append(u'*')
                                m.reset()

        # run the simulations for the optimal solution
        print('    optimal...')
        for rec_type in self.data_set.graphs:
            for graph in self.data_set.graphs[rec_type]:
                print('        ', graph)
                sp_file = graph.rsplit('.', 1)[0] + '.npy'
                with open(sp_file, 'rb') as infile:
                    sp = pickle.load(infile)
                for miss in self.data_set.missions[rec_type][graph]['optimal']:
                    for m in self.data_set.missions[rec_type][graph]['optimal'][miss]:
                        for ti in xrange(len(m.targets_original)):
                            start = m.path[-2] if m.path else m.start
                            debug('++++' * 16, 'mission', ti, '/', len(m.targets_original))
                            debug(m.targets_original[ti])
                            self.optimal_path(m, start, sp)
                            if not (ti + 1) == len(m.targets_original):
                                m.path.append(u'*')
                            m.reset()

        # # DEBUG
        # item2matrix = os.path.join(self.data_set.base_folder, 'item2matrix.txt')
        # for rec_type in ['rbar']:
        #     for graph in self.data_set.graphs[rec_type]:
        #         print('        ', graph)
        #         gt_graph = load_graph(graph)
        #         sp_file = graph.rsplit('.', 1)[0] + '.npy'
        #         with open(sp_file, 'rb') as infile:
        #             sp = pickle.load(infile)
        #         m_newc = self.data_set.matrices[rec_type][graph]['title'][1]
        #         matrix = SimilarityMatrix(item2matrix, m_newc)
        #         sc = 'Berrypicking'
        #         mc1 = self.data_set.missions[rec_type][graph]['title'][sc]
        #         mc2 = self.data_set.missions[rec_type][graph]['optimal'][sc]
        #         mc3 = self.data_set.missions[rec_type][graph]['random'][sc]
        #         for m1, m2, m3 in zip(
        #             mc1,
        #             mc2,
        #             mc3
        #         ):
        #             # evalute with title strategy
        #             for ti in xrange(len(m1.targets_original)):
        #                 start = m1.path[-2] if m1.path else m1.start
        #                 debug('++++' * 16, 'mission', ti, '/', len(m1.targets_original))
        #                 # debug(m1.targets_original[ti])
        #                 self.navigate(gt_graph, 'title', m1, start, None, matrix)
        #                 # print(m1.path, ti, len(m1.targets_original[ti]), len(m1.targets[0]))
        #                 if ti > 0 and len(m1.targets_original[ti]) == len(m1.targets[0]):
        #                     # print('breaking...')
        #                     m1.reset()
        #                     break
        #                 if not (ti + 1) == len(m1.targets_original):
        #                     m1.path.append(u'*')
        #                 m1.reset()
        #
        #             # evaluate with optimal strategy
        #             for ti in xrange(len(m2.targets_original)):
        #                 start = m2.path[-2] if m2.path else m2.start
        #                 # debug('++++' * 16, 'mission', ti, '/', len(m2.targets_original))
        #                 # debug(m2.targets_original[ti])
        #                 self.optimal_path(m2, start, sp)
        #                 if not (ti + 1) == len(m2.targets_original):
        #                     m2.path.append(u'*')
        #                     m2.reset()
        #             # pdb.set_trace()
        #
        #             # if len(m1.path) < len(m2.path):
        #             #     print(len(m1.path), len(m2.path))
        #             #     pdb.set_trace()
        #             # m1.compute_stats()
        #             # m2.compute_stats()
        #             # if m1.stats[-1] > m2.stats[-1]:
        #             #     print(m1.stats)
        #             #     print(m2.stats)
        #             #     pdb.set_trace()
        #
        #         print('MISSION COLLECTION DONE')
        #         mc1.compute_stats()
        #         mc2.compute_stats()
        #         print(mc1.stats[-1], mc2.stats[-1])
        #         pdb.set_trace()

        # fname_5 = u'../data/bookcrossing/graphs/rbar_5.gt'
        # fname_20 = u'../data/bookcrossing/graphs/rbar_20.gt'
        # sp_file_5 = fname_5.rsplit('.', 1)[0] + '.npy'
        # sp_file_20 = fname_20.rsplit('.', 1)[0] + '.npy'
        # with open(sp_file_5, 'rb') as infile:
        #     sp_5 = pickle.load(infile)
        # with open(sp_file_20, 'rb') as infile:
        #     sp_20 = pickle.load(infile)
        # sc = 'Berrypicking'
        # mc_5 = self.data_set.missions['rbar'][fname_5]['optimal'][sc]
        # mc_52 = self.data_set.missions['rbar'][fname_5]['title'][sc]
        # mc_20 = self.data_set.missions['rbar'][fname_20]['optimal'][sc]
        # mc_202 = self.data_set.missions['rbar'][fname_20]['title'][sc]
        # for m5, m20, m52, m202 in zip(
        #     mc_5,
        #     mc_20,
        #     mc_52,
        #     mc_202
        # ):
        #     # evaluate 5 with optimal strategy
        #     for ti in xrange(len(m5.targets_original)):
        #         start = m5.path[-2] if m5.path else m5.start
        #         self.optimal_path(m5, start, sp_5)
        #         if not (ti + 1) == len(m5.targets_original):
        #             m5.path.append(u'*')
        #             m5.reset()
        #
        #     # evaluate 20 with optimal strategy
        #     for ti in xrange(len(m20.targets_original)):
        #         start = m20.path[-2] if m20.path else m20.start
        #         self.optimal_path(m20, start, sp_20)
        #         if not (ti + 1) == len(m20.targets_original):
        #             m20.path.append(u'*')
        #             m20.reset()
        #
        #     # if len(m5.path) < len(m20.path) or \
        #     if m5.path.count('*') > m20.path.count('*'):
        #         print(len(m5.path))
        #         for part in ' '.join(m5.path[2:]).split('*'):
        #             print('   ', part)
        #         print(len(m20.path))
        #         for part in ' '.join(m20.path[2:]).split('*'):
        #             print('   ', part)
        #         pdb.set_trace()
        #
        # print('MISSION COLLECTION DONE')
        # mc_5.compute_stats()
        # mc_20.compute_stats()
        # print(mc_5.stats[-1], mc_20.stats[-1])
        #
        # for m5, m20 in zip(mc_5.missions, mc_20.missions):
        #     if m5.stats[-1] > m20.stats[-1]:
        #         print(m5.stats)
        #         print(m20.stats)
        #         pdb.set_trace()
        # pdb.set_trace()

        # write the results to a file
        # self.write_paths()
        self.save()

    def optimal_path(self, mission, start, sp):
        """write a fake path to the mission, that is of the correct length
        if more evaluations such as the set of visited nodes are needed,
        this needs to be extended
        """
        mission.add(start)
        while mission.targets[0] and mission.is_active():
            ds = [(sp[start][t], t) for t in mission.targets[0] if t in sp[start]]
            if not ds:
                mission.add(u'-1')  # target not connected --> fill with dummies
                continue
            target = min(ds)
            for i in range(target[0] - 1):
                mission.add(u'0')
            mission.add(target[1])
            start = target[1]

    def navigate(self, graph, strategy, mission, node, parent_node=None,
                 matrix=None):
        debug('-' * 32 + '\n')
        debug('navigate called with', node, '(parent: ', parent_node, ')')
        # debug(mission.targets[0])
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
                debug('backtracking')
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
                        for miss in Mission.missions:
                            outfile.write('----' * 2 + miss + '\n')
                            stras = self.data_set.missions[rec_type][graph][strategy][miss]
                            for m in stras:
                                outfile.write('\t'.join(m.path) + '\n')

    def save(self):
        fp = 'data_sets_' + self.data_set.label + '_' + str(STEPS_MAX) + '.obj'
        with open(fp, 'wb') as outfile:
            pickle.dump([self.data_set], outfile, -1)


class PlotData(object):
    def __init__(self):
        self.missions = {}


class Evaluator(object):
    """Class responsible for calculating stats and plotting the results"""
    def __init__(self, datasets, stochastic=False, personalized=False,
                 suffix='', pdf=False, subtract_baseline=False):
        self.data_sets = []
        self.stochastic = stochastic
        self.personalized = personalized
        self.personalized_suffix = '_personalized' if self.personalized else ''
        self.suffix = suffix
        self.subtract_baseline = subtract_baseline
        if self.subtract_baseline:
            self.suffix += '_sb'
        for dataset in datasets:
            try:
                with open('data_sets_' + dataset + '_' + str(STEPS_MAX) +
                                  self.personalized_suffix + '_new.obj', 'rb')\
                        as infile:
                    print('loading...')
                    self.data_sets.append(pickle.load(infile)[0])
                print('loaded')
            except (IOError, EOFError):
                print('loading failed... computing from scratch (%s)' % dataset)
                with open('data_sets_' + dataset + '_' + str(STEPS_MAX) +
                                  '.obj', 'rb') as infile:
                    data_set = pickle.load(infile)[0]
                data_set_new = self.compute(label=dataset, data_set=data_set)
                self.data_sets.append(data_set_new)
                print('saving to disk...')
                with open('data_sets_' + dataset + '_' + str(STEPS_MAX) +
                                  self.personalized_suffix + '_new.obj', 'wb')\
                        as outfile:
                    pickle.dump([data_set_new], outfile, -1)

        if not os.path.isdir('plots'):
            os.makedirs('plots')
        self.sc2abb = {
            u'Greedy Search': u'ptp',
            u'Greedy Search (Random)': u'ptp_random',
            u'Information Foraging': u'if',
            u'Information Foraging (Random)': u'if_random',
            u'Berrypicking': u'bp',
            u'Berrypicking (Random)': u'bp_random',
        }
        self.colors = ['#FFA500', '#FF0000', '#0000FF', '#05FF05', '#000000']
        self.hatches = ['----', '/', 'xxx', '///', '---']
        self.linestyles = ['-', '--', ':', '-.']
        if not self.personalized:
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
        else:
            self.graphs = {
                'MF': ['rbmf_' + str(c) + p for c in n_vals for p in personalized_types],
            }
            p2pl = {
                '_personalized_min': 'Pure',
                '_personalized_median': 'Pure',
                '_personalized_max': 'Pure',
                '_personalized_mixed_min': 'Mixed',
                '_personalized_mixed_median': 'Mixed',
                '_personalized_mixed_max': 'Mixed',
            }
            self.graph_labels = {
                'MF': [p2pl[p]for c in n_vals for p in personalized_types],
            }
            self.graph_order = ['MF']

        self.rec_type2label = {
            'rb': 'CF',
            'rbmf': 'MF',
            'rbar': 'AR',
            'rbiw': 'IW',
        }
        self.label2rec_type = {v: k for k, v in self.rec_type2label.items()}
        self.plot_file_types = [
            '.png',
        ]
        if pdf:
            self.plot_file_types.append('.pdf')

    def compute(self, label, data_set):
        print('computing...')
        print('   ', label)
        pt = PlotData()
        pt.label = data_set.label
        pt.folder_graphs = data_set.folder_graphs

        if not self.personalized:
            for i, rec_type in enumerate(data_set.missions):
                pt.missions[rec_type] = {}
                for j, g in enumerate(n_vals):
                    graph = os.path.join(
                        data_set.folder_graphs,
                        rec_type + '_' + unicode(g) + '.gt'
                    )
                    pt.missions[rec_type][graph] = {}
                    for strategy in Strategy.strategies:
                    # for strategy in ['title']:
                        pt.missions[rec_type][graph][strategy] = {}
                        for scenario in Mission.missions:
                        # for scenario in ['Berrypicking']:
                            debug(rec_type, graph, strategy, scenario)
                            m = data_set.missions[rec_type][graph][strategy][scenario]
                            m.compute_stats()
                            pt.missions[rec_type][graph][strategy][scenario] = m.stats
        else:
            for i, rec_type in enumerate(data_set.missions):
                if rec_type not in pers_recs:
                    continue
                pt.missions[rec_type] = {}
                for j, g in enumerate(n_vals):
                    for pers_type in personalized_types:
                        graph = os.path.join(
                            data_set.folder_graphs,
                            rec_type + '_' + unicode(g) + pers_type + '.gt'
                        )
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
                                     np.arange(STEPS_MAX + 1),
                                     stats, label=strategy, linewidth=1)
                            axes[i, j].set_ylim(0, 100)
                            axes[i, j].set_xlim(0, STEPS_MAX * 1.1)
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
                        x = np.arange(STEPS_MAX + 1)
                        cidx = 0 if i < len(rec_types)/2 else 1
                        ppl.plot(ax, x, c_max, label=lab, linewidth=2,
                                 # linestyle=ls, color=colors[i][j])
                                 linestyle=ls, color=colors[cidx][j])

                ax.set_xlabel('#Hops')
                ax.set_ylabel('Success Ratio')
                ax.set_ylim(0, 70)
                ax.set_xlim(0, STEPS_MAX * 1.1)
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
        print('---------------------------------------------------------------')

        # plot the scenarios
        better = []
        x_vals = [1, 2, 4, 5, 7, 8, 10, 11]
        for scenario in Mission.missions:
            if 'Information' not in scenario:
                continue
            print(scenario)
            av_5, av_20 = [], []
            for data_set in self.data_sets:
                print('   ', data_set.label)
                fig, ax = plt.subplots(1, figsize=(6, 3))

                # plot optimal solutions
                bar_vals = []
                for graph_type in self.graph_order:
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals):
                        # print('       ', N, rec_type)
                        g = data_set.folder_graphs + '/' + rec_type +\
                                '_' + str(N) + '.gt'
                        o = data_set.missions[rec_type][g]['optimal'][scenario][-1]
                        r = data_set.missions[rec_type][g]['random'][scenario][-1]
                        val = o-r if self.subtract_baseline else o
                        # print('           ', o)
                        bar_vals.append(val)
                bars = ax.bar(x_vals, bar_vals, align='center', color='#EFEFEF')

                # Beautification
                for bidx, bar in enumerate(bars):
                    # bar.set_fill(False)
                    bar.set_edgecolor('#AAAAAA')

                # plot simulation results
                bar_vals = []
                for graph_type in self.graph_order:
                    # print('       ', graph_type)
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals):
                        # print('           ', N)
                        g = data_set.folder_graphs + '/' + rec_type +\
                            '_' + str(N) + '.gt'
                        if self.stochastic:
                            s = data_set.missions[rec_type][g]['title_stochastic'][scenario][-1]
                        else:
                            s = data_set.missions[rec_type][g]['title'][scenario][-1]
                        r = data_set.missions[rec_type][g]['random'][scenario][-1]
                        o = data_set.missions[rec_type][g]['optimal'][scenario][-1]
                        if self.subtract_baseline:
                            bar_vals.append(s-r)
                        else:
                            bar_vals.append(s)
                        if r == 0:
                            print('for', g, 'random walk found no targets')
                        else:
                            better.append(s/r)

                        if N == 5:
                            av_5.append(s)
                        elif N == 20:
                            av_20.append(s)
                        print('       ', scenario, graph_type, N, '%.2f' % (s))
                        # pdb.set_trace()

                        # print('                %.2f, %.2f, %.2f' % (r, bar_vals[-1], o))
                        # if s > o:
                        #     print(scenario, data_set.label, graph_type, N, '%.2f > %.2f' % (bar_vals[-1], o))
                            # print(g)
                            # pdb.set_trace()
                bars = ax.bar(x_vals, bar_vals, align='center')
                # print('        5: %.2f' % np.mean(av_5))
                # print('       20: %.2f' % np.mean(av_20))

                # Beautification
                for bidx, bar in enumerate(bars):
                    bar.set_fill(False)
                    bar.set_hatch(self.hatches[bidx % 2])
                    bar.set_edgecolor(self.colors[int(bidx/2)])

                if not self.subtract_baseline:
                    # plot random walk solutions (as a dot)
                    bar_vals = []
                    for graph_type in self.graph_order:
                        rec_type = self.label2rec_type[graph_type]
                        for nidx, N in enumerate(n_vals):
                            g = data_set.folder_graphs + '/' + rec_type +\
                                    '_' + str(N) + '.gt'
                            val = data_set.missions[rec_type][g]['random'][scenario][-1]
                            bar_vals.append(val)
                    ax.plot(x_vals, bar_vals, c='black', ls='', marker='.', ms=10)

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
                stochastic_suffix = '_stochastic' if self.stochastic else ''
                sc = scenario.lower().replace(' ', '_').replace('(', '').replace(')', '')
                fname = data_set.label + '_' + str(STEPS_MAX) + '_' + sc + \
                        stochastic_suffix + self.suffix
                fpath = os.path.join('plots', fname)
                for ftype in self.plot_file_types:
                    plt.savefig(fpath + ftype)
                plt.close()
            # print('random walks average is %.2f' % np.average(hugo))
        print('simulations were on average %.2f times better than'
              ' the random walks' % np.average(better))
        print('---------------------------------------------------------------')

    def plot_bar_personalized(self):
        print('plot_bar_personalized()')
        # plot the scenarios
        better = []
        x_vals = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        for scenario in Mission.missions:
            hugo = []
            # print(scenario)
            for data_set in self.data_sets:
                # print('   ', data_set.label)
                fig, ax = plt.subplots(1, figsize=(12, 6))

                # plot optimal solutions
                bar_vals = []
                for graph_type in self.graph_order:
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals):
                        for pidx, pt in enumerate(personalized_types):
                            g = data_set.folder_graphs + '/' + rec_type +\
                                    '_' + str(N) + pt + '.gt'
                            bar_vals.append(data_set.missions[rec_type][g]['optimal'][scenario][-1])
                bars = ax.bar(x_vals, bar_vals, align='center', color='#EFEFEF')

                # Beautification
                for bidx, bar in enumerate(bars):
                    # bar.set_fill(False)
                    bar.set_edgecolor('#AAAAAA')

                # plot simulation results
                bar_vals = []
                for graph_type in self.graph_order:
                    # print('       ', graph_type)
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals):
                        for pidx, pt in enumerate(personalized_types):
                            g = data_set.folder_graphs + '/' + rec_type +\
                                    '_' + str(N) + pt + '.gt'
                            if self.stochastic:
                                s = data_set.missions[rec_type][g]['title_stochastic'][scenario][-1]
                            else:
                                s = data_set.missions[rec_type][g]['title'][scenario][-1]
                            r = data_set.missions[rec_type][g]['random'][scenario][-1]
                            o = data_set.missions[rec_type][g]['optimal'][scenario][-1]
                            bar_vals.append(s)
                            better.append(s/r)
                            hugo.append(r)
                            # print('            %.2f, %.2f, %.2f' % (r, bar_vals[-1], o))
                            if s > o:
                                print(scenario, data_set.label, graph_type, '%.2f > %.2f' % (bar_vals[-1], o))
                bars = ax.bar(x_vals, bar_vals, align='center')

                # Beautification
                for bidx, bar in enumerate(bars):
                    bar.set_fill(False)
                    bar.set_hatch(self.hatches[bidx % 2])
                    bar.set_edgecolor(self.colors[3])

                # plot random walk solutions (as a dot)
                bar_vals = []
                for graph_type in self.graph_order:
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals):
                        for pidx, pt in enumerate(personalized_types):
                            g = data_set.folder_graphs + '/' + rec_type +\
                                    '_' + str(N) + pt + '.gt'
                            bar_vals.append(data_set.missions[rec_type][g]['random'][scenario][-1])
                ax.plot(x_vals, bar_vals, c='black', ls='', marker='.', ms=10)

                ax.set_xlim(0.25, x_vals[-1] + 0.75)
                ax.set_xticks([x - 0.25 for x in x_vals])
                for tic in ax.xaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                labels = [g for k in self.graph_order for g in self.graph_labels[k]]
                ax.set_xticklabels(labels, rotation='-50', ha='left')

                ax.set_ylim(0, 100)
                ylabel = 'Found Nodes (%)'
                ax.set_ylabel(ylabel)
                plt.tight_layout()
                stochastic_suffix = 'stochastic_' if self.stochastic else ''
                fname = data_set.label + '_' + str(STEPS_MAX) + '_personalized_' + \
                        stochastic_suffix +\
                        scenario.lower().replace(' ', '_').replace('(', '').replace(')', '') +\
                        self.suffix
                if not os.path.isdir(os.path.join('plots', 'personalized')):
                    os.makedirs(os.path.join('plots', 'personalized'))
                fpath = os.path.join('plots', 'personalized', fname)
                for ftype in self.plot_file_types:
                    plt.savefig(fpath + ftype)
                plt.close()
                # print('random walks average is %.2f' % np.average(hugo))
        print('simulations were on average %.2f times better than'
              ' the random walks' % np.average(better))

    def plot_bar_personalized_simple(self):
        print('plot_bar_personalized_simple()')
        personalized_types_simple = [
            '_personalized_median',
            '_personalized_mixed_median',
        ]
        n_vals_simple = [
            20
        ]

        # plot the scenarios
        better = []
        x_vals = [1, 2]
        for scenario in Mission.missions:
            # print(scenario)
            if 'random' in scenario.lower():
                continue
            for data_set in self.data_sets:
                # print('   ', data_set.label)
                fig, ax = plt.subplots(1, figsize=(2, 3))

                # plot optimal solutions
                bar_vals = []
                for graph_type in self.graph_order:
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals_simple):
                        for pidx, pt in enumerate(personalized_types_simple):
                            g = data_set.folder_graphs + '/' + rec_type + \
                                '_' + str(N) + pt + '.gt'
                            bar_vals.append(
                                data_set.missions[rec_type][g]['optimal'][
                                    scenario][-1])
                bars = ax.bar(x_vals, bar_vals, align='center', color='#EFEFEF')

                # Beautification
                for bidx, bar in enumerate(bars):
                    # bar.set_fill(False)
                    bar.set_edgecolor('#AAAAAA')

                # plot simulation results
                bar_vals = []
                for graph_type in self.graph_order:
                    # print('       ', graph_type)
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals_simple):
                        for pidx, pt in enumerate(personalized_types_simple):
                            g = data_set.folder_graphs + '/' + rec_type + \
                                '_' + str(N) + pt + '.gt'
                            if self.stochastic:
                                s = data_set.missions[rec_type][g][
                                    'title_stochastic'][scenario][-1]
                            else:
                                s = data_set.missions[rec_type][g]['title'][scenario][-1]
                            r = data_set.missions[rec_type][g]['random'][scenario][-1]
                            o = data_set.missions[rec_type][g]['optimal'][scenario][-1]
                            bar_vals.append(s)
                            better.append(s / r)
                            # print('            %.2f, %.2f, %.2f' % (r, bar_vals[-1], o))
                            if s > o:
                                print(scenario, data_set.label, graph_type,
                                      '%.2f > %.2f' % (bar_vals[-1], o))
                bars = ax.bar(x_vals, bar_vals, align='center')

                # Beautification
                for bidx, bar in enumerate(bars):
                    bar.set_fill(False)
                    bar.set_hatch(self.hatches[1])
                    bar.set_edgecolor(self.colors[3])

                # plot random walk solutions (as a dot)
                bar_vals = []
                for graph_type in self.graph_order:
                    rec_type = self.label2rec_type[graph_type]
                    for nidx, N in enumerate(n_vals_simple):
                        for pidx, pt in enumerate(personalized_types_simple):
                            g = data_set.folder_graphs + '/' + rec_type + \
                                '_' + str(N) + pt + '.gt'
                            bar_vals.append(
                                data_set.missions[rec_type][g]['random'][
                                    scenario][-1])
                ax.plot(x_vals, bar_vals, c='black', ls='', marker='.', ms=10)

                ax.set_xlim(0.25, x_vals[-1] + 0.75)
                ax.set_xticks([x - 0.25 for x in x_vals])
                for tic in ax.xaxis.get_major_ticks():
                    tic.tick1On = tic.tick2On = False
                labels = ['Pure', 'Mixed']

                ax.set_xticklabels(labels, rotation='-50', ha='left')

                ax.set_ylim(0, 100)
                ylabel = 'Found Nodes (%)'
                ax.set_ylabel(ylabel)
                plt.tight_layout()
                stochastic_suffix = 'stochastic_' if self.stochastic else ''
                fname = data_set.label + '_' + str(
                    STEPS_MAX) + '_personalized_' + \
                        stochastic_suffix + \
                        scenario.lower().replace(' ', '_').replace('(',
                                                                   '').replace(
                            ')', '') + \
                        self.suffix + '_simple'
                if not os.path.isdir(os.path.join('plots', 'personalized_simple')):
                    os.makedirs(os.path.join('plots', 'personalized_simple'))
                fpath = os.path.join('plots', 'personalized_simple', fname)
                for ftype in self.plot_file_types:
                    plt.savefig(fpath + ftype)
                plt.close()
                # print('random walks average is %.2f' % np.average(hugo))
        print('simulations were on average %.2f times better than'
              ' the random walks' % np.average(better))

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
                ppl.plot(axes[i], np.arange(STEPS_MAX + 1),
                         m.stats, label=strategy, linewidth=2)
                axes[i].set_xlabel(u'#Hops')
                axes[i].set_ylabel(u'Success Ratio')
                axes[i].set_ylim(0, 85)
                axes[i].set_xlim(0, STEPS_MAX * 1.01)
                axes[i].set_title(titles[i])
            ppl.legend(axes[i], loc=0)


        # plt.suptitle(u'Greedy Search on the BookCrossing for N=15',
        #              size='xx-large', x=0.5)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.9)

        plt.savefig('plots/sample.png')
        plt.savefig('plots/sample.pdf')

    def print_results(self):
        import pandas as pd
        num_rows = len(self.data_sets) * len(rec_types) * len(n_vals) *\
                   len(Strategy.strategies) * len(Mission.missions)
        df = pd.DataFrame(
            index=np.arange(0, num_rows),
            columns=['val', 'data_set', 'rec_type', 'is_random', 'N', 'strategy', 'scenario']
        )
        i = 0
        for data_set in self.data_sets:
            # print(data_set.label)
            dm = data_set.missions
            for rec_type in dm:
                for nidx, N in enumerate(n_vals):
                    graph = data_set.folder_graphs + '/' + rec_type + \
                        '_' + str(N) + '.gt'
                    # print('   ', graph)
                    for strategy in dm[rec_type][graph]:
                        # print('       ', strategy)
                        for scenario in dm[rec_type][graph][strategy]:
                            is_random = True if 'Random' in scenario else False
                            val = dm[rec_type][graph][strategy][scenario][-1]
                            # print('            %.2f %s' % (val, scenario))
                            df.loc[i] = [val, data_set.label, rec_type, is_random, N, strategy, scenario]
                            i += 1
        df['val'] = df['val'].astype(float)
        # pd.pivot_table(df, values='val', index='scenario', columns=['rec_type', 'N'])
        # pd.pivot_table(df[df['strategy'] == 'title'], values='val', index='rec_type', columns=['is_random'])
        df_agg = pd.pivot_table(
            df[df['strategy'] == 'title'], values='val',
            index='rec_type',
            columns=['is_random', 'data_set']
        )

        df_agg = pd.pivot_table(df[df['strategy'] == 'title'], values='val', index='scenario', columns=['is_random', 'data_set'])

        pdb.set_trace()

rec_types = [
    'rb',
    'rbar',
    'rbiw',
    'rbmf',
]

pers_recs = [
    'rbmf',
]

personalized_types = [
    '_personalized_min',
    '_personalized_median',
    '_personalized_max',
    '_personalized_mixed_min',
    '_personalized_mixed_median',
    '_personalized_mixed_max',
]

n_vals = [
    5,
    20
]


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in [
        'bookcrossing',
        'movielens',
        'imdb',
        'evaluate'
    ]:
        print('dataset not supported')
        sys.exit()
    dataset_label = sys.argv[1]
    if len(sys.argv) > 2:
        STEPS_MAX = int(sys.argv[2])
    else:
        STEPS_MAX = 50
    print(dataset_label)
    print(STEPS_MAX)

    if dataset_label != 'evaluate':
        dataset = DataSet(dataset_label, rec_types, pers_recs, personalized_types)
        nav = Navigator(dataset)
        print('running...')
        nav.run()
    else:
        datasets = [
            'bookcrossing',
            'movielens',
            'imdb'
        ]
        evaluator = Evaluator(datasets=datasets, pdf=True)
        # evaluator.plot_bar()

        # evaluator = Evaluator(datasets=datasets, subtract_baseline=True, pdf=True)
        # evaluator.plot_bar()
        evaluator.print_results()

        # evaluator = Evaluator(datasets=datasets, personalized=True, pdf=True)
        # evaluator.plot_bar_personalized_simple()


