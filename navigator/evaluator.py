# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os
import cPickle as pickle
import pdb

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


def debug(*text):
    """wrapper for the print(function that can be turned on and off"""
    if False:
        print(' '.join(str(t) for t in text))


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
        self.colors = [['#FFA500', '#05FF05', '#000000'],
                       ['#FF0000', '#0000FF', '#000000']]
        self.hatches = ['', 'xx', '//', '--']

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
                            debug(rec_type, graph, strategy, scenario)
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
            p[0].set_edgecolor('black')
            p[0].set_hatch(self.hatches[pidx])
        figlegend = plt.figure(figsize=(7.65, 0.465))
        figlegend.legend(patches, ['No Diversification', 'ExpRel', 'Diversify', 'ExpRel'], ncol=4)
        fig.subplots_adjust(left=0.19, bottom=0.06, right=0.91, top=0.92,
                            wspace=0.34, hspace=0.32)
        # plt.show()
        figlegend.savefig('plots/nav_legend.pdf')

        # plot the scenarios
        bars = None
        for sind, scenario in enumerate([u'Information Foraging']):
            debug('\n', scenario)
            for dind, data_set in enumerate(self.data_sets):
                fig, axes = plt.subplots(1, len(div_types), figsize=(13, 3.25),
                                         squeeze=False)
                for nidx, div_type in enumerate(div_types):
                    debug(data_set.label, div_type)
                    ax = axes[0, nidx]
                    for rtsidx, rec_type_split in enumerate([rec_types[:4], rec_types[4:]]):
                        for ridx, rec_type in enumerate(rec_type_split):
                            debug('    ', rec_type)
                            bar_vals = [0 for r in rec_type_split]
                            max_strategy = ''
                            for k, strategy in enumerate(['title']):
                                if strategy in [u'random', u'optimal']:
                                    print('random or optimal strategy')
                                    continue
                                graph = data_set.folder_graphs + '/' +\
                                        rec_type + div_type + '.gt'
                                stats = data_set.missions[rec_type][graph][strategy][scenario]
                                if stats[-1] > bar_vals[ridx]:
                                    bar_vals[ridx] = stats[-1]
                                    max_strategy = strategy
                            debug('         ', max_strategy)
                            x = np.arange(len(rec_type_split))
                            x = [v + rtsidx * (len(rec_type_split) + 1) for v in x]
                            bars = ax.bar(x, bar_vals)
                            for bidx, bar in enumerate(bars):
                                bar.set_fill(False)
                                bar.set_hatch(self.hatches[bidx])
                                bar.set_edgecolor(self.colors[rtsidx][dind])

                    ax.set_title(div_type)
                    ax.set_ylabel('Success Ratio (%)')
                    ax.set_ylim(0, 50)
                    ax.set_xlim([-0.25, None])
                    ax.set_xticks([2, 7])
                    ax.set_xticklabels(['CF', 'CB'])

                fig.subplots_adjust(left=0.06, bottom=0.14, right=0.98,
                                    top=0.88, wspace=0.38, hspace=0.32)
                # plt.show()
                plt.savefig('plots/nav_success_rate.pdf')

rec_types = [
    'CF_6',
    'CF_12',
    'CB_5',
    'CB_10',
]
div_types = [
    '',
    '_div_exprel',
    '_div_diversify',
    '_div_random',
]

if __name__ == '__main__':
    evaluator = Evaluator()
    evaluator.plot_bar()
