# -*- coding: utf-8 -*-

from __future__ import division, print_function

import io
import itertools
import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import cPickle as pickle


class Plotter(object):
    def __init__(self, label, to_plot, personalized, personalized_suffices=('',)):
        self.label = label
        self.personalized = personalized
        self.personalized_suffices = personalized_suffices
        self.stats_folder = os.path.join('data', self.label, 'stats')
        self.colors = ['#FFA500', '#FF0000', '#0000FF', '#05FF05', '#000000']
        self.colors_set2 = [
            (0.4, 0.7607843137254902, 0.6470588235294118),
            (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
            (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
            (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
            (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
            (1.0, 0.8509803921568627, 0.1843137254901961),
            (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
            (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)
        ]
        # self.hatches = ['', 'xxx', '///', '---']
        self.hatches = ['----', '/', 'xxx', '///', '---']
        self.linestyles = ['-', '--', ':', '-.']
        self.graphs = {
            'RB': [['rb_' + c + p for c in n_vals] for p in self.personalized_suffices],
            'MF': [['rbmf_' + c + p for c in n_vals] for p in self.personalized_suffices],
            'AR': [['rbar_' + c + p for c in n_vals] for p in self.personalized_suffices],
            'IW': [['rbiw_' + c + p for c in n_vals] for p in self.personalized_suffices],
        }
        self.graph_labels = {
            'RB': [['CF (' + c + p + ')' for c in n_vals] for p in self.personalized_suffices],
            'MF': [['MF (' + c + p + ')' for c in n_vals] for p in self.personalized_suffices],
            'AR': [['AR (' + c + p + ')' for c in n_vals] for p in self.personalized_suffices],
            'IW': [['IW (' + c + p + ')' for c in n_vals] for p in self.personalized_suffices],
        }
        self.graph_order = ['AR', 'RB', 'IW', 'MF']
        self.rec_type2label = {
            'RB': 'CF',
            'MF': 'MF',
            'AR': 'AR',
            'IW': 'IW',
        }
        self.graph_data = {}
        self.bowtie_changes = {}
        self.plot_folder = 'plots'
        if self.personalized:
            self.plot_folder = os.path.join(self.plot_folder, 'personalized')
            self.graphs = {k: v for k, v in self.graphs.items() if k in personalized_recs}
            self.graph_labels = {k: v for k, v in self.graph_labels.items() if k in personalized_recs}
            self.graph_order = [v for v in self.graph_order if v in personalized_recs]
        if not os.path.exists(self.plot_folder):
            os.makedirs(os.path.join(self.plot_folder))

        self.load_graph_data()
        self.plot_file_types = [
            # '.png',
            '.pdf',
        ]

        if 'cp_size' in to_plot:
            self.plot('cp_size')
        if 'cp_count' in to_plot:
            self.plot('cp_count')
        if 'cc' in to_plot:
            self.plot('cc')
        if 'ecc' in to_plot:
            self.plot_ecc()
        if 'bow_tie' in to_plot:
            self.plot_bow_tie()
        if 'bow_tie_alluvial' in to_plot:
            self.plot_alluvial()

    def load_graph_data(self):
        for graph_type in self.graphs:
            for pidx, pers_type in enumerate(self.personalized_suffices):
                for graph_name in self.graphs[graph_type][pidx]:
                    fpath = os.path.join(self.stats_folder, graph_name + '.obj')
                    with open(fpath, 'rb') as infile:
                        graph_data = pickle.load(infile)
                    self.graph_data[graph_name] = graph_data

    def plot(self, prop):
        fig, ax = plt.subplots(1, figsize=(6, 3))
        bar_vals = []
        for graph_type in self.graph_order:
            bar_vals += [self.graph_data[graph_name][prop]
                         for graph_name in self.graphs[graph_type]]
            print(graph_type)
            for b, N in zip(bar_vals[-2:], n_vals):
                print('   ', N, ' ', b)
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

        if prop == 'cc':
            ylabel = 'Clustering Coefficient'
            ax.set_ylim(0, 0.5)
        elif prop == 'cp_count':
            ylabel = '# of components'
            # ax.set_ylim(0, 110)
        else:
            ylabel = 'Share of Nodes (%)'
            ax.set_ylim(0, 110)
        ax.set_ylabel(ylabel)

        plt.tight_layout()
        fpath = os.path.join(self.plot_folder, self.label + '_' + prop)
        for ftype in self.plot_file_types:
            plt.savefig(fpath + ftype)
        plt.close()

    def plot_ecc(self):
        print(self.label)
        for ecc_type in [
            # 'ecc_max',
            'ecc_median',
        ]:
            # plot_ecc_legend()
            # fig = plt.figure()
            # figlegend = plt.figure(figsize=(3, 2))
            # ax = fig.add_subplot(111)
            # objects = [
            #     matplotlib.patches.Patch(color='black', hatch='---'),
            #     matplotlib.patches.Patch(color='black', hatch='//')
            # ]
            # labels = ['N = 5', 'N = 20']
            # for pidx, patch in enumerate(objects):
            #     patch.set_fill(False)
            #
            # figlegend.legend(objects, labels, ncol=2)
            # figlegend.savefig('plots/legend_ecc_full.pdf', bbox_inches='tight')
            # cmd = 'pdfcrop --margins 5 ' +\
            #       'plots/legend_ecc_full.pdf plots/legend_ecc.pdf'
            # os.system(cmd)
            # print(cmd)
            for gidx, graph_type in enumerate(self.graph_order):
                fig, ax = plt.subplots(1, figsize=(6.25, 2.5))
                vals = [
                    self.graph_data[graph_name][ecc_type]
                    for graph_name in
                    [self.graphs[graph_type][0][4], self.graphs[graph_type][0][7]]
                ]
                print('   ', graph_type)
                for vidx, val, in enumerate(vals):
                    val = [100 * v / sum(val) for v in val]
                    print('       ', len(val) - 1)
                    # av = 0
                    # for vidx2, v in enumerate(val):
                    #     print('%.2f, ' % v, end='')
                    #     av += vidx2 * v
                    # print('average = %.2f' % (av/100))
                    # print()
                    bars = ax.bar(range(len(val)), val, color=self.colors[gidx], lw=2)
                    # Beautification
                    for bidx, bar in enumerate(bars):
                        bar.set_fill(False)
                        bar.set_hatch(self.hatches[vidx])
                        bar.set_edgecolor(self.colors[gidx])
                ax.set_xlim(0, 45)
                ax.set_ylim(0, 100)
                ax.set_xlabel('Eccentricity')
                ax.set_ylabel('% of Nodes')
                # plt.title(self.rec_type2label[graph_type])
                plt.tight_layout()
                fpath = os.path.join(self.plot_folder, self.label + '_' + graph_type + '_' + ecc_type)
                for ftype in self.plot_file_types:
                    plt.savefig(fpath + ftype)
                plt.close()

    def plot_bow_tie(self):
        # TODO FIXME legend plotting doesn't work
        # plot the legend in a separate plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # patches = [ax.bar([0], [0]) for i in range(7)]
        # for pidx, p in enumerate(patches):
        #     p[0].set_color(self.colors_set2[pidx])
        #     p[0].set_edgecolor('white')
        #     p[0].set_hatch(self.hatches[pidx % 4])
        # figlegend = plt.figure(figsize=(10.05, 0.475))
        # legend_labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
        # pdb.set_trace()
        # leg = figlegend.legend(patches, legend_labels, ncol=7)
        # leg.get_frame().set_linewidth(0.0)
        # fig.subplots_adjust(left=0.19, bottom=0.06, right=0.91, top=0.92,
        #                     wspace=0.34, hspace=0.32)
        # for ftype in self.plot_file_types:
        #     figlegend.savefig(os.path.join('plots', 'bowtie_legend' + ftype))

        fig, ax = plt.subplots(1, figsize=(6, 3))
        x_vals = [1, 2, 4, 5, 7, 8, 10, 11]
        bar_x = [x - 0.25 for x in x_vals]
        bar_vals = []
        for graph_type in self.graph_order:
            bar_vals += [self.graph_data[graph_name]['bow_tie']
                         for graph_name in self.graphs[graph_type]]
        bars = []
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
        bottom = np.zeros(2 * len(self.graph_order))
        for idx, label in enumerate(labels):
            vals = [v[idx] for v in bar_vals]
            p = plt.bar(bar_x, vals, bottom=bottom,
                        edgecolor='white', color=self.colors_set2[idx],
                        align='center')
            bottom += vals
            bars.append(p)
            for bidx, bar in enumerate(p):
                bar.set_hatch(self.hatches[idx % 4])

        # Beautification
        ax.set_ylabel('Component Membership')
        ax.set_xlim(0.25, 3 * len(self.graph_order))
        ax.set_xticks(bar_x)
        labels = [g for k in self.graph_order for g in self.graph_labels[k]]
        ax.set_xticklabels(labels, rotation='-50', ha='left')

        ax.set_ylim(0, 105)

        # plt.legend((p[0] for p in bars), labels, loc='center left',
        #            bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        # fig.subplots_adjust(right=0.73)
        # plt.show()
        fpath = os.path.join(self.plot_folder, self.label + '_' + 'bowtie')
        for ftype in self.plot_file_types:
            plt.savefig(fpath + ftype)
        plt.close()

    def plot_alluvial(self):
        """ produce an alluvial diagram (sort of like a flow chart) for the
        bowtie membership changes over N"""

        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        # indices = [0, 4, 9, 14, 19]
        indices = [0, 1, 2, 3, 4, 9, 14, 19]
        ind = u'    '
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
        with io.open('plots/alluvial/alluvial.html', encoding='utf-8')\
                as infile:
            template = infile.read().split('"data.js"')
        for graph_type in self.graph_order:
            for pidx, pers_type in enumerate(self.personalized_suffices):
                data_raw = [self.graph_data[graph_name]['bow_tie']
                            for graph_name in self.graphs[graph_type][pidx]]
                data = [[] for i in range(20)]
                for i, d in zip(indices, data_raw):
                    data[i] = d
                changes = [self.graph_data[graph_name]['bow_tie_changes']
                           for graph_name in self.graphs[graph_type][pidx]]
                changes = [[]] + [c.T for c in changes[1:]]

                print(self.label, graph_type)
                print()
                print(labels)
                for d, c in zip(data_raw, changes):
                    print('------------------------------------------------------------')
                    print(100*c)
                    for dd in d:
                        print('%.2f, ' % dd, end='')
                    print()
                    print()
                # /DEBUG
                # pdb.set_trace()

                fname = 'data_' + self.label + '_' + graph_type + pers_type + '.js'
                outdir = os.path.join(self.plot_folder, 'alluvial')
                if not os.path.exists(outdir):
                    os.makedirs(os.path.join(outdir))
                with io.open(os.path.join(outdir, fname), 'w', encoding='utf-8')\
                        as outfile:
                    outfile.write(u'var data = {\n')
                    outfile.write(ind + u'"times": [\n')
                    for iden, idx in enumerate(indices):
                        t = data[idx]
                        outfile.write(ind * 2 + u'[\n')
                        for jdx, n in enumerate(t):
                            outfile.write(ind * 3 + u'{\n')
                            outfile.write(ind * 4 + u'"nodeName": "Node ' +
                                          unicode(jdx) + u'",\n')
                            nid = unicode(iden * len(labels) + jdx)
                            outfile.write(ind * 4 + u'"id": ' + nid +
                                          u',\n')
                            outfile.write(ind * 4 + u'"nodeValue": ' +
                                          unicode(int(n * 100)) + u',\n')
                            outfile.write(ind * 4 + u'"nodeLabel": "' +
                                          labels[jdx] + u'"\n')
                            outfile.write(ind * 3 + u'}')
                            if jdx != (len(t) - 1):
                                outfile.write(u',')
                            outfile.write(u'\n')
                        outfile.write(ind * 2 + u']')
                        if idx != (len(data) - 1):
                            outfile.write(u',')
                        outfile.write(u'\n')
                    outfile.write(ind + u'],\n')
                    outfile.write(ind + u'"links": [\n')

                    for cidx, ci in enumerate(changes):
                        for mindex, val in np.ndenumerate(ci):
                            outfile.write(ind * 2 + u'{\n')
                            s = unicode((cidx - 1) * len(labels) + mindex[0])
                            t = unicode(cidx * len(labels) + mindex[1])
                            outfile.write(ind * 3 + u'"source": ' + s +
                                          ',\n')
                            outfile.write(ind * 3 + u'"target": ' + t
                                          + ',\n')
                            outfile.write(ind * 3 + u'"value": ' +
                                          unicode(val * 5000) + '\n')
                            outfile.write(ind * 2 + u'}')
                            if mindex != (len(ci) - 1):
                                outfile.write(u',')
                            outfile.write(u'\n')
                    outfile.write(ind + u']\n')
                    outfile.write(u'}')
                hfname = os.path.join(outdir, 'alluvial_' + self.label + '_' +\
                         graph_type + pers_type + '.html')
                with io.open(hfname, 'w', encoding='utf-8') as outfile:
                    outfile.write(template[0] + '"' + fname + '"' + template[1])

    def plot_alluvial_legend(self):
        # plot the legend in a separate plot
        fig = plt.figure()
        figlegend = plt.figure(figsize=(3, 2))
        ax = fig.add_subplot(111)
        objects = [matplotlib.patches.Patch(color=c) for c in self.colors_set2]
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']

        figlegend.legend(objects, labels, ncol=7, frameon=False)
        figlegend.savefig('plots/alluvial/alluvial_legend_full.pdf',
                          bbox_inches='tight')
        cmd = 'pdfcrop --margins 5 ' +\
              'plots/alluvial/alluvial_legend_full.pdf plots/alluvial/alluvial_legend.pdf'
        os.system(cmd)


def plot_selection_sizes(dataset):
    colors = ['#FFA500', '#FF0000', '#0000FF', '#05FF05', '#000000']
    fname = dataset + '.obj'
    fpath = os.path.join('data', dataset, 'stats_selection_size', fname)
    with open(fpath, 'rb') as infile:
        results = pickle.load(infile)
    plot_folder = os.path.join('plots', 'selection_size')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    pt2label = {
        '_personalized_min': 'Minimum',
        '_personalized_median': 'Median',
        '_personalized_max': 'Maximum',
    }
    pt2color = {
        '_personalized_min': colors[0],
        '_personalized_median': colors[1],
        '_personalized_max': colors[2],
    }
    for rec_type in results:
        print(rec_type)
        for N in results[rec_type]:
            print('   ', N)
            fig, ax = plt.subplots(1, figsize=(6, 4))
            for pt in results[rec_type][N]:
                print('       ', pt)
                data = results[rec_type][N][pt]
                plt.plot(data, lw=2, color=pt2color[pt], label=pt2label[pt])
            plt.legend()
            plt.xlabel('Selection Size (# of Items)')
            plt.ylabel('SCC Size (%)')
            plt.xlim(0, 151)
            plt.ylim(0, 100)
            plt.tight_layout()
            fname = 'selection_sizes_' + dataset + '.pdf'
            plt.savefig(os.path.join(plot_folder, fname))
            plt.close()


if __name__ == '__main__':
    n_vals = [
            '1',
            '2',
            '3',
            '4',
            '5',
            '10',
            '15',
            '20',
        ]
    to_plot = [
        # 'cp_count',
        # 'cp_size',
        # 'cc',
        'ecc',
        # 'bow_tie',
        # 'bow_tie_alluvial',
    ]
    personalized_recs = [
        'MF'
    ]
    personalized_suffix_list = [
        '_personalized_min',
        '_personalized_median',
        '_personalized_max',
    ]

    for sf in [
        'bookcrossing',
        'movielens',
        'imdb',
    ]:
        p = Plotter(sf, to_plot=to_plot, personalized=False)
        # p = Plotter(sf, to_plot=to_plot, personalized=True, personalized_suffices=personalized_suffix_list)
        # p.plot_alluvial_legend()
        # plot_selection_sizes(sf)
