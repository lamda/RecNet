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
    def __init__(self, label, to_plot):
        self.label = label
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
            'RB': ['rb_' + c for c in n_vals],
            'MF': ['rbmf_' + c for c in n_vals],
            'AR': ['rbar_' + c for c in n_vals],
            'IW': ['rbiw_' + c for c in n_vals],
        }
        self.graph_labels = {
            'RB': ['CF (' + c + ')' for c in n_vals],
            'MF': ['MF (' + c + ')' for c in n_vals],
            'AR': ['AR (' + c + ')' for c in n_vals],
            'IW': ['IW (' + c + ')' for c in n_vals],
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
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
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
            for graph_name in self.graphs[graph_type]:
                graph_fname = graph_name
                fpath = os.path.join(self.stats_folder, graph_fname + '.obj')
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
            'ecc_max',
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
                    [self.graphs[graph_type][4], self.graphs[graph_type][7]]
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
                ax.set_xlim(0, 85)
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
            data_raw = [self.graph_data[graph_name]['bow_tie']
                        for graph_name in self.graphs[graph_type]]
            data = [[] for i in range(20)]
            for i, d in zip(indices, data_raw):
                data[i] = d
            changes = [self.graph_data[graph_name]['bow_tie_changes']
                       for graph_name in self.graphs[graph_type]]
            changes = [[]] + [c.T for c in changes[1:]]


            # DEBUG
            # data = [[0.38461538461538464, 0.054945054945054944, 0.0, 0.0, 0.0, 0.0, 99.56043956043956], [7.362637362637362, 0.32967032967032966, 0.35714285714285715, 10.714285714285714, 11.868131868131869, 2.82967032967033, 66.53846153846153], [23.818681318681318, 8.928571428571429, 23.708791208791208, 0.46703296703296704, 25.384615384615383, 17.225274725274726, 0.46703296703296704], [38.51648351648352, 56.73076923076923, 3.131868131868132, 0.0, 0.989010989010989, 0.4945054945054945, 0.13736263736263737], [27.335164835164836, 68.98351648351648, 2.6923076923076925, 0.0, 0.7142857142857143, 0.27472527472527475, 0.0], [21.181318681318682, 75.16483516483517, 2.857142857142857, 0.0, 0.6593406593406593, 0.13736263736263737, 0.0], [16.703296703296704, 80.13736263736264, 2.5824175824175826, 0.0, 0.43956043956043955, 0.13736263736263737, 0.0], [13.626373626373626, 83.35164835164835, 2.5824175824175826, 0.0, 0.38461538461538464, 0.054945054945054944, 0.0], [11.813186813186814, 85.46703296703296, 2.3626373626373627, 0.0, 0.3021978021978022, 0.054945054945054944, 0.0], [10.219780219780219, 87.3076923076923, 2.197802197802198, 0.0, 0.24725274725274726, 0.027472527472527472, 0.0], [9.093406593406593, 88.54395604395604, 2.1153846153846154, 0.0, 0.21978021978021978, 0.027472527472527472, 0.0], [7.6098901098901095, 91.4010989010989, 0.8241758241758241, 0.0, 0.10989010989010989, 0.054945054945054944, 0.0], [6.758241758241758, 92.33516483516483, 0.7692307692307693, 0.0, 0.10989010989010989, 0.027472527472527472, 0.0], [5.989010989010989, 93.1043956043956, 0.7692307692307693, 0.0, 0.10989010989010989, 0.027472527472527472, 0.0], [5.769230769230769, 93.37912087912088, 0.7142857142857143, 0.0, 0.10989010989010989, 0.027472527472527472, 0.0], [5.549450549450549, 94.45054945054945, 0.0, 0.0, 0.0, 0.0, 0.0], [5.1098901098901095, 94.89010989010988, 0.0, 0.0, 0.0, 0.0, 0.0], [4.8901098901098905, 95.10989010989012, 0.0, 0.0, 0.0, 0.0, 0.0], [4.697802197802198, 95.3021978021978, 0.0, 0.0, 0.0, 0.0, 0.0], [4.532967032967033, 95.46703296703296, 0.0, 0.0, 0.0, 0.0, 0.0]]
            # changes = [[], np.array([[  8.24175824e-04,   3.02197802e-03,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   5.49450549e-04,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [2.72527473e-01,   6.86263736e-01,   2.69230769e-02, 0.00000000e+00,   7.14285714e-03,   2.74725275e-03, 0.00000000e+00]]), np.array([[  1.01373626e-01,   1.71978022e-01,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   6.89835165e-01,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   9.06593407e-03,   1.78571429e-02, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [8.24175824e-04,   8.24175824e-04,   2.74725275e-03, 0.00000000e+00,   2.47252747e-03,   2.74725275e-04, 0.00000000e+00], [0.00000000e+00,   1.37362637e-03,   1.37362637e-03, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00]]), np.array([[  5.71428571e-02,   4.50549451e-02,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   8.73076923e-01,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   1.53846154e-02,   6.59340659e-03, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [5.49450549e-04,   2.74725275e-04,   2.74725275e-04, 0.00000000e+00,   1.09890110e-03,   2.74725275e-04, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   2.74725275e-04, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00]]), np.array([[  4.50549451e-02,   1.26373626e-02,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   9.33791209e-01,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   7.14285714e-03,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [2.74725275e-04,   8.24175824e-04,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   2.74725275e-04,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00], [0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00,   0.00000000e+00,   0.00000000e+00, 0.00000000e+00]])]
            print(self.label, graph_type)
            print()
            print(labels)
            for d, c in zip(data_raw, changes):
                print('------------------------------------------------------------')
                print(c)
                for dd in d:
                    print('%.2f, ' % dd, end='')
                print()
                print()
            # /DEBUG
            # pdb.set_trace()

            fname = 'data_' + self.label + '_' + graph_type + '.js'
            with io.open('plots/alluvial/' + fname, 'w', encoding='utf-8')\
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
            hfname = 'plots/alluvial/alluvial_' + self.label + '_' +\
                     graph_type + '.html'
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
    for sf in [
        'movielens',
        'bookcrossing',
        'imdb',
    ]:
        p = Plotter(sf, to_plot=to_plot)
        # p.plot_alluvial_legend()
