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
            'RB': ['RB (' + c + ')' for c in n_vals],
            'MF': ['MF (' + c + ')' for c in n_vals],
            'AR': ['AR (' + c + ')' for c in n_vals],
            'IW': ['IW (' + c + ')' for c in n_vals],
        }
        self.graph_order = ['AR', 'RB', 'IW', 'MF']
        self.graph_data = {}
        self.bowtie_changes = {}
        self.plot_folder = 'plots'
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        self.load_graph_data()
        self.plot_file_types = [
            '.png',
            # '.pdf',
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
        fig, ax = plt.subplots(1, figsize=(10, 5))
        bar_vals = []
        for graph_type in self.graph_order:
            bar_vals += [self.graph_data[graph_name][prop]
                         for graph_name in self.graphs[graph_type]]
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
            ylabel = 'CC'
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
        def plot_ecc_legend(label, color):
            # plot the legend in a separate plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            lines = [plt.plot([0, 1], [1, 2])[0] for i in range(self.graph_types)]
            for lidx, l in enumerate(lines):
                l.set_color(color)
                l.set_linestyle(self.linestyles[lidx])
                l.set_linewidth(3)
            figlegend = plt.figure(figsize=(2, 2.5))
            labels = ['Unmodified'] + [l[2:] for l in self.div_labels[1:]]
            leg = figlegend.legend(lines, labels, ncol=1)
            # leg.get_frame().set_linewidth(0.0)
            fig.subplots_adjust(left=0.19, bottom=0.06, right=0.91, top=0.92,
                                wspace=0.34, hspace=0.32)
            for ftype in self.plot_file_types:
                figlegend.savefig(os.path.join('plots', 'ecc_legend_' +
                                               label + ftype))

        # import seaborn.apionly as sns  # apionly doesn't change style
        # for gidx, graph_type in enumerate(self.graph_order):
        #     fig, ax = plt.subplots(1, figsize=(10, 5))
        #     vals = [self.graph_data[graph_name]['lc_ecc']
        #             for graph_name in self.graphs[graph_type]]
        #     for vidx, val, in enumerate(vals):
        #         vs = [[idx] * v for idx, v in enumerate(val) if v]
        #         vs = reduce(lambda x, y: x + y, vs)
        #         ax = sns.distplot(vs, bins=range(40), color=self.colors[gidx],
        #                           kde_kws={'bw': 0.5},
        #                           hist_kws={'align': 'left', 'lw': 3})
        #     bars = [rect for rect in ax.get_children()
        #             if isinstance(rect, matplotlib.patches.Rectangle)]
        #     # Beautification
        #     for bidx, bar in enumerate(bars[:40]):
        #         bar.set_fill(False)
        #         bar.set_hatch(self.hatches[0])
        #         bar.set_edgecolor(self.colors[gidx])
        #     for bidx, bar in enumerate(bars[40:-1]):
        #         bar.set_fill(False)
        #         bar.set_hatch(self.hatches[1])
        #         bar.set_edgecolor(self.colors[gidx])
        #     ax.set_xlim(0, 40)
        #     plt.tight_layout()
        #     # plt.show()
        #     fpath = os.path.join(self.plot_folder, self.label + '_' + graph_type + '_ecc')
        #     for ftype in self.plot_file_types:
        #         plt.savefig(fpath + ftype)
        #     plt.close()

        for gidx, graph_type in enumerate(self.graph_order):
            fig, ax = plt.subplots(1, figsize=(10, 5))
            vals = [self.graph_data[graph_name]['lc_ecc']
                    for graph_name in self.graphs[graph_type]]

            for vidx, val, in enumerate(vals):
                val = [100 * v / sum(val) for v in val]
                bars = ax.bar(range(len(val)), val, color=self.colors[gidx], lw=2)
                # Beautification
                for bidx, bar in enumerate(bars):
                    bar.set_fill(False)
                    bar.set_hatch(self.hatches[vidx])
                    bar.set_edgecolor(self.colors[gidx])
            ax.set_xlim(0, 40)
            ax.set_ylim(0, 100)
            plt.tight_layout()
            fpath = os.path.join(self.plot_folder, self.label + '_' + graph_type + '_ecc')
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

        bar_x = [x - 0.25 for x in self.bar_x]
        for graph_type in self.graphs:
            fig, ax = plt.subplots(1, figsize=(10, 5))
            bar_vals = [self.graph_data[graph_name]['bow_tie']
                        for graph_name in self.graphs[graph_type]]
            labels = ['inc', 'scc', 'outc',
                      'in_tendril', 'out_tendril', 'tube', 'other']
            bars = []
            bottom = np.zeros(2 * self.graph_types)
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
            ax.set_xlim(0.25, 2 * self.graph_types + 1.75)
            ax.set_xticks(bar_x)
            ax.set_xticklabels(self.graph_labels[graph_type],
                               rotation='-50', ha='left')

            ax.set_ylim(0, 105)

            # plt.legend((p[0] for p in bars), labels, loc='center left',
            #            bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            # fig.subplots_adjust(right=0.73)
            # plt.show()
            fpath = os.path.join(self.plot_folder, self.label + '_' +
                                 'bowtie_' + graph_type)
            for ftype in self.plot_file_types:
                plt.savefig(fpath + ftype)
            plt.close()

    def plot_alluvial(self):
        """ produce an alluvial diagram (sort of like a flow chart) for the
        bowtie membership changes over N"""
        fpath = os.path.join(self.stats_folder, 'bowtie_changes.obj')
        with open(fpath, 'rb') as infile:
            self.bowtie_changes = pickle.load(infile)
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']

        graphs = [
            'rb',
            'rbmf',
            'rbiw',
            'rbar',
        ]
        for N in self.Ns:
            base_graphs = [g + '_' + N for g in graphs]
            for g1 in base_graphs:
                for g2_suffix in self.div_types[1:]:
                    g2 = g1 + g2_suffix
                    changes = self.bowtie_changes[g1][g2]
                    with io.open('plots/alluvial/alluvial.html',
                                 encoding='utf-8-sig') as infile:
                        template = infile.read().split('"data.js"')
                    fname = self.label + '_' + 'data_' + g1 + '_' + g2 + '.js'
                    data = [
                        self.graph_data[g1]['bow_tie'],
                        self.graph_data[g2]['bow_tie']
                    ]
                    ind = u'    '
                    with io.open('plots/alluvial/' + fname, 'w',
                                 encoding='utf-8')as outfile:
                        outfile.write(u'var data = {\n')
                        outfile.write(ind + u'"times": [\n')
                        for iden, d in enumerate(data):
                            t = d
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
                            if iden != (len(data) - 1):
                                outfile.write(u',')
                            outfile.write(u'\n')
                        outfile.write(ind + u'],\n')

                        outfile.write(ind + u'"links": [\n')
                        for cidx, ci in enumerate([changes]):
                            for mindex, val in np.ndenumerate(ci):
                                outfile.write(ind * 2 + u'{\n')
                                s = unicode(cidx * len(labels) + mindex[0])
                                t = unicode((cidx+1) * len(labels) + mindex[1])
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
                             g1 + '_' + g2 + '.html'
                    with io.open(hfname, 'w', encoding='utf-8') as outfile:
                        outfile.write(template[0] + '"' + fname + '"' +
                                      template[1])


if __name__ == '__main__':
    n_vals = [
            '5',
            '20',
        ]
    to_plot = [
        # 'cp_size',
        # 'cp_count',
        # 'cc',
        'ecc',
        # 'bow_tie',
        # 'bow_tie_alluvial',
    ]
    for sf in [
        'movielens',
        'bookcrossing',
    ]:
        p = Plotter(sf, to_plot=to_plot)
