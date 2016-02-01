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
    def __init__(self, label, use_sample=True):
        self.label = label
        self.stats_folder = os.path.join('data', self.label, 'stats')
        self.use_sample = use_sample
        self.div_types = [
            '',
            '_div_exprel',
            '_div_diversify',
            '_div_random',
        ]
        self.graph_types = len(self.div_types)
        self.Ns = [
            '5',
            '10',
        ]
        self.graphs = {
            # 'CB': ['cb_' + c + d for c in self.Ns for d in self.div_types],
            'RB': ['rb_' + c + d for c in self.Ns for d in self.div_types],
            'MF': ['rbmf_' + c + d for c in self.Ns for d in self.div_types],
            'AR': ['rbar_' + c + d for c in self.Ns for d in self.div_types],
            'IW': ['rbiw_' + c + d for c in self.Ns for d in self.div_types],
        }
        self.div_labels = [
            '',
            ', ExpRel',
            ', Diversify',
            ', Random',
        ]
        self.graph_labels = {
            # 'CB': ['CB (' + c + d + ')' for c in self.Ns for d in self.div_labels],
            'RB': ['RB (' + c + d + ')' for c in self.Ns for d in self.div_labels],
            'MF': ['MF (' + c + d + ')' for c in self.Ns for d in self.div_labels],
            'AR': ['AR (' + c + d + ')' for c in self.Ns for d in self.div_labels],
            'IW': ['IW (' + c + d + ')' for c in self.Ns for d in self.div_labels],
        }
        self.graph_data = {}
        self.bowtie_changes = {}
        self.plot_folder = 'plots'
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)
        self.load_graph_data()
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
        self.hatches = ['', 'xxx', '///', '---']
        self.linestyles = ['-', '--', ':', '-.']
        self.bar_x = range(1, self.graph_types+1)\
            + range(self.graph_types+2, 2*self.graph_types+2)
        self.plot_file_types = [
            '.png',
            # '.pdf',
        ]

        # for prop in [
        #     'cp_size',
        #     'cp_count',
        #     'cc',
        # ]:
        #     self.plot(prop)
        # self.plot_ecc()
        self.plot_bow_tie()
        # self.plot_alluvial()

    def load_graph_data(self):
        for graph_type in self.graphs:
            for graph_name in self.graphs[graph_type]:
                if self.use_sample:
                    graph_fname = graph_name[:3] + 'sample_' + graph_name[3:]
                else:
                    graph_fname = graph_name
                fpath = os.path.join(self.stats_folder, graph_fname + '.obj')
                with open(fpath, 'rb') as infile:
                    graph_data = pickle.load(infile)
                self.graph_data[graph_name] = graph_data

    def plot(self, prop):
        for graph_type in self.graphs:
            fig, ax = plt.subplots(1, figsize=(5, 5))
            bar_vals = [self.graph_data[graph_name][prop]
                        for graph_name in self.graphs[graph_type]]
            bars = ax.bar(self.bar_x, bar_vals, align='center')

            # Beautification
            for bidx, bar in enumerate(bars):
                bar.set_fill(False)
                bar.set_hatch(self.hatches[bidx % 4])
                bar.set_edgecolor(self.colors[bidx % 4])

            ax.set_xlim(0.25, 2 * self.graph_types + 1.75)
            ax.set_xticks([x - 0.25 for x in self.bar_x])
            for tic in ax.xaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_xticklabels(self.graph_labels[graph_type], rotation='-50',
                               ha='left')

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
            # plt.show()
            fpath = os.path.join(self.plot_folder, self.label + '_' + prop + '_' + graph_type)
            for ftype in self.plot_file_types:
                plt.savefig(fpath + ftype)

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

        for graph_type in self.graphs:
            Ns = [5, 10]
            c = '#FF0000'
            # xlim = (0, 150)
            # c = '#0000FF'
            xlim = (0, 50)
            plot_ecc_legend(graph_type, c)
            for Nidx, N in enumerate(Ns):
                graphs = [g for g in self.graphs[graph_type] if unicode(N) in g]
                fig, ax = plt.subplots(1, figsize=(7*0.7, 4*0.7))
                for gidx, graph_name in enumerate(graphs):
                    try:
                        ecc = self.graph_data[graph_name]['lc_ecc']
                    except KeyError:
                        print(graph_name)
                        ecc = [e/sum(range(15)) for e in range(15)]
                    x = range(len(ecc))
                    label = self.graph_labels[graph_type][self.graph_types * Nidx + gidx]
                    ax.plot(x, ecc, linewidth=2, linestyle=self.linestyles[gidx],
                            color=c, label=label)

                # Beautification
                ax.set_xlabel('Eccentricity')
                ax.set_ylabel('% of Nodes')
                ax.set_xlim(xlim)
                ax.set_ylim(0, 100)

                # plt.legend(loc=0)
                plt.tight_layout()
                fpath = os.path.join(self.plot_folder, 'ecc_' + graph_type +
                                     '_' + unicode(N))
                for ftype in self.plot_file_types:
                    plt.savefig(fpath + ftype)

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
            fpath = os.path.join(self.plot_folder, 'bowtie_' + graph_type)
            for ftype in self.plot_file_types:
                plt.savefig(fpath + ftype)

    def plot_alluvial(self):
        """ produce an alluvial diagram (sort of like a flow chart) for the
        bowtie membership changes over N"""
        fpath = os.path.join(self.stats_folder, 'bowtie_changes.obj')
        with open(fpath, 'rb') as infile:
            self.bowtie_changes = pickle.load(infile)
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
        base_graphs = [
            'CF_6',
            'CF_12',
            'CB_5',
            'CB_10',
        ]
        for g1 in base_graphs:
            for g2_suffix in self.div_types[1:]:
                g2 = g1 + g2_suffix
                changes = self.bowtie_changes[g1][g2]
                with io.open('plots/alluvial/alluvial.html',
                             encoding='utf-8-sig') as infile:
                    template = infile.read().split('"data.js"')
                fname = 'data_' + g1 + '_' + g2 + '.js'
                data = [
                    self.graph_data[g1]['bow_tie'],
                    self.graph_data[g2]['bow_tie']
                ]
                ind = u'    '
                with io.open('plots/alluvial/' + fname, 'w', encoding='utf-8')\
                        as outfile:
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
                hfname = 'plots/alluvial/alluvial_' + g1 + '_' + g2 + '.html'
                with io.open(hfname, 'w', encoding='utf-8') as outfile:
                    outfile.write(template[0] + '"' + fname + '"' + template[1])


if __name__ == '__main__':
    for sf in [
        os.path.join('movielens'),
        # os.path.join('bookcrossing'),
    ]:
        p = Plotter(sf, use_sample=False)
