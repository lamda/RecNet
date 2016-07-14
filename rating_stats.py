# -*- coding: utf-8 -*-

from __future__ import division, print_function

import cPickle as pickle
import graph_tool.all as gt
import io
import numpy as np
import os
import pandas as pd
import pdb


def get_stats(dataset, dataset_id2rating_count, dataset_id2title, rec_type):
    folder = os.path.join('data', dataset, 'graphs')
    res = {}
    nodes = {}
    for N in Ns:
        print('       ', N)
        gt_file = os.path.join(folder, '%s_%d.gt' % (rec_type, N))
        graph = gt.load_graph(gt_file, fmt='gt')

        # # DEBUG
        # title2dataset_id = {v: k for k, v in dataset_id2title.items()}
        # dataset_id2node = {graph.vp['name'][n]: n for n in graph.vertices()}
        # title2node = {title: dataset_id2node[title2dataset_id[title]] for title in dataset_id2title.values()}
        # #/DEBUG

        dataset_id2bow_tie = {graph.vp['name'][n]: graph.vp['bowtie'][n]
                              for n in graph.vertices()}
        bt2ratings = {l: [] for l in bt_labels}
        bt2nodes = {l: [] for l in bt_labels}
        for did, bt in dataset_id2bow_tie.items():
            bt2ratings[bt].append(dataset_id2rating_count[did])
            bt2nodes[bt].append(dataset_id2title[did])
        res[N] = {k: np.mean(v) for k, v in bt2ratings.items()}
        res[N] = {k: v for k, v in res[N].items() if not np.isnan(v)}
        nodes[N] = bt2nodes
    return res, nodes

if __name__ == '__main__':
    datasets = [
        'movielens',
        'bookcrossing',
        'imdb'
    ]

    Ns = [
        5,
        20
    ]

    rec_types = [
        'rbar',
        'rb',
        'rbiw',
        'rbmf'
    ]
    bt_labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
    results = {}
    result_nodes = {}
    for dataset in datasets:
        print(dataset)
        results[dataset] = {}
        result_nodes[dataset] = {}
        df = pd.read_pickle(os.path.join('data', dataset, 'item_stats.obj'))
        dataset_id2rating_count = {r['dataset_id']: r['rating_count']
                                   for ridx, r in df.iterrows()}
        dataset_id2title = {r['dataset_id']: r['original_title']
                            for ridx, r in df.iterrows()}
        for rec_type in rec_types:
            print('   ', rec_type)
            results[dataset][rec_type], result_nodes[dataset][rec_type] = \
                get_stats(dataset, dataset_id2rating_count, dataset_id2title, rec_type)

    out_folder = os.path.join('data', 'rating_stats')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with io.open(os.path.join(out_folder, 'counts.txt'), 'w', encoding='utf-8') as outfile:
        for dataset in datasets:
            outfile.write(u'%s\n' % (dataset))
            for rec_type in rec_types:
                outfile.write(u'    %s\n' % (rec_type))
                for N in Ns:
                    outfile.write(u'        %d\n' % (N))
                    for label in results[dataset][rec_type][N]:
                        outfile.write(u'            %.2f\t%s\n' % (
                            results[dataset][rec_type][N][label], label
                        ))

    for dataset in datasets:
        for rec_type in rec_types:
            for N in Ns:
                outpath = os.path.join(out_folder, '%s_%s_%d.txt' % (dataset, rec_type, N))
                with io.open(outpath, 'w', encoding='utf-8') as outfile:
                    for label, nodes in result_nodes[dataset][rec_type][N].items():
                        outfile.write(u'%s %s\n' % (label, u'%' * 64))
                        for n in sorted(nodes):
                            outfile.write(u'    %s\t[%s]\n' % (n, label))


