﻿# -*- coding: utf-8 -*-

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
    stats = {}
    links_to_scc = {}
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
        res[N] = {k: [np.mean(v), np.median(v)] for k, v in bt2ratings.items()}
        res[N] = {k: v for k, v in res[N].items() if not np.isnan(v[0])}
        stats[N] = graph_stats(graph)
        links_to_scc[N] = get_links_to_scc(graph)
        nodes[N] = bt2nodes
    return res, nodes, stats, links_to_scc


def graph_stats(graph):
    clustering_coefficient = 0
    neighbors = {int(node): set([int(n) for n in node.out_neighbours()])
                 for node in graph.vertices()}
    for idx, node in enumerate(graph.vertices()):
        node = int(node)
        if len(neighbors[node]) < 2:
            continue
        edges = sum(len(neighbors[int(n)] & neighbors[node])
                    for n in neighbors[node])
        cc = edges / (len(neighbors[node]) * (len(neighbors[node]) - 1))
        clustering_coefficient += cc
    component, histogram = gt.label_components(graph)
    return [
        clustering_coefficient / graph.num_vertices(),
        len(histogram),
    ]


def get_links_to_scc(graph):
    count = 0
    count_scc = 0
    for node in graph.vertices():
        if not graph.vp['bowtie'][node] == 'IN':
            continue
        for nb in node.out_neighbours():
            count += 1
            if graph.vp['bowtie'][nb] == 'SCC':
                count_scc += 1
    if count == 0:
        return -1
    return count_scc / count


if __name__ == '__main__':
    datasets = [
        'movielens',
        'bookcrossing',
        'imdb'
    ]

    Ns = [
        # 2,
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
    result_stats = {}
    result_links_to_scc = {}
    for dataset in datasets:
        print(dataset)
        results[dataset] = {}
        result_nodes[dataset] = {}
        result_stats[dataset] = {}
        result_links_to_scc[dataset] = {}
        df = pd.read_pickle(os.path.join('data', dataset, 'item_stats.obj'))
        dataset_id2rating_count = {r['dataset_id']: r['rating_count']
                                   for ridx, r in df.iterrows()}
        dataset_id2title = {r['dataset_id']: r['original_title']
                            for ridx, r in df.iterrows()}
        for rec_type in rec_types:
            print('   ', rec_type)
            results[dataset][rec_type], result_nodes[dataset][rec_type], \
            result_stats[dataset][rec_type], result_links_to_scc[dataset][rec_type] = \
                get_stats(dataset, dataset_id2rating_count, dataset_id2title,
                          rec_type)

    out_folder = os.path.join('data', 'rating_stats')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with io.open(os.path.join(out_folder, 'counts.txt'), 'w', encoding='utf-8') as outfile:
        for dataset in datasets:
            outfile.write(u'%s\n' % (dataset))
            for rec_type in rec_types:
                outfile.write(u'    %s\n' % (rec_type))
                for N in Ns:
                    outfile.write(u'        %d (%d components, %.2f cc, %.2f links to scc)\n' % (
                        N, result_stats[dataset][rec_type][N][1],
                        result_stats[dataset][rec_type][N][0],
                        result_links_to_scc[dataset][rec_type][N]))
                    for label in results[dataset][rec_type][N]:
                        if label not in ['IN', 'SCC', 'OUT']:
                            continue
                        outfile.write(u'            %.2f\t%.2f\t%s\n' % (
                            results[dataset][rec_type][N][label][0],
                            results[dataset][rec_type][N][label][1],
                            label
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


