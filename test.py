# -*- coding: utf-8 -*-

from __future__ import division, print_function

import graph_tool.all as gt
import os
import pdb

# from main import DATA_BASE_FOLDER


# def check_recommendation_duplicates(dataset):
#     data_folder = os.path.join(DATA_BASE_FOLDER, dataset)
#     graph_folder = os.path.join(data_folder, 'graphs')
#
#     graph_files = os.listdir(graph_folder)
#     for graph_file in graph_files:
#         left = None
#         right = []
#         with open(os.path.join(graph_folder, graph_file)) as infile:
#             for line in infile:
#                 l, r = line.strip().split('\t')
#                 if l == r:
#                     print(graph_file, line)
#                     # pdb.set_trace()
#                 if left != l:
#                     if len(set(right)) != len(right):
#                         print(graph_file, line, left, right)
#                         pdb.set_trace()
#                     left = l
#                     right = [r]
#                 else:
#                     right.append(r)

def prepare_test_graphs():
    from graph import Graph
    for dataset in ['test_paper', 'test_wiki']:
        g = Graph(dataset=dataset, fname='graph_5', refresh=True)
        g.load_graph(refresh=True)

if __name__ == '__main__':
    # check_recommendation_duplicates('movielens')
    prepare_test_graphs()
