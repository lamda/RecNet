# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import pdb

from main import DATA_BASE_FOLDER


def check_recommendation_duplicates(dataset):
    data_folder = os.path.join(DATA_BASE_FOLDER, dataset)
    graph_folder = os.path.join(data_folder, 'graphs')

    graph_files = os.listdir(graph_folder)
    for graph_file in graph_files:
        left = None
        right = []
        with open(os.path.join(graph_folder, graph_file)) as infile:
            for line in infile:
                l, r = line.strip().split('\t')
                if left != l:
                    if len(set(right)) != len(right):
                        print(left, right)
                        pdb.set_trace()
                    left = l
                    right = [r]
                else:
                    right.append(r)


if __name__ == '__main__':
    check_recommendation_duplicates('movielens')