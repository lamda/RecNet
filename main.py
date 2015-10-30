# -*- coding: utf-8 -*-

from __future__ import division, print_function

import collections
import io
import numpy as np
import cPickle as pickle
import pdb

import decorators


class SimilarityMatrix(object):
    def __init__(self):
        pass


class RecommendationStrategy(object):
    def __init__(self):
        pass


class TopNRecommendationStrategy(RecommendationStrategy):
    def __init__(self):
        pass


class TopNDivRandomRecommendationStrategy(RecommendationStrategy):
    def __init__(self):
        pass


class TopNDivDiversifyRecommendationStrategy(RecommendationStrategy):
    def __init__(self):
        pass


class TopNDivExpRelRecommendationStrategy(RecommendationStrategy):
    def __init__(self):
        pass


class Recommender(object):
    def __init__(self):
        pass

    def get_data(self):
        raise NotImplementedError

    def get_similarity_matrix(self):
        raise NotImplementedError

    def get_recommendations(self):
        # get top X (50?) most similar recommendations
        # get standard recommendations
        # get the three diversified recommendation types


class ContentBasedRecommender(Recommender):
    def __init__(self):
        super(Recommender, ContentBasedRecommender).__init__()


class RatingBasedRecommender(Recommender):
    def __init__(self):
        super(Recommender, RatingBasedRecommender).__init__()


class MatrixFactorizationRecommender(RatingBasedRecommender):
    def __init__(self):
        pass


class InterpolationWeightRecommender(RatingBasedRecommender):
    def __init__(self):
        pass


class Graph(object):
    def __init__(self):
        pass

    def compute_stats(self):
        pass

    def save(self):
        pass


if __name__ == '__main__':
    # TODO: use     @decorators.Cached
    pass



