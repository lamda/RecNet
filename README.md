# RecNet
A framework to evalute the static and dynamic navigability of recommendation networks.

INPUT: rating data, i.e., triples of (user, item, rating)
OUTPUT: measures for the static and dynamic navigability (such as path length analysis, component analysis and simulation of navigation scenarios)

This framework computers four types of recommendations, namely association rules, collaborative filtering, interpolation weights and matrix factorization.
The static navigability is evaluated in terms of clustering coefficients, component sizes, a bow-tie analysis and an eccentricity analysis.
The dynamic navigability is evaluated by conducting simulations of navigational scenarios inspired by information seeking models from the literature, namely point-to-point search, berrypicking and information foraging.
