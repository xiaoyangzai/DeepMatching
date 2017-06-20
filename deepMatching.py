#!/usr/bin/python
import numpy as np
import time as ti
import networkx as nx
from functools import partial
from scipy.io import loadmat
import matplotlib.pyplot as plt
from cpdcore import DeformableRegistration, RigidRegistration, AffineRegistration
from sklearn.decomposition import PCA
from node2vec import learn_nodevec
import time
from graph_matching import sample_graph
import sys
from deepwalk import __main__ as deepwalk
from bimatching.sparse_munkres import munkres

def visualize(X, Y, ax=None, words = None):
    pass

def nodes_embedding_deepwalk(G, p=1, q=1, dimensions=128):
    # Extract the features for each node using deepwalk
	print "start transport the graph to deepwalk,edges: %d\tnodes: %d" % (G.number_of_edges(),G.order())

	model = deepwalk.process(G.edges(), dimensions=dimensions, number_walks = 20)
	nodes = [word for word, vcab in model.vocab.iteritems()]
	inds = [vcab.index for word, vcab in model.vocab.iteritems()]
	X = model.syn0[inds]
	return nodes,X

def nodes_embedding(G, p=1, q=1, dimensions=128):
	# Extract the features for each node using node2vec
	#model = learn_nodevec(G, dimensions=dimensions, argp=p, argq=q, num_walks=100)
	model = deepwalk.process(G.edges(), dimensions=dimensions, number_walks = 30)
	nodes = [word for word, vcab in model.wv.vocab.iteritems()]
	inds = [vcab.index for word, vcab in model.wv.vocab.iteritems()]
	X = model.wv.syn0[inds]
	# for node in nodes:
	#     X.append(model.wv.word_vec(node))
	# i = 0
	# for node in nodes:
	#     print node, model.wv.word_vec(node)
	#     print X[i]
	#     i = i+1
	# print nodes
	return nodes, X



def match_probalities(G1, G2, p=1, q=1, dimensions=128):
	# match the nodes according to the node feature based on Coherent Point Drift
	nodes1, X = nodes_embedding(G1, p = p, q=q, dimensions=dimensions)
	nodes2, Y = nodes_embedding(G2, p = p, q=q, dimensions=dimensions)
	reg = RigidRegistration(Y, X)
	callback = partial(visualize)
	reg.register(callback)
	P = reg.P
	return (nodes1, nodes2, P)


def bipartite_matching(G1, G2, p=1, q=1, dimensions=128):
	node1, node2, proM = match_probalities(G1, G2, p=p, q=q, dimensions=dimensions)
	M, N = proM.shape
	values = [(i, j, 1 - proM[i][j]) for i in xrange(M) for j in xrange(N) if proM[i][j] > 1e-2]
	values_dict = dict(((i, j), v) for i, j, v in values)
	munkres_match = munkres(values)
	matches = []
	for p1, p2 in munkres_match:
		if p1 > len(node1) or p2 > len(node2):
			continue
		else:
			matches.append((node1[p1], node2[p2], 1 - values_dict[(p1,p2)]))
	return matches





def nodes_embedding_node2vec(G, p=1, q=1, dimensions=128):
    # Extract the features for each node using node2vec
    model = learn_nodevec(G, dimensions=dimensions, argp=p, argq=q, num_walks=100)
    nodes = [word for word, vcab in model.vocab.iteritems()]
    inds = [vcab.index for word, vcab in model.vocab.iteritems()]
    X = model.syn0[inds]
    return nodes,X


def map_prob_maxtrix(G1, G2, p=1, q=1, dimensions=128):
    # match the nodes according to the node feature based on Coherent Point Drift
    nodes1, X = nodes_embedding_deepwalk(G1, p = p, q=q, dimensions=dimensions)
    nodes2, Y = nodes_embedding_deepwalk(G2, p = p, q=q, dimensions=dimensions)
    reg = AffineRegistration(Y, X)
    callback = partial(visualize)
    reg.register(callback)
    P = reg.P
    return (nodes1, nodes2, P)


