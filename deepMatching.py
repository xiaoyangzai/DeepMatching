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


