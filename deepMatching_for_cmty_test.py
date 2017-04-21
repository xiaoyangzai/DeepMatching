#!/usr/bin/python

import sys

import networkx as nx
import deepMatching_for_cmty as dc

G = dc.load_graph_from_file(sys.argv[1],delimiter = ' ')

nodes = [5,6,7,8]
print nodes
edges = dc.obtain_edges_of_cmty(G,nodes)
print edges

outdegree = dc.obtain_degree_extern_cmty(G,nodes)
print "outdegree: ",
print outdegree
degree_nodes = dc.obtain_degree_inter_cmty(G,nodes)
print "inter degree: ",
print degree_nodes
average_degree = float(sum(degree_nodes))/len(degree_nodes)	
print "average: %.5f" % average_degree

midian_degree = dc.obtain_midian_list(degree_nodes)
print "midian: %.5f" % midian_degree


dc.draw_networkx(G)
