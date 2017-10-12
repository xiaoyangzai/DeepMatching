import math
import random as rd
from deepMatching import *
from refinement import *

def consistent_edges(matches, G1, G2):
    '''
    Extract all the consistent edge between the two matching graphs based on the matches
    :param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :return: A dict, where the keys are the consistent edges in G_1, the values are the consistent edges in G_2
    '''
    # matchmapping = dict([(string.atoi(match[0]), string.atoi(match[1])) for match in matches])
    matchmapping = dict([(match[0], match[1]) for match in matches])
    cedges = {}
    for edge in G1.edges():
        # print edge, (matchmapping.get(edge[0], -1), matchmapping.get(edge[1], -1))
        if G2.has_edge(matchmapping.get(edge[0], -1), matchmapping.get(edge[1], -1)):
            cedges[edge] = (matchmapping.get(edge[0]), matchmapping.get(edge[1]))
    return cedges


def match_consistent_degree(matches, G1, G2):
    '''
    Calculate the consistent degree for each match in matches, the consistent degree is define the number of 
    consistent edges connected to the matching node in each graph.
    :param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :return: A dict, where the keys are the matches, the values are the consistent degree defined above
    '''
    cedges = consistent_edges(matches, G1, G2)
    mcdeg = {}
    for G1_edge, G2_edge in cedges.items():
        mcdeg[(G1_edge[0], G2_edge[0])] = mcdeg.get((G1_edge[0], G2_edge[0]), 0) + 1
        mcdeg[(G1_edge[1], G2_edge[1])] = mcdeg.get((G1_edge[1], G2_edge[1]), 0) + 1
    return mcdeg

def mapping_consistency(matches, G1, G2):
    nodes1 = [n1 for n1, n2 in matches]
    nodes2 = [n2 for n1, n2 in matches]
    subG1 = G1.subgraph(nodes1)
    subG2 = G2.subgraph(nodes2)
    cedges = consistent_edges(matches, G1, G2)
    return graph_consistency(cedges, subG1, subG2)

def mapping_credibility(matches, G1, G2):
    nodes1 = [n1 for n1, n2 in matches]
    nodes2 = [n2 for n1, n2 in matches]
    subG1 = G1.subgraph(nodes1)
    subG2 = G2.subgraph(nodes2)
    seed_edge_consistency = mapping_consistency(matches, subG1, subG2)
    # print 'seed edge consistency:', seed_edge_consistency
    m, s = random_mapping_parameters_estimate(subG1, subG2,nodes1,nodes2)
    # print 'mean, std', m, s
    cred = (seed_edge_consistency - m) / s
    return cred

def obtain_seed_with_edges_credibility(matches,G1,G2,real_common_nodes,left_index,right_index,loop_index = 0):
	seed_nodes_list = maximum_consistency_matches(matches,G1,G2) 		
	if len(seed_nodes_list) == 0:
		return seed_nodes_list,0,0,0
	count = 0
	for item in seed_nodes_list:
		if item[0] == item[1]:
			count += 1
	seed_rate = float(count) / len(seed_nodes_list)
        cred = mapping_credibility(seed_nodes_list, G1, G2)
        print 'Credibility:', cred

	return seed_nodes_list,count,seed_rate,cred

def random_mapping_parameters_estimate(G1, G2,nodes1,nodes2):
    edge_consistency_list = []
    random_number = len(nodes1) if len(nodes1) <= len(nodes2) else len(nodes2) 

    for i in range(100):
        rd.shuffle(nodes1)
        rd.shuffle(nodes2)
        matches = [(nodes1[j], nodes2[j]) for j in range(random_number)]
        edge_consistency_list.append(mapping_consistency(matches, G1, G2))
    edge_consistency_list = np.array(edge_consistency_list)
    return (np.mean(edge_consistency_list), np.std(edge_consistency_list))

def maximum_consistency_matches(matches, G1, G2, nodenum_limit=7, cth = 7):
    '''
    Extract a sublist of matches in order to maximize the consistency between the two subgraphs. The two subgraphs 
    are extracted from the two matching graphs according to the sublist of matches. The consistency between two 
    graphs is defined as the ratio of the number of consistent edges between the two graphs over the maximun number 
    of edges of the two graphs. 
    :param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
    :param G1: the matching graph
    :param G2: the matching graph
    :param nodenum_limit: The minimum number of nodes in each subgraph. 
    :param cth: The consistency threshold, a consistency below this threshold suggests a failed matching. 
    :return: A sublist of matches. An empty list represents a failed matching. 
    '''
    mcdeg = match_consistent_degree(matches, G1, G2)
    seeds = []
    for match, deg in mcdeg.items():
        if deg > cth:
            seeds.append(match)
    if len(seeds) < nodenum_limit:
        seeds = []
    return seeds

def graph_consistency(consistent_edges, G1, G2):
    '''
    Calculate the consistency between two graphs
    :param consistent_edges: A dict, composed of the consistent edges
    :param G1: the matching graph
    :param G2: the matching graph
    :return: the consistency, a float value between 0.0 to 1.0. 
    '''
    G1_size = G1.size()
    G2_size = G2.size()
    if G1_size == 0 or G2_size ==0:
        return 0.0
    consistent_edge_count = 0.0
    for edge in G1.edges():
        if edge in consistent_edges or (edge[1], edge[0]) in consistent_edges:
            consistent_edge_count += 1.0
    return consistent_edge_count*1.0/max(G2_size, G1_size)
