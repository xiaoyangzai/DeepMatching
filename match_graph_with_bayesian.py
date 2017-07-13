#!/usr/bin/python

import networkx as nx
import math
from scipy.special import comb
from bimatching.sparse_munkres import munkres
from bimatching.HungarianAlgorithm import maxWeightMatching
from collections import Counter


'''
	Notice : the usage of this bayesian method is in ./graph_matching.py
'''

def calcualte_probability_of_degree(G1,G2):
	'''
		Returns the probabilities of the degree 

		Parameters
		----------
		G1 : a networkx graph
		G2 : a networkx graph

		Returns
		-------
		degree_probability : A dictionary with degrees as keys and probabilities as values
	'''
	degree_probability = {}
	degree_of_G1 = G1.degree()
	degree_of_G2 = G2.degree()
	degree_all_nodes =  degree_of_G1.values()
	degree_all_nodes.extend(degree_of_G2.values())
	total_length = len(degree_all_nodes)
	degree_all_nodes_count = Counter(degree_all_nodes)
	for key in degree_all_nodes_count:
		degree_probability[key] = degree_all_nodes_count[key] * 1.0 / total_length
	return degree_probability	
	
def calculate_probability_of_distance(G1,G2):
	'''
		Returns the probabilities of the distance

		Parameters
		----------
		G1 : a networkx graph
		G2 : a networkx graph

		Returns
		-------
		distance_probability : A dictionary with distances as keys and probabilities as values
	'''
	node_pairs_shortest_length_G1 = nx.all_pairs_shortest_path_length(G1)	
	node_pairs_shortest_length_G2 = nx.all_pairs_shortest_path_length(G2)	
	distance_probability = {}
	distance = []  
	for key in node_pairs_shortest_length_G1:
		distance.extend(node_pairs_shortest_length_G1[key].values())
	for key in node_pairs_shortest_length_G2:
		distance.extend(node_pairs_shortest_length_G2[key].values())
	total_length = len(distance)		
	distance_all_nodes_count = Counter(distance)
	for key in distance_all_nodes_count:
		distance_probability[key] = distance_all_nodes_count[key] * 1.0 / total_length

	return distance_probability
		

def obtain_matches_with_bipartite_matching(r):
	'''
	Returns the maximum cardinality matching of nodes pairs in r list

	Parameters
	----------
	r : A list consists of the nodes pairs and cor corresponding cardinalities. 
		[[G1_node1,G2_node1,value_r],[G1_node2,G2_node2,value_r],[G1_node3,G2_node3,value_r],....]

	Returns
	-------
	matches : list 
		[[G1_node1,G2_node4,probability],[G1_node1,G2_node4,probability],...]
	'''
	print"size of r: %d * %d" % (len(r)/2,len(r)/2)
	node1 = list(set([item[0] for item in r]))
	node2 = list(set([item[1] for item in r]))
	len_node1 = len(node1)
	len_node2 = len(node2)
	print "len_node1: %d\tlen_node2:%d" % (len_node1,len_node2)
	proM = []
	for i in xrange(len_node1):
		temp = []
		for j in xrange(len_node2):
			for item in r:
				if item[0] == node1[i] and item[1] == node2[j]:
					temp.append(item[2])
					break
		proM.append(temp)
	values = [(i, j, 1 - proM[i][j]) for i in xrange(len_node1) for j in xrange(len_node2)]
	values_dict = dict(((i, j), v) for i, j, v in values)
	munkres_match = munkres(values)

	matches = []
	for p1, p2 in munkres_match:
		if p1 > len_node1 or p2 > len_node2:
			continue
		else:
			matches.append((node1[p1], node2[p2], 1 - values_dict[(p1,p2)]))
	return matches	
	
def matching_with_beyesian(G1,G2,sample_rate):
	'''
	Returns the maximum cardinality matching of nodes pairs between nodes of G1 and nodes of G2 by using the bayesian method with the nodes fingerprint.

	Parameters
	----------
	G1,G2 : A networkx graph

	sample_rate : float and less than 1
		Using the fix graph G, a sample process applies edge sampling to obtain G1 and G2. 

	Returns
	-------
	matches : list 
		[[G1_node1,G2_node4,probability],[G1_node1,G2_node4,probability],...]


	'''
	number_of_nodes = nx.number_of_nodes(G1)
	degree_of_G1 = G1.degree()
	degree_of_G1 = sorted(degree_of_G1.items(),key=lambda x:x[1],reverse=True)
	degree_of_G2 = G2.degree()
	degree_of_G2 = sorted(degree_of_G2.items(),key=lambda x:x[1],reverse=True)
	nodes_of_G1 = G1.nodes();
	nodes_of_G2 = G2.nodes();
	#calculate probability of each degree
	print "matching by beyesian.....start!!"
	print "calculate the probability of degree both in G1 and G2...."
	probability_of_degree = calcualte_probability_of_degree(G1,G2)
	print "calculate the probability of degree both in G1 and G2....ok"

	print "calculate the probability of distance both in G1 and G2...."
	probability_of_distance = calculate_probability_of_distance(G1,G2)
	print "calculate the probability of distance both in G1 and G2....ok"
	candidate_nodes_G1 = [item[0] for item in degree_of_G1]
	candidate_nodes_G2 = [item[0] for item in degree_of_G2]

	s_previous = []
	t_set = int(math.ceil(math.log(number_of_nodes,2)))
	for index in range(1,t_set+1):
		matches = []
		candidate_number = 2 ** index;
		if(candidate_number > number_of_nodes):
			candidate_number = number_of_nodes
		print "candidate number: %d" % candidate_number
		candidate_nodes_set_of_G1 = candidate_nodes_G1[:candidate_number]
		candidate_nodes_set_of_G2 = candidate_nodes_G2[:candidate_number]
		values = []
		#normalized posterior r set
		print "calculate the normalized posterior r set..."
		r = calculate_normalized_posterior(G1,G2,candidate_nodes_set_of_G1,candidate_nodes_set_of_G2,s_previous,probability_of_degree,probability_of_distance, sample_rate)
		print "calculate the normalized posterior r set...ok"
		#bipartite matching
		print "bipartite matching..."
		matches = obtain_matches_with_bipartite_matching(r)


		print "bipartite matching...ok"
		#descending order of their matching posterior in matching nodes set
		matches.sort(key=lambda x:x[2],reverse=True)
		print "=================matched nodes pairs=================="
		print matches
		print "======================================================"
		candidate_len = candidate_number / 2
		if candidate_len == 0:
			candidate_len = 1
		s_previous = [[item[0],item[1]] for item in matches][:candidate_len]
		#s_previous = [[item[0],item[1]] for item in matches if item[2] >= math.log(1,2)]

	return matches

def obtain_fingerprint_of_nodes(G1,G2,candidate_nodes_set_of_G1,candidate_nodes_set_of_G2,s_previous):
	'''
	Returns the fingerprints of nodes in 'candidata_nodes_set_of_G1' and 'candidate_nodes_set_of_G2'. 
	The a node fingerprint is defined as following:
	[degree of node, distance between the node and each node in 's_previous']

	Parameters
	----------

	G1,G2 : networkx graph 

	candidate_nodes_set_of_G1, candidate_nodes_set_of_G2 : List
		Consists of nodes that will be handled 

	s_previous : List
		Contains the anchor nodes which are used to obtain the distance
		
	Returns
	-------

	fingerprint_of_G1 : A dictionary
		A dictionary with the nodes in the 'candidate_nodes_set_of_G1' as the keys and the fingerprint as the values that are lists consisted of degree and distance to the nodes in 's_previous'.
		{G1_node1 : [degree,distance_1,distance_2,...,distance_m],G1_node2 : [degree, distance_1,distance_2,...,distance_m],....}

	fingerprint_of_G2 : A dictionary
		A dictionary with the nodes in the 'candidate_nodes_set_of_G2' as the keys and the fingerprint as the values that are lists consisted of degree and distance to the nodes in 's_previous'.
		{G1_node1 : [degree,distance_1,distance_2,...,distance_m],G1_node2 : [degree, distance_1,distance_2,...,distance_m],....}
	
	'''
	#fingerprint = {node1 : [degree,length1,length2,...],node2 : [degree,length1,..],...}	
	fingerprint_of_G1 = {} 
	fingerprint_of_G2 = {} 
	degree_of_G1 = G1.degree()
	degree_of_G2 = G2.degree()
	
	#generate the fingerprint of nodes in G1
	for node in candidate_nodes_set_of_G1:
		fingerprint = []
		path = []
		fingerprint.append(degree_of_G1[node])	  
		for item in s_previous:
			path = [p for p in nx.all_shortest_paths(G1,node,item[0])]
			if len(path) == 0:
				fingerprint.append(0)
			else:
				fingerprint.append(len(path[0]))
		fingerprint_of_G1[node] = fingerprint
	#generate the fingerprint of nodes in G2
	for node in candidate_nodes_set_of_G2:
		fingerprint = []
		path = []
		fingerprint.append(degree_of_G2[node])	  
		for item in s_previous:
			path = [p for p in nx.all_shortest_paths(G2,node,item[1])]
			if len(path) == 0:
				fingerprint.append(0)
			else:
				fingerprint.append(len(path[0]))
		fingerprint_of_G2[node] = fingerprint
	return fingerprint_of_G1,fingerprint_of_G2

def calculate_normalized_posterior(G1,G2,candidate_nodes_set_of_G1,candidate_nodes_set_of_G2,s_previous,probability_of_degree,probability_of_distance,sample_rate):
	''' 
	Returns matching probability of every nodes pairs

	Parameters
	----------

	G1,G2 : networkx graph

	candidate_nodes_set_of_G1, candidate_nodes_set_of_G2 : List
		Consists of nodes that will be handled 

	s_previous : List
		Contains the anchor nodes which are used to obtain the distance

	degree_probability : List
		A dictionary with degrees as keys and probabilities as values

	distance_probability : List
		A dictionary with distances as keys and probabilities as values

	sample_rate : float and less than 1
		Using the fix graph G, a sample process applies edge sampling to obtain G1 and G2. 
		
	Returns
	-------
	
	r_delt_set : List
		Consists of the matching probability of nodes pairs. The higher the probability value is, the greater similarity they have.

		[[G1_node1,G2_node2,probability],[G1_node3,G2_node1,probability],....]
	'''
	print "generate the fingerprint for both G1 and G2..."
	finger_of_G1,finger_of_G2 = obtain_fingerprint_of_nodes(G1,G2,candidate_nodes_set_of_G1,candidate_nodes_set_of_G2,s_previous)
	print "generate the fingerprint for both G1 and G2...ok"
	#arraibute
	degree_y = max(probability_of_degree)
	distance_y = max(probability_of_distance)
	nodes_of_G1 = G1.nodes();
	nodes_of_G2 = G2.nodes();
	number_of_nodes = nx.number_of_nodes(G1)
	m = len(s_previous)
	r_set = []
	for u1 in candidate_nodes_set_of_G1:
		for u2 in candidate_nodes_set_of_G2:
			posterior_u1 = 0
			posterior_u2 = 0
			A = 1.0 / number_of_nodes
			B = 1 - 1.0 / number_of_nodes
			sum_A_l = 0
			sum_A_m = 0
			sum_B_l = 0
			sum_B_m = 0
			x1_i = finger_of_G1[u1][0]
			x2_i = finger_of_G2[u2][0]
			for y in range(1,degree_y+ 1):
				q_x11_y = comb(y,x1_i)* (sample_rate**x1_i) * ((1 - sample_rate)**(y-x1_i)) 
				q_x21_y = comb(y,x2_i)* (sample_rate**x2_i) * ((1 - sample_rate)**(y-x2_i)) 
				if y in probability_of_degree.keys():
					sum_A_l += q_x11_y * q_x21_y * probability_of_degree[y] 
					posterior_u1 += probability_of_degree[y] * q_x11_y 
					posterior_u2 += probability_of_degree[y] * q_x21_y 
			B *= posterior_u1 *  posterior_u2
			A *= sum_A_l

				
			for i in range(1,m):
				x1_i = finger_of_G1[u1][i]
				x2_i = finger_of_G2[u2][i]
				q_x1i_y = 0
				q_x2i_y = 0
				for y in range(distance_y):
					if x1_i > y and x1_i < 2*y:
						q_x1i_y = comb(y,x1_i - y) * ((1 - sample_rate)**(x1_i - y)) * (sample_rate**x1_i)
						posterior_u1 += probability_of_distance[y] * q_x1i_y 
					if x2_i > y and x2_i < 2 * y:
						q_x2i_y = comb(y,x2_i - y) * ((1 - sample_rate)**(x2_i - y)) * (sample_rate**x2_i)
						posterior_u2 += probability_of_distance[y] * q_x2i_y 
					sum_A_m += q_x1i_y * q_x2i_y * probability_of_distance[y]
				B *= posterior_u1 * posterior_u2	
				
				A *=  (sum_A_m + posterior_u1 *  posterior_u2)
			#calculate the r(x,y)	
			if (A + B) == 0:
				r_set.append([u1,u2,0])
			else:
				r_set.append([u1,u2,(A * 1.0) / (A + B)])
	#normailized posterior r_set	
	r_delt_set = []
	denominator = 0
	sum_u1 = {} 
	sum_u2 = {} 
	for u1 in candidate_nodes_set_of_G1:
		temp_u1 = 0
		for v in r_set:
			if v[0] == u1 and v[1] in nodes_of_G2:
				temp_u1 += v[2]
		sum_u1[u1] = temp_u1

	for u2 in candidate_nodes_set_of_G2:
		temp_u2 = 0
		for v in r_set:
			if v[1] == u2 and v[0] in nodes_of_G1:
				temp_u2 += v[2]
		sum_u2[u2] = temp_u2
	
	for item in r_set:
		value = math.sqrt(sum_u1[item[0]] * sum_u2[item[1]])
		if value == 0:
			temp_r = 0
		else:
			temp_r = item[2] * 1.0 / value 
		r_delt_set.append([item[0],item[1],temp_r])
	
	return r_delt_set
