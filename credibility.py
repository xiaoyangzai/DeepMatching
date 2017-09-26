from deepMatching import *
import networkx as nx


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
    #matchmapping = matches
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


def maximum_consistency_matches(matches, G1, G2, nodenum_limit=7, cth = 2.0):
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
	:return: A sublist of matches. A empty list represents a failed matching. 
	'''
	mcdeg = match_consistent_degree(matches, G1, G2)
	seeds = []
	for match, deg in mcdeg.items():
		if deg > cth:
			seeds.append(match)
	if len(seeds) < nodenum_limit:
		seeds = []
	return seeds

def z_score(gcr, subG1, subG2, k_c):
	subG1_m = nx.number_of_edges(subG1)
	subG2_m = nx.number_of_edges(subG2)
	subG1_n = nx.number_of_nodes(subG1)
	subG2_n = nx.number_of_nodes(subG2)
	if subG1_n <= 1 or subG2_n <= 1 or subG1_m == 0 or subG2_m == 0:
		return 0.0
	density_subG1 = (2 * subG1_m * 1.0) / (subG1_n * (subG1_n - 1))
	density_subG2 = (2 * subG2_m * 1.0) / (subG2_n * (subG2_n - 1))
	density = density_subG1 if density_subG1 <= density_subG2 else density_subG2
	if density == 1:
		return 0.0
	z = 0.0
	p1 = (9 * density) / (1 - density)
	p2 = (9 * (1 - density)) / density
	p = p1 if p1 >= p2 else p2
	if k_c >= p:
		z = ((gcr - density) * 1.0) / (density * (1 - density))
	return z

def plot_edge_conssitencies(edge_consists,left_index,right_index):
	'''
	Used to plot the consistency sequence
	:param edge_consists: 
	:return: 
	'''
	fig, ax = plt.subplots()
	ax.plot(edge_consists, '.-')
	ax.set_xlabel('Match index (ordered by consistent degree)')
	#ax.set_ylabel('Consistent edge ratio')
	ax.set_ylabel('Z_score')
	plt.tight_layout()
	fig.savefig("../results/week4_community_new_zscore_check/%d-%d.png"%(left_index,right_index))

def consistency_sequence(matches, G1, G2, nodenum_limit=7, cth = 2.0):
	'''
	Obtain a sequence of consistencies between a series of subgraphs extracted from G_1 and G_2. Firstly we sort the 
	matches according to their consistent degree in descending order.  Add one node in the node list recursively in 
	the order of the sorted matches. Then we extract subgraphs based on the node list and calculate the consistency. 
	graphs.
	:param matches: A list of tuple (v_i, u_i), where v_i \in G_1, u_i \in G_2
	:param G1: the matching graph
	:param G2: the matching graph
	:return: A list of consistencies. 
	'''
	cons_edges = consistent_edges(matches, G1, G2)
	mcdeg = {}
	for G1_edge, G2_edge in cons_edges.items():
		mcdeg[(G1_edge[0], G2_edge[0])] = mcdeg.get((G1_edge[0], G2_edge[0]), 0) + 1
		mcdeg[(G1_edge[1], G2_edge[1])] = mcdeg.get((G1_edge[1], G2_edge[1]), 0) + 1
	mcdeg = sorted(mcdeg.items(), key=lambda x:x[1], reverse=True)
	nodes1 = []
	nodes2 = []
	graph_matching_consistent_ratio_list = []
	graph_matching_z_score_list = []
	seeds = []
	for match, cred in mcdeg:
		seeds.append(match)
		nodes1.append(match[0])
		nodes2.append(match[1])
		subG1 = G1.subgraph(nodes1)
		subG2 = G2.subgraph(nodes2)
		gcr = graph_consistency(cons_edges, subG1, subG2)
		graph_matching_consistent_ratio_list.append(gcr)
		k_c = gcr * max(G2.size(), G1.size())
		z = z_score(gcr, subG1, subG2, k_c)
		graph_matching_z_score_list.append(z)
	return graph_matching_consistent_ratio_list, graph_matching_z_score_list,nodes1,nodes2

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


def obtain_seed_with_edges_credibility(matches,G1,G2,real_common_nodes,left_index,right_index,loop_index = 0):

	matching_consistent_ratio_list,graph_matching_z_score_list,nodes_left,nodes_right = consistency_sequence(matches, G1, G2)

	#length = len(matching_consistent_ratio_list)
	length = len(graph_matching_z_score_list)
	eligible_nodes = []
	if length == 0:
		return eligible_nodes,0,0,0
	dic = {}
	for i in range(length):
		dic[i] = graph_matching_z_score_list[i]

	dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
	if dic[0][1] <= 2:
		h = dic[0][0];
	else:
		for i in range(length):
			if dic[i][1] <= 2:
				h = dic[i-1][0]
				break

	eligible_nodes = [(nodes_left[i],nodes_right[i]) for i in range(h+1)] 

	#print "seed list total len: %d" % len(eligible_nodes) 
	if len(eligible_nodes) == 0:
		#print "real seed: 0"
		#print "seed rate: 0" 
		return eligible_nodes,0,0,0
	count = 0
	for node in eligible_nodes:
		if node[0] == node[1]: 
			count += 1
	rate = float(count)/len(eligible_nodes)
	#print "real seed: %d" % count
	#print "seed accuracy rate: %.2f" % rate
	print "consistency degree: "
	mcdeg = match_consistent_degree(matches, G1, G2)
	mcdeg = sorted(mcdeg.items(), key=lambda x:x[1], reverse=True)

	average_degree = 0
	for i in range(len(mcdeg)):
		#temp_count = 0
		#print "%d :" % i,
		#print mcdeg[i],
		average_degree += mcdeg[i][1]
		#temp = nodes_left[:i+1]
		
		#for node in temp:
		#	if node in real_common_nodes:
		#		temp_count += 1
		#temp_rate = float(temp_count)/len(temp)
		#print"\t %.2f" % temp_rate
		#if i == 20:
		#	break

	#plot_edge_conssitencies(matching_consistent_ratio_list)
	print "h = %d,z_score = %.2f" % (h,graph_matching_z_score_list[h])
	plot_edge_conssitencies(graph_matching_z_score_list,left_index,right_index)
	print "average_degree: %.3f" % (average_degree* 1.0 / (i+1))
	return eligible_nodes,count,rate,graph_matching_z_score_list[h]
	


