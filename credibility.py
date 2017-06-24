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


def maximum_consistency_matches(matches, G1, G2, nodenum_limit=20, cth = 1.0):
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
    cseq = consistency_sequence(matches, G1, G2)
    for i in range(nodenum_limit):
        cseq[i] = 0
    mcdeg = match_consistent_degree(matches, G1, G2)
    mcdeg = sorted(mcdeg.items(), key=lambda x: x[1], reverse=True)
    maxindex = cseq.index(max(cseq))
    maxcons = cseq[maxindex]
    print maxindex, maxcons
    credible_matches = []
    if maxindex > nodenum_limit -1 and maxcons > cth:
        credible_matches = [match for match, cdeg in mcdeg[0:maxindex+1]]
    return credible_matches
def z_score(gcr,subG1,subG2):

	subg1_m = nx.number_of_edges(subG1)
	subg2_m = nx.number_of_edges(subG2)

	subg1_n = nx.number_of_nodes(subG1)
	subg2_n = nx.number_of_nodes(subG2)
	if subg1_n <= 1 or subg2_n <= 1 or subg1_m == 0 or subg2_m == 0:
		return 0.0

	density_subg1 = 2*subg1_m * 1.0 / (subg1_n *(subg1_n - 1)) 
	density_subg2 = 2*subg2_m * 1.0 / (subg2_n *(subg2_n - 1)) 

	density = density_subg1 if density_subg1 <= density_subg2 else density_subg2
	if density == 1:
		return 0.0
	z = ((gcr - density)*1.0) / (density * (1 - density)) 
	return z

def consistency_sequence(matches, G1, G2):
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
    for match, cred in mcdeg:
		nodes1.append(match[0])
		nodes2.append(match[1])
		subG1 = G1.subgraph(nodes1)
		subG2 = G2.subgraph(nodes2)
		gcr = graph_consistency(cons_edges, subG1, subG2)
		z = z_score(gcr,subG1,subG2)
		graph_matching_consistent_ratio_list.append(gcr)
		graph_matching_z_score_list.append(z)
        # print match, gcr
    return graph_matching_consistent_ratio_list,graph_matching_z_score_list,nodes1,nodes2


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
        if edge in consistent_edges:
            consistent_edge_count += 1.0
    return consistent_edge_count*1.0/max(G2_size, G1_size)


def plot_edge_conssitencies(edge_consists):
    '''
    Used to plot the consistency sequence
    :param edge_consists: 
    :return: 
    '''
    fig, ax = plt.subplots()
    ax.plot(edge_consists, '.-')
    ax.set_xlabel('Match index (ordered by consistent degree)')
    ax.set_ylabel('Consistent edge ratio')
    plt.tight_layout()
    plt.show()



def obtain_seed_with_edges_credibility(matches,G1,G2,real_common_nodes):

	matching_consistent_ratio_list,graph_matching_z_score_list,nodes_left,nodes_right = consistency_sequence(matches, G1, G2)

	#length = len(matching_consistent_ratio_list)
	length = len(graph_matching_z_score_list)
	dic = {}
	for i in range(length):
		dic[i] = graph_matching_z_score_list[i]

	dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)

	if dic[0][1] <= 2:
		h = dic[0][0];
	else:
		for i in range(length):
			if dic[i][1] <= 2:
				h = dic[i][0]
				break
	


	#while True:
	#	if matching_consistent_ratio_list[h] >= 1:
	#		matching_consistent_ratio_list[h] = 0
	#		h = matching_consistent_ratio_list.index(max(matching_consistent_ratio_list))
	#	else:
	#		break

	eligible_nodes = [(nodes_left[i],nodes_right[i]) for i in range(h+1)] 

	#print "seed list total len: %d" % len(eligible_nodes) 
	if len(eligible_nodes) == 0:
		#print "real seed: 0"
		#print "seed rate: 0" 
		return eligible_nodes,0,0
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
		temp_count = 0
		print "%d :" % i,
		print mcdeg[i],
		average_degree += mcdeg[i][1]
		temp = nodes_left[:i+1]
		
		for node in temp:
			if node in real_common_nodes:
				temp_count += 1
		temp_rate = float(temp_count)/len(temp)
		print"\t %.2f" % temp_rate
		#if i == 20:
		#	break

	#plot_edge_conssitencies(matching_consistent_ratio_list)
	print "h = %d,match_ratio = %.2f" % (h,matching_consistent_ratio_list[h])
	plot_edge_conssitencies(graph_matching_z_score_list)
	print "\t average_degree: %.3f" % (average_degree* 1.0 / (i+1))
	return eligible_nodes,count,rate
	


