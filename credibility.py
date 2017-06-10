from deepMatching import *


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
    for match, cred in mcdeg:
        nodes1.append(match[0])
        nodes2.append(match[1])
        subG1 = G1.subgraph(nodes1)
        subG2 = G2.subgraph(nodes2)
        gcr = new_graph_consistency(cons_edges, subG1, subG2)
        graph_matching_consistent_ratio_list.append(gcr)
        # print match, gcr
    return graph_matching_consistent_ratio_list


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


def new_graph_consistency(consistent_edges, G1, G2):
    '''
    Calculate the consistency between two graphs
    :param consistent_edges: A dict, composed of the consistent edges
    :param G1: the matching graph
    :param G2: the matching graph
    :return: the consistency, a float value between 0.0 to 1.0. 
    '''
    G1_size = G1.size()
    G2_size = G2.size()
    N = G1.order()
    if G1_size == 0 or G2_size ==0:
        return 0.0
    consistent_edge_count = 0.0
    for edge in G1.edges():
        if edge in consistent_edges:
            consistent_edge_count += 1.0
    return consistent_edge_count*0.25*N*(N-1)/(G2_size*G1_size)


def main():
    nx_G = nx.erdos_renyi_graph(500, 0.1)
    # nx_G = nx.barabasi_albert_graph(100, 5)
    # nx_G = read_graph('./data/karate.edgelist')
    # nx.draw_spring(nx_G, with_labels=True)
    G1 = sample_graph(nx_G, 0.85)
    G2 = sample_graph(nx_G, 0.85)
    count = 0
    matches = bipartite_matching(G1, G2, dimensions=160)
    for match in matches:
        if match[0] == match[1]:
            count += 1
    print 'Accuracy:', count*1.0/G2.order()
    mcdegs = match_consistent_degree(matches, G1, G2)
    for match, cdeg in sorted(mcdegs.items(), key=lambda x:x[1], reverse=True):
        print match, cdeg, G1.degree(match[0]), G2.degree(match[1])
    edge_consists = consistency_sequence(matches, G1, G2)
    plot_edge_conssitencies(edge_consists)
    matches = maximum_consistency_matches(matches, G1, G2)
    nx.write_edgelist(G1, './data/G1.edgelist', data=False)
    nx.write_edgelist(G2, './data/G2.edgelist', data=False)
    matchf = open('./data/matches.txt', 'w')
    print len(matches)
    for match in matches:
        print match
        matchf.write(str(match[0]) + ',' + str(match[1]) + '\n')
    matchf.close()


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

if __name__ == '__main__':
    main()
