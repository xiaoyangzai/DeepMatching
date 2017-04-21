import networkx as nx
import random
import numpy as np


class GraphMatcher():
    def __init__(self, graph1, graph2, seeds, s1=0.5, s2=0.5,
                 underlying_graph=None, p=0.5):
        self.G1 = graph1
        self.G2 = graph2
        self.seeds = seeds
        self.s1 = s1
        self.s2 = s2
        self.G = underlying_graph
        self.p = p
        self.matched = {}
        for seed in seeds:
            self.matched[seed] = 1.0

    def calculate_credits(self, node_pair):
        """
        Calculate the credits of a node pair
        :param node_pair: a tuple (node1, node2)
        :return: the credits of the node pair
        """
        matched = self.matched
        G1 = self.G1
        G2 = self.G2
        node1 = node_pair[0]
        node2 = node_pair[1]

        N1 = set(G1.neighbors(node1))
        N2 = set(G2.neighbors(node2))
        # print N1, N2
        cred = 0.0
        sup_matched = list()
        for n1, n2 in matched.keys():
            # find all the matched node pairs connected to the node pair
            if n2 in N2 or n1 in N1: #
                sup_matched.append((n1, n2))
        # print node1, node2, sup_matched

        for n1, n2 in sup_matched:
            # calculate the credits
            p1 = 1.0 if G1.has_edge(n1, node1) else self.p * (1-self.s1)
            p2 = 1.0 if G2.has_edge(n2, node2) else self.p * (1-self.s2)
            cred += matched.get((n1, n2), 0.0) * p1 * p2

        return cred

    def get_candidate_matching_pairs(self):
        """
        get candidate matching node pairs based on current matched pairs
        :return:
        """
        matched = self.matched
        G1 = self.G1
        G2 = self.G2
        M1 = set([n1 for n1, n2 in matched.keys()])
        M2 = set([n2 for n1, n2 in matched.keys()])

        candidates = set()
        for n1, n2 in matched.keys():
            if G1.has_node(n1) and G2.has_node(n2):
                N1 = G1.neighbors(n1)
                N2 = G2.neighbors(n2)
            else: continue
            for node1 in N1:
                if node1 in M1: continue    # node 1 is already matched
                for node2 in N2:
                    if node2 in M2: continue    # node 2 is already matched
                    candidates.add((node1, node2))
        return candidates


def read_graph(input, weighted=False, directed=False):
    """weighted
    Reads the input network in networkx.
    :param input: the edge list file of the graph
    :param weighted:
    :param directed:
    :return: networkx graph
    """
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight',float),),
                             create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(input, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    if not directed:
        G = G.to_undirected()
    return G


def sample_graph(G, s):
    newG = nx.Graph()
    newG.add_edges_from(random.sample(G.edges(), int(len(G.edges())*s)))
    for edge in newG.edges():
        newG[edge[0]][edge[1]]['weight'] = 1.0
    return newG


def main():
    G = read_graph('./data/karate.edgelist') # read_graph('./data/G.edgelist')
    print nx.density(G), G.order(), G.size()
    print G.degree()
    G1 = sample_graph(G, 0.8) # read_graph('./data/G1.edgelist') #
    print G1.order(), G1.size()
    G2 = sample_graph(G, 0.8) # read_graph('./data/G2.edgelist')
    print G2.order(), G2.size()
    gm = GraphMatcher(G1, G2, seeds=[(1,1), (2, 2), (3, 3), (4, 4), (5, 5),
                                     (6, 6), (7, 7), (8, 8), (11, 11),
                                     (14, 14), (17, 17), (9, 9), (10, 10),
                                     (34, 34)],
                      s1=0.2,
                      s2=0.2, underlying_graph=G, p=0.75)
    candidates = gm.get_candidate_matching_pairs()
    candidate_credits = []
    for node_pair in candidates:
        credit = gm.calculate_credits(node_pair)
        candidate_credits.append((node_pair[0], node_pair[1], credit))
        print node_pair, credit

    sum_credit1 = {}
    sum_credit2 = {}
    for cand_credit in candidate_credits:
        sum_credit1[cand_credit[0]] = sum_credit1.get(cand_credit[0], 0.0) + \
                                      cand_credit[2]
        sum_credit2[cand_credit[1]] = sum_credit2.get(cand_credit[1], 0.0) + \
                                      cand_credit[2]
    candidate_prob = []
    for cand_credit in candidate_credits:
        cand_prob = cand_credit[2] * 1.0 / np.sqrt(sum_credit1[
                                                            cand_credit[0]] *
                                                        sum_credit2[
                                                            cand_credit[1]])
        candidate_prob.append((cand_credit[0], cand_credit[1], cand_credit[
            2], cand_prob))

    sorted_prob = sorted(candidate_prob, key=lambda x:x[2])
    for prob in sorted_prob:
        print prob


if __name__=="__main__":
    main()