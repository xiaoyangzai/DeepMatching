#!/usr/bin/python
from credibility import *

def match_propagation(matches, G1, G2):
	'''
	Start from the seeds, matching the remaining nodes of the two graph gradually. 
	In each step, the top nodes whose degrees are greater than 2^i, i=\log n, ..., 1 are matched based on the seeds. 
	If the two matching nodes share at least 3 matched neighbors, then the two nodes are also matched. 
	:param matches: The initial seeds
	:param G1: the graph
	:param G2: the graph
	:return: a larger match list
	'''
	Seeds = matches
	deg1 = G1.degree()
	deg2 = G2.degree()
	maxD1 = max([deg for node, deg in deg1.items()])
	maxD2 = max([deg for node, deg in deg2.items()])
	maxD = max(maxD1, maxD2)
	for i in range(1, int(np.log2(maxD)))[::-1]:
		limited_deg = 2**i
		newmatches = propagation_phase(Seeds, G1, G2, limited_deg)
		# print limited_deg, len(newmatches), len(Seeds)
		Seeds = newmatches
	count = 0
	for item in Seeds:
		if item[0] == item[1]:
			count += 1
	if len(Seeds) == 0:
		rate = 0.0	
	else:
		rate = count *1.0 / len(Seeds)
	#print "refine rate: %.2f" % rate

	return Seeds,count,rate


def propagation_phase(Seeds, G1, G2, limitdeg):
	'''
	Match the nodes whose degrees are greater than limitdeg. If the two matching nodes share at least 3 matched 
	neighbors, then the two nodes are also matched. 
	:param Seeds: The seeds
	:param G1: the graph
	:param G2: the graph
	:param limitdeg: the limited degree
	:return: A list of newly matched nodes, some matched nodes may also be included.
	'''
	# G1matched = set([s1 for s1, s2 in Seeds])
	# G2matched = set([s2 for s1, s2 in Seeds])
	WN = {}
	for seed in Seeds:
		N1 = G1.neighbors(seed[0])
		N2 = G2.neighbors(seed[1])
		for n1 in N1:
			if G1.degree(n1)<limitdeg:#n1 in G1matched or
				continue
			for n2 in N2:
				if G2.degree(n2) < limitdeg: # n2 in G2matched or
					continue
				if n1 not in WN:
					WN[n1] = {}
					WN[n1][n2] = 1
				else:
					WN[n1][n2] = WN[n1].get(n2, 0) + 1
	newmatches = []
	for key, cand in WN.items():
		maxWN = max([(n2, wn) for n2, wn in cand.items()], key=lambda x:x[1])
		if maxWN[1] > 2:
			newmatches.append((key, maxWN[0], maxWN[1]))
	dupnodes = {}
	for match in newmatches:
		# dupnodes.get(match[1], []).append(match)
		if match[1] not in dupnodes:
			dupnodes[match[1]] = [match]
		else:
			dupnodes[match[1]].append(match)
	newmatches = []
	for dupnode, matches in dupnodes.items():
		newmatches.append(max(matches, key=lambda x:x[2]))
	return [(n1, n2) for n1, n2, wits in newmatches]


def read_matches(filename):
	'''
	Read matches from a file
	:param filename: the filename of the file storing the matches
	:return: a list of matches
	'''
	matches = []
	for line in open(filename, 'r'):
		line = line.strip()
		nodes = line.split(',')
		matches.append((string.atoi(nodes[0]), string.atoi(nodes[1])))
	return matches


#def main():
#	nx_G = nx.erdos_renyi_graph(500, 0.1)
#	G1 = sample_graph(nx_G, 0.85)
#	G2 = sample_graph(nx_G, 0.85)
#	# G1 = read_graph('./data/G1.edgelist')
#	# G2 = read_graph('./data/G2.edgelist')
#	# Seeds = read_matches('./data/matches.txt')
#	matches = bipartite_matching(G1, G2, dimensions=160)
#	count = 0
#	for match in matches:
#		if match[0] == match[1]:
#			count += 1
#	print 'Accuracy:', count * 1.0 / len(matches)
#	# mcdegs = match_consistent_degree(matches, G1, G2)
#	# for match, cdeg in sorted(mcdegs.items(), key=lambda x: x[1], reverse=True):
#	#	 print match, cdeg, G1.degree(match[0]), G2.degree(match[1])
#	# Seeds = maximum_consistency_matches(matches, G1, G2)
#	refined_matches = match_propagation(matches, G1, G2)
#	count = 0
#	for match in refined_matches:
#		if match[0] == match[1]:
#			count += 1
#	rate = count * 0.1 / len(refined_matches)
#	print "refine rate = %.2f" % rate
#	edge_consists = consistency_sequence(refined_matches, G1, G2)
#	plot_edge_conssitencies(edge_consists)
#
#
#if __name__ == '__main__':
#	main()
