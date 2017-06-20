#!/usr/bin/python
import numpy as np
import community as cmty
import os
import time as ti
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from graph_matching import sample_graph
import sys
import cnm
from sklearn.metrics.pairwise import *
import snap
import deepMatching as dm
from itertools import izip
from credibility import *

def transform_snap_to_networkx(SG):
	'''
	Transforming the graph of snap specified by the @SG to the graph of networkx
	Return the graph of networkx as the result of this function
	'''
	G_networkx = nx.Graph()
	#add the nodes from the SG
	for node in SG.Nodes():
		G_networkx.add_node(node.GetId())
	#add edges to the G_networkx
	for i in G_networkx.nodes():
		for j in G_networkx.nodes():
			if SG.IsEdge(i,j):
				G_networkx.add_edge(i,j)
	return G_networkx	
def transform_networkx_to_snap(G):
	'''
	Transforming the graph of networkx specified by the @G to the graph of snap 
	Return the graph of snap as the result of this function
	'''
	TG = snap.TUNGraph.New()
	print "start transform the graph format from the networkx to snap ......"
	start_time = time.time()
	#add the nodes from the G
	print len(G.nodes())
	for node in G.nodes():
		TG.AddNode(node)
	#add edges to the TG
	for edge in G.edges():
		TG.AddEdge(edge[0],edge[1])

	print "transform finished,the Graph nodes: %d\t edges: %d" % (TG.GetNodes(),TG.GetEdges())
	running_time = time.time() - start_time
	print "sample finished,running time :%d mins %d secs" % (int(running_time / 60),int(running_time % 60))
	return TG
	

def bi_sample_graph(nx_G,sample_rate = 0.8):
	G1 = sample_graph(nx_G, sample_rate)
	G2 = sample_graph(nx_G, sample_rate)
	return G1,G2
	
def community_detect_graph(G1,G2,detect_method = None):
	'''
	community detection and create graph with community as the node 
	'''
	print "start sample ......"

	start_time = time.time()
	#transforms SG to networkx
	if detect_method == None :
		#detect with the cnm
		SG1 = transform_networkx_to_snap(G1)
		SG2 = transform_networkx_to_snap(G2)
		SG1_ret_list_cnm = cnm.community_cnm(SG1)
		SG2_ret_list_cnm = cnm.community_cnm(SG2)
		SG1 = G1
		SG2 = G2
		SG1_ret_list_spectral = graph_spectral.partition_super_cliques(SG1)
		SG2_ret_list_spectral = graph_spectral.partition_super_cliques(SG2)
		running_time = time.time() - start_time
		print "sample finished,running time :%d mins %d secs" % (int(running_time / 60),int(running_time % 60))
		return SG1_ret_list_cnm,SG2_ret_list_cnm,SG1_ret_list_spectral,SG2_ret_list_spectral
	if detect_method == cnm.community_cnm_with_limit:
		SG1 = transform_networkx_to_snap(G1)
		SG2 = transform_networkx_to_snap(G2)
		print "limit nodes of the community for G1 : %d" % (len(G1.nodes()) / 8)
		SG1_ret_list = detect_method(SG1,len(G1.nodes())/8)
		print "limit nodes of the community for G2 : %d" % (len(G2.nodes()) / 8)
		SG2_ret_list = detect_method(SG2,len(G2.nodes()) /8)
		print "SG1 community size: %d \t SG2 community size :%d " % (len(SG1_ret_list),len(SG2_ret_list))
		running_time = time.time() - start_time
		print "sample finished,running time :%d mins %d secs" % (int(running_time / 60),int(running_time % 60))
		return SG1,SG2,SG1_ret_list,SG2_ret_list
	if detect_method == cmty.best_partition:
		s1_partition = cmty.best_partition(G1)
		s2_partition = cmty.best_partition(G2)
		SG1_ret_list = []
		SG2_ret_list = []
		for com in set(s1_partition.values()):
			temp = [nodes for nodes in s1_partition.keys() if s1_partition[nodes] == com]
			SG1_ret_list.append(temp)
		for com in set(s2_partition.values()):
			temp = [nodes for nodes in s2_partition.keys() if s2_partition[nodes] == com]
			SG2_ret_list.append(temp)
		running_time = time.time() - start_time
		print "SG1 community size: %d \t SG2 community size :%d " % (len(SG1_ret_list),len(SG2_ret_list))
		print "sample finished,running time :%d mins %d secs" % (int(running_time / 60),int(running_time % 60))
		SG1 = transform_networkx_to_snap(G1)
		SG2 = transform_networkx_to_snap(G2)
		return SG1,SG2,SG1_ret_list,SG2_ret_list


	if detect_method == cnm.community_cnm:
		SG1 = transform_networkx_to_snap(G1)
		SG2 = transform_networkx_to_snap(G2)
	else:
		SG1 = G1
		SG2 = G2
	SG1_ret_list = detect_method(SG1)
	SG2_ret_list = detect_method(SG2)

	running_time = time.time() - start_time
	print "SG1 community size: %d \t SG2 community size :%d " % (len(SG1_ret_list),len(SG2_ret_list))
	print "sample finished,running time :%d mins %d secs" % (int(running_time / 60),int(running_time % 60))
	
	return SG1,SG2,SG1_ret_list,SG2_ret_list
	 
	
def transform_gml_to_networkx(gml_filename):
	sf = open(gml_filename)
	G = nx.Graph()
	
	line = sf.readline()
	while "edge" not in line:
		line = sf.readline()

	while line != '':
		line = sf.readline()
		if line == '':
			break
		line = sf.readline()
		if line == '':
			break
		line = line.split(' ')

		source = int(line[-1][:-1]) + 1
		line = sf.readline()
		if line == '':
			break
		line = line.split(' ')
		dest = int(line[-1][:-1]) + 1
		G.add_edge(source,dest)
		line = sf.readline()
		if line == '':
			break
		line = sf.readline()

	sf.close()
	print "load finished,the Graph nodes: %d\t edges: %d" % (len(G.nodes()),len(G.edges()))
	return G


def load_graph_from_file(filename,comments = '#',delimiter = ' '):
	print "start load graph from file: %s" % filename
	if ".gml" in filename:
		return transform_gml_to_networkx(filename)
	G = nx.Graph()
	start_time = time.time()
	sf = open(filename)
	line = sf.readline()
	while line != '':
		if comments not in line:
			break
		line = sf.readline()
	while line != '':
		line = line.split(delimiter)
		# plus 1 reprents that the index of nodes is start from 1 at least(For sanp graph ,the nodes' index must start from 1 at least)
		G.add_edge(int(line[0]) + 1,int(line[1][:-1]) + 1)
		#G.add_edge(int(line[0]),int(line[1][:-1]))
		line = sf.readline()
	running_time = time.time() - start_time
	#print "G nodes: %d \t edges: %d" % (len(G.nodes()),len(G.edges()))
	#print "running time :%d mins %d secs" % (int(running_time / 60),int(running_time % 60))
	print "load graph finish!! "
	sf.close()
	return G
		


def create_cmty_graph(nx_G,SG1_ret_list,SG2_ret_list):
	'''create graph of which the communities of the list as the nodes and the count of the connection between the nodes from the different community acts as the weight value"
	'''
	print "start create graph with the community geted from privous step......"
	nx_SG1 = nx.Graph()
	nx_SG2 = nx.Graph()
	#The graph adds the nodes' index equeled to the index community
	for size in range(0,len(SG1_ret_list)):
		nx_SG1.add_node(size)
	for size in range(0,len(SG2_ret_list)):
		nx_SG2.add_node(size)
	#count the weight value to the edge
	for i in range(0,len(SG1_ret_list) - 1):
		for cmty in SG1_ret_list[(i + 1):]:
			count = 0
			#count the total number connections between the community
			for node_i in SG1_ret_list[i]:
				for node_j in cmty:
					#print "judge that whether exist an edge between nodes:%d to %d" % (node_i,node_j)
					if nx_G.has_edge(node_i,node_j):
						count = count + 1
			if count == 0:
				continue
			#print "compare %d - %d : %d" % (i,SG1_ret_list.index(cmty),count)
			nx_SG1.add_edge(i,SG1_ret_list.index(cmty),weight=count)
			#nx_SG1.add_edge(i,SG1_ret_list.index(cmty))

	for i in range(0,len(SG2_ret_list) - 1):
		for cmty in SG2_ret_list[(i + 1):]:
			count = 0
			#count the total number connections between the community
			for node_i in SG2_ret_list[i]:
				for node_j in cmty:
					#print "judge that whether exist an edge between nodes:%d to %d" % (node_i,node_j)
					if nx_G.has_edge(node_i,node_j):
						count = count + 1
			if count == 0:
				continue
			#print "compare %d - %d : %d" % (i,SG2_ret_list.index(cmty),count)
			nx_SG2.add_edge(i,SG2_ret_list.index(cmty),weight = count)
			#nx_SG2.add_edge(i,SG2_ret_list.index(cmty))
	
	return nx_SG1,nx_SG2

def draw_networkx(G):
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G,pos);
	nx.draw_networkx(G,pos)
	plt.show()
	return


def draw_networkx_with_weight(G):
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G,pos);
	nx.draw_networkx_edges(G,pos);
	nx.draw_networkx_labels(G,pos);
	nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'weight'))
	plt.show()
	return
def draw_partitions(G, partitions):
    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    color_index = 0
    partition_color = {}
    for partition in partitions:
        for node in partition:
            partition_color[node] = color_map[color_index]
        color_index += 1
        color_index = color_index % len(color_map)
    node_color = [partition_color.get(node, 'o') for node in G.nodes()]
    nx.draw_spring(G, node_color=node_color, with_labels=True)
    plt.show()

def obtain_degree_inter_cmty(G,nodes_list):
	print "obtain inter degree of the cmty"
	degrees_of_nodes = [] 
	edges_list = []
	for node in nodes_list:
		temp = nx.neighbors(G,node)
		#internal degree of the node in community
		neighbors_in_cmpty = [i for i in temp if i in nodes_list]
		degrees_of_nodes.append([node,len(neighbors_in_cmpty)])
		for j in neighbors_in_cmpty:
			edges_list.append([node,j])
		#degrees_of_nodes.append(len(temp))
	
	print "obtain inter degree of the cmty...ok"
	return degrees_of_nodes,edges_list

def obtain_degree_extern_cmty(G,nodes_list):
	print "obtain extern degree of cmty"
	degrees_of_nodes = 0
	for node in nodes_list:
		temp = nx.neighbors(G,node)
		for i in temp:
			if i not in nodes_list:
				degrees_of_nodes += 1
	print "obtain extern degree of cmty....ok"
	return degrees_of_nodes


def obtain_clustering_coefficient(G,nodes_list):
	'''
	calculate the clustering coefficient of node in nodes list
	Return between centrality list
	Return type: sorted list
	'''
	dic_cc = nx.clustering(G,nodes_list)
	cc = []
	for k in dic_cc:
		cc.append(dic_cc[k])
	cc = sorted(cc)
	return cc

def obtain_between_centrality(G,edges_list):
	'''
	calculate betweenness centrality 
	'''
	#1. creat graph with nodes list
	g = nx.Graph()
	for item in edges_list:
		g.add_edge(item[0],item[1])
	bc = nx.betweenness_centrality(g)
	bc_list = []
	for key in bc:
		bc_list.append(bc[key])
	return bc_list

def obtain_midian_list(value_list):
	value_list = sorted(value_list)	
	half = len(value_list) // 2
	return float(value_list[half] + value_list[~half]) / 2
def obtain_triangles_count(G,nodes_list):
	return nx.triangles(G,nodes_list)

def obtain_feature_of_cmty(G,SG,nodes_list,throd):
	'''
	obtain the feature of the community
	Return: the community feature
	Return type: list which consists of:
		1. outdegree of the community
		2. number of nodes
		3. number of edges
		4. maxmiun degree 1th,2th and 3th
		5. average degree of community
		6. midian degree
		7. density of community
		8. triangles number of the maximun degree
		9. modularity

		10. maxmiun bs contained 1th,2th and 3th
		11. average bs
		12. midian bs
		13. average cc
		14. midian cc.
	'''
	feature = []	
	#obtain the outdegree of the community
	outdegree = obtain_degree_extern_cmty(G,nodes_list)
	feature.append(outdegree)

	#1.calculate number of nodes in community
	feature.append(len(nodes_list))

	#degree of the nodes
	degree_nodes,edges = obtain_degree_inter_cmty(G,nodes_list)

	#2. count the number of edges in community
	degree_list = [item[1] for item in degree_nodes]
	edges_count = sum(degree_list) / 2
	feature.append(edges_count)

	#3.obtain the max three maxmiun value of degree
	degree_nodes = sorted(degree_nodes,key=lambda x:x[1],reverse = True)
	for i in range(int(throd * 0.75)):
		feature.append(degree_nodes[i][1])

	#average and midian degree
	average_degree = float(sum(degree_list))/len(degree_list)	
	midian_degree = obtain_midian_list(degree_list)
	feature.append(average_degree)
	feature.append(midian_degree)

	#density of the community
	d = float(2 * edges_count) / (len(nodes_list) *(len(nodes_list) - 1))
	feature.append(d)
	
	max_degree_nodes_list = []

	for i in range(int(throd * 0.75)):
		max_degree_nodes_list.append(degree_nodes[i][0])
	triangles_count = obtain_triangles_count(G,max_degree_nodes_list)
	for k in triangles_count:
		feature.append(triangles_count[k])

	#4.calculate betweenness centrality 
	between_centrality_list = obtain_between_centrality(G,edges)
	midian_bs = obtain_midian_list(between_centrality_list)

	#max bs 
	for i in range(int(throd * 0.75)):
		feature.append(between_centrality_list[i])
	average_bs = float(sum(between_centrality_list)) / len(between_centrality_list)
	feature.append(average_bs)
	feature.append(midian_bs)

	#modularity
	Nodes = snap.TIntV()
	for nodeId in nodes_list:
		    Nodes.Add(nodeId)
	modularity = snap.GetModularity(SG,Nodes) 
	feature.append(modularity*1000)

	#5 calculate clustering coefficients
	cc = obtain_clustering_coefficient(G,nodes_list)
	average_cc = float(sum(cc)) / len(cc)
	midian_cc = obtain_midian_list(cc)
	feature.append(average_cc)
	feature.append(midian_cc)
	return feature

def obtain_edges_between_cmty(edges_list,s_nodes,d_nodes):
	edges_number = 0
	for i in s_nodes:
		for j in d_nodes:
			item = (i,j)
			item_1 = (j,i)
			if item in edges_list or item_1 in edges_list:
				edges_number += 1
	return edges_number
	

def merge_small_community(G,rest_small_cmty,new_cmty_list):

	print "merge small community....."
	edges_list = G.edges()
	discard_count = 0
	merge_order = []
	for i in range(len(rest_small_cmty)):
		distance = []
		for j in range(len(new_cmty_list)):
			temp =  obtain_edges_between_cmty(edges_list,rest_small_cmty[i],new_cmty_list[j])
			#print "%i <--> %i : %d" % (i,j,temp)
			distance.append([j,temp])
		distance = sorted(distance,key=lambda x:x[1],reverse=True)
		print distance
		if distance[0][1] == 0:
			print rest_small_cmty[i]
			discard_count += 1
			continue
		#join the i into j
		merge_order.append([i,distance[0][0]])
	
	print "merge list: ",
	print merge_order
	for item in merge_order:
		for node in rest_small_cmty[item[0]]:
			new_cmty_list[item[1]].append(node)
	print "after merging,the number of the new communities: %d" % len(new_cmty_list)	
	print "discard %d small communitis" % discard_count
	print "merge small community....ok!!"
	return new_cmty_list

def obtain_cmty_feature_array(G,SG,cmty_list,throd_value):
	print "obtain cmty feature array"
	feature = [] 
	eligible_cmty_list = []
	rest_small_cmty = []
	new_cmty_list = []
	for cmty in cmty_list:
		if len(cmty) <= throd_value:
				rest_small_cmty.append(cmty)
		else:
			new_cmty_list.append(cmty)
	
	#neglect the small size community
	eligible_cmty_list = new_cmty_list

	
	##join the small community into community with which it is most connected
	#if len(rest_small_cmty) > 0:
	#	eligible_cmty_list = merge_small_community(G,rest_small_cmty,new_cmty_list)
	#else:
	#	eligible_cmty_list = new_cmty_list
	#print "hanld the cmty with throd..ok"
	loop_index = 0
	for cmty in eligible_cmty_list:
		print "cmty: %d" % loop_index
		print "length: %d" % len(cmty) 
		loop_index += 1
		temp = obtain_feature_of_cmty(G,SG,cmty,throd_value)
		feature.append(temp)
		sys.stdout.flush()
	print "obtain cmty feature array finished"
	return eligible_cmty_list,feature

def euclidean_metric(sg1_feature_list,sg2_feature_list):
	X = [] 
	Y = []
	X.append(sg1_feature_list)
	Y.append(sg2_feature_list)
	distance = euclidean_distances(X,Y) 
	return distance[0][0]
	
def euclidean_distance(sg1_feature_list,sg2_feature_list):
	count = len(sg1_feature_list)
	x = sg1_feature_list
	y = sg2_feature_list
	temp = 0.0
	for i in range(count):
		temp += (float(x[i] - y[i]))**2
	distance = temp ** 0.5	
	return distance
	
		

def obtain_score_between_features(sg1_feature_list,sg2_feature_list,method = euclidean_metric):
	'''
	calculate the similarity score between features by algritom specified method
	'''
	big_feature_list = sg1_feature_list if len(sg1_feature_list) >= len(sg2_feature_list) else sg2_feature_list
	score_list = [[] for i in range(len(big_feature_list))]
	small_feature_list = sg1_feature_list if len(sg1_feature_list) < len(sg2_feature_list) else sg2_feature_list
	for index in range(0,len(big_feature_list)):
		s_feature = big_feature_list[index]
		for d_feature in small_feature_list:
			score = method(s_feature,d_feature)
			score_list[index].append(score)
	return score_list

def calculate_common_nodes_between_cmties(s_nodes_list,d_nodes_list):
	'''
	calculate the number of the nodes existed in both s_nodes_list and d_nodes_list
	'''
	if len(s_nodes_list) == 0 or len(d_nodes_list) == 0:
		return 0
	small_node_list = s_nodes_list if len(s_nodes_list) <= len(d_nodes_list) else d_nodes_list
	big_node_list = s_nodes_list if len(s_nodes_list) > len(d_nodes_list) else d_nodes_list
	common_nodes_list = []
	total_count = len(small_node_list) 
	for node in small_node_list:
		if node in big_node_list:
			common_nodes_list.append(node)
	
	print "source cmty: ",
	print s_nodes_list[:8]
	print "dest cmty: ",
	print d_nodes_list[:8]
	common_count = len(common_nodes_list)
	print "common nodes count: %d" % common_count
	print "small length: %d" % len(small_node_list)
	return float(common_count)/total_count,common_nodes_list

def repeated_eavalute_accuracy_by_feature(G1,G2,throd_value = 0.75,limit_cmty_nodes = 10,method = euclidean_metric,detect_method = cmty.best_partition):
	#print "sample rate: %.4f" % sample_rate
	#print "limit cmty nodes: %d" % limit_cmty_nodes


	# The type of SG1 and SG2 is the type Snap needs 
	# SG1_ret_list and SG2_ret_list is the result of the detection of graph
	SG1,SG2,SG1_ret_list,SG2_ret_list = community_detect_graph(G1,G2,detect_method = detect_method)

	# The SG1_rest_small_cmty, same as SG2_rest_small_cmty, holds the communities in which the nodes is lower than the throd. The SG1_new_cmty_list just includes the eligible communities.
	SG1_eligible_cmty_list,SG1_features_list = obtain_cmty_feature_array(G1,SG1,SG1_ret_list,limit_cmty_nodes)
	SG2_eligible_cmty_list,SG2_features_list = obtain_cmty_feature_array(G2,SG2,SG2_ret_list,limit_cmty_nodes)
	score_list = obtain_score_between_features(SG1_features_list,SG2_features_list,method = method)

	rate,left_Graph,right_Graph,left_cmty_list,right_cmty_list,matched_index = calculate_accuracy_rate_by_feature(G1,SG1_eligible_cmty_list,G2,SG2_eligible_cmty_list,score_list,SG1_features_list,SG2_features_list)

	return rate,left_Graph,right_Graph,left_cmty_list,right_cmty_list,matched_index


def calculate_accuracy_rate_by_feature(SG1,SG1_new_cmty,SG2,SG2_new_cmty,score_list,SG1_feature,SG2_feature,throd_value = 0.75):
	'''	
	calculate the accuracy 
	return the result of the matched communities index
	'''
	matched_count = 0
	unmatched_count = 0
	matched_index = []

	big_new_cmty,big_G = (SG1_new_cmty,SG1) if len(SG1_new_cmty) >= len(SG2_new_cmty) else (SG2_new_cmty,SG2)
	small_new_cmty,small_G = (SG1_new_cmty,SG1) if len(SG1_new_cmty) < len(SG2_new_cmty) else (SG2_new_cmty,SG2)
	big_feature = SG1_feature if len(SG1_feature) >= len(SG2_feature) else SG2_feature
	small_feature = SG1_feature if len(SG1_feature) < len(SG2_feature) else SG2_feature

	loop_count = len(big_new_cmty)	
	small_cmty_count = len(small_new_cmty)
	for i in range(0,loop_count):
		print "**************************************************************"
		#obtain the similarity list of ith community with all of the other community
		similarity_list =[(index,score_list[i][index]) for index in range(0,small_cmty_count)]
		#sort the similiarity of ith community such that obtain the most similar one
		similarity_list = sorted(similarity_list,key=lambda x:x[1])

		print "cmty: %d" % i
		print "similarity list: ",
		print similarity_list
		first_flag =True 

		while len(similarity_list) > 0:
			#obtain the similiarity list of the best one similiaried with the ith cmty from big community list
			best_score = similarity_list[0][1]
			C_index = similarity_list[0][0]
			print "best score: %.5f" % best_score
			print "C index: %d"% C_index

			#stored the index of the  first matched community
			if first_flag:
				first_flag = False
				first_matched_index = C_index

			similarity_list.pop(0)
			#obtain the community which is most similar to the community specified by C_index
			dest_similarity_list = []	
			for item in score_list:
				dest_similarity_list.append(item[C_index])
			dest_similarity_list = sorted(dest_similarity_list)
			#print dest_similarity_list
			if best_score > dest_similarity_list[0]:
				continue;
			break
		#no matched if length of the similiarity list is zero,guasee the firsted matched community is the best one
		if len(similarity_list) == 0:
			print "guasee matched: %d  ==>  %d" %(i,first_matched_index)
			print "socre: %.5f" % score_list[i][first_matched_index]
			print "cmty: %d" % i
			print big_new_cmty[i][:10]
			print big_feature[i]
			print "cmty: %d" % first_matched_index 
			print small_new_cmty[first_matched_index][:10]
			print small_feature[first_matched_index]
			#calculate the common node between ith community of SG1 and first matched community of SG2
			temp_rate,common_nodes_list = calculate_common_nodes_between_cmties(big_new_cmty[i],small_new_cmty[first_matched_index]) 
			matched_index.append([i,first_matched_index,len(common_nodes_list)])
			if temp_rate >= throd_value:
				matched_count += 1
				print "matched count: %d" % matched_count
				print "mapping successful!"
			else:
				print "mapping failed"
				unmatched_count += 1
				print "unmatched count: %d" % unmatched_count
			continue
		print "best candidate: %d" % C_index
		temp_rate,common_nodes_list = calculate_common_nodes_between_cmties(big_new_cmty[i],small_new_cmty[C_index]) 
		matched_index.append([i,C_index,len(common_nodes_list)])
		print "rate: %.4f" % temp_rate
		if temp_rate >= throd_value:
			print "mapping successful!"
			matched_count += 1
		else:
			print "mapping failed"
			unmatched_count += 1
		print "matched count: %d" % matched_count
		print "unmatched count: %d" % unmatched_count
	
	accuracy_rate = float(matched_count)/loop_count
	print "total count: %d" % loop_count
	print "accuracy rate: %.5f" % accuracy_rate
	return accuracy_rate,big_G,small_G,big_new_cmty,small_new_cmty,matched_index

def obtain_accuracy_rate_in_matched_cmty(left_graph,left_cmty_list,right_graph,right_cmty_list,matched_cmty_index_pairs,dimensions=65):
	'''
	the @left_cmty_list obtained by the best_partition to @left_graph consists of nodes of the communities
	matched_cmty_index_pairs contains the pairs of index of the communities from the @left_cmty_list and @right_cmty_list and the number of the nodes in both communities,just like [[1,2,300],...] which means the 1th community of the @left_graph matched with the 2th of the @right_graph. they have 300 nodes in both 1th and 2th communities
	the dimensions of the deepwalk is specified by @dimensions 
	Return:
		@matched_nodes_number: list, contains the number of total nodes in the  matched communities and the number of same nodes bewteen matched communities, and the number of total nodes obtained by deepwalk and the number of the same nodes, and the number of the total seed nodes obtained by edge_consistency_method and the number of the real seed nodes
		@seed_rate: list,the ratio of the number of real seed nodes to the number of the total seed nodes of the each matched communities
		@seed_nodes_list: list, consisted of the total seed nodes index of each matched communities obtained by edge_consistency_method
	'''
	matched_nodes_number = []
	seed_rate = []
	seed_nodes_list = []
	for item in matched_cmty_index_pairs:
		matched_nodes_pairs = {} 
		left_index = item[0]
		right_index = item[1]
		common_nodes = item[2]
		original_size = len(left_cmty_list[left_index]) if len(left_cmty_list[left_index]) <= len(right_cmty_list[right_index]) else len(right_cmty_list[right_index])

		print "%i -> %i common nodes: %d" % (left_index,right_index,common_nodes)
		if common_nodes == 0:
			continue
		#tansfrom the community to the graph
		G1 = left_graph.subgraph(left_cmty_list[left_index])
		G2 = right_graph.subgraph(right_cmty_list[right_index])

		#judge the small number of nodes of graph
		small_G = G1 if G1.order() <= G2.order() else G2
		long_G = G1 if G1.order() > G2.order() else G2

		
		#mapping the nodes between the communities
		#method 1:bipartite
		matches = dm.bipartite_matching(small_G,long_G)
		#print "map prob maxtrix....."
		#small_node,long_node,P = dm.map_prob_maxtrix(small_G,long_G,dimensions=dimensions)
		#count = 0
		#long_loop_count = long_G.order()
		deepwalk_common_nodes_list = []	


		#for i in range(len(small_node)):
		#	#obtain ith row
		#	similarity_list =[(index,P[i][index]) for index in range(long_loop_count)]
		#	similarity_list = sorted(similarity_list,key=lambda x:x[1],reverse = True)
		#	while len(similarity_list) > 0:
		#	#record the matched nodes obtained by deepwalk	
		#		#obtain the similiarity list of the best one similiaried with the ith cmty from big community list
		#		best_score = similarity_list[0][1]
		#		C_index = similarity_list[0][0]

		#		similarity_list.pop(0)

		#		dest_similarity_list = []	
		#		for item in P:
		#			dest_similarity_list.append(item[C_index])

		#		dest_similarity_list = sorted(dest_similarity_list,reverse=True)
		#		if best_score < dest_similarity_list[0]:
		#			continue;
		#		break

		#	if len(similarity_list) == 0:
		#		#not finding the best matched node
		#		continue
		#	
		#	#record the nodes pairs which is same one node judeged by algrithm
		#	if best_score >= 1.0:
		#		matched_nodes_pairs[small_node[i]] = long_node[C_index]
		#		if small_node[i] == long_node[C_index]:
		#			#real seed node
		#			count += 1
		#			deepwalk_common_nodes_list.append(small_node[i])

		#deepwalk_matched_nodes_size = len(matched_nodes_pairs)

		#matched_nodes_number.append([original_size,common_nodes,deepwalk_matched_nodes_size,count])
		deepwalk_matched_nodes_size = len(matches)
		count = 0
		for item in matches:
			if item[0] == item[1]:
				count += 1
				deepwalk_common_nodes_list.append(item[0])
		print "matched nodes count: %d" % deepwalk_matched_nodes_size 
		print "real seed nodes count: %d" % count 

		
		print "identification of seed rate....."
		if count == 0:
			seed_nodes_list.append([])
			seed_rate.append(0)
			matched_nodes_number.append([original_size,common_nodes,deepwalk_matched_nodes_size,count,0,0])
			print "identification of seed rate...OK"
			continue

		# seed nodes detection by using the edge_consisitency method	
		#temp_list,temp_count,temp_rate = obtain_optimal_edges_consistency_throd(small_G,long_G,matched_nodes_pairs,deepwalk_common_nodes_list)

		#seed nodes detection by using edge_cridibility method
		#temp_list,temp_count,temp_rate = obtain_seed_with_edges_credibility(matched_nodes_pairs,small_G,long_G,deepwalk_common_nodes_list)

		#seed nodes detection by using bipartite method
		temp_list,temp_count,temp_rate = obtain_seed_with_edges_credibility(matches,small_G,long_G,deepwalk_common_nodes_list)
		seed_rate.append(temp_rate)
		seed_nodes_list.append(temp_list)
		matched_nodes_number.append([original_size,common_nodes,deepwalk_matched_nodes_size,count,len(temp_list),temp_count])
		
	print "map prob maxtrix....OK"
	return matched_nodes_number,seed_rate,seed_nodes_list

def obtain_optimal_edges_consistency_throd(small_G,long_G,matched_nodes_pairs,deepwalk_common_nodes_list):
	'''
	obtain the seed nodes list with the threshold 
	th number of the nodes of the @small_G is less than @long_G
	the @matched_nodes_pairs consisted of pairs of matched nodes just like [[1,1],[3,4]...] which means node 3 of the small_G is same with node 4 of the long_G
	the @deepwalk_common_nodes_list consisted of the same nodes that come from the @small_G 

	Return:
		@eligible_nodes: list, consisted of the all seed nodes index obtained by edge_consistency_method
		@count: int, the number of the real seed nodes
		@rate: float, the accuracy of seed nodes detection by using the edge_consistency_method
	'''
	source_matched_nodes = [node for node in matched_nodes_pairs]	
	subsmall_G = small_G.subgraph(source_matched_nodes)

	subsmall_nodes_degree = subsmall_G.degree()
	degree_list = [subsmall_nodes_degree[key] for key in subsmall_nodes_degree]
	source_size = len(source_matched_nodes)

	#calculate the threshold vlue
	midian = obtain_midian_list(degree_list)
	average = float(sum(degree_list)) / len(degree_list)

	dest_matched_nodes = [matched_nodes_pairs[node] for node in matched_nodes_pairs]
	sublong_G = long_G.subgraph(dest_matched_nodes)

	eligible_nodes = []
	while len(source_matched_nodes) > 0:
		node = source_matched_nodes.pop(0)
		count = 0
		dest_node = matched_nodes_pairs[node]
		dest_node_neighbors = sublong_G.neighbors(dest_node)
		node_neighbors = subsmall_G.neighbors(node)
		for n_node in node_neighbors:
			matched_node = matched_nodes_pairs[n_node] 
			if matched_node  in dest_node_neighbors:
				count += 1

		if count >= midian and count > 1:
			eligible_nodes.append(node)	

	print "seed list total len: %d" % len(eligible_nodes) 
	count = 0
	if len(eligible_nodes) == 0:
		print "real seed: %d" % count
		print "seed rate: 0" 
		return eligible_nodes,0,0
	for node in eligible_nodes:
		if node in deepwalk_common_nodes_list:
			#real seed nodes
			count += 1

	rate = float(count)/len(eligible_nodes)
	print "real seed: %d" % count
	print "seed accuracy rate: %.2f" % rate
	return eligible_nodes,count,rate
		
def obtain_optimal_edges_consistency(small_G,long_G,matched_nodes_pairs,deepwalk_common_nodes_list):
	
	#initialize the list consisted of the matched nodes
	source_matched_nodes = [node for node in matched_nodes_pairs]	

	subsmall_G = small_G.subgraph(source_matched_nodes)
	subsmall_nodes_degree = subsmall_G.degree()
	degree_list = [subsmall_nodes_degree[key] for key in subsmall_nodes_degree]
	source_size = len(source_matched_nodes)
	limit_size = obtain_midian_list(degree_list)
	print "limit of the degree: %.2f" % limit_size 
	average_value = float(sum(degree_list)) / len(degree_list)
	dest_matched_nodes = [matched_nodes_pairs[node] for node in matched_nodes_pairs]
	
	sublong_G = long_G.subgraph(dest_matched_nodes)
	subsmall_G = small_G.subgraph(source_matched_nodes)
	front_nodes = subsmall_G.number_of_nodes()
	front_edges = subsmall_G.number_of_edges()
	while True:
		#obtain the degree list of the nodes in the subgraph
		subsmall_nodes_degree = subsmall_G.degree()
		subsmall_nodes_degree = sorted(subsmall_nodes_degree.items(),key=lambda x:x[1],reverse=True)
		for node in subsmall_nodes_degree:
			node = node[0]
			count = 0
			dest_node = matched_nodes_pairs[node]
			dest_node_neighbors = sublong_G.neighbors(dest_node)
			node_neighbors = subsmall_G.neighbors(node)
			len_node_neighbors = len(node_neighbors)
			if len_node_neighbors == 0:
				subsmall_G.remove_node(node)
				continue
			for n_node in node_neighbors:
				matched_node = matched_nodes_pairs[n_node] 
				if matched_node  in dest_node_neighbors:
					count += 1
			if count == len_node_neighbors:# and count > limit_size*0.5:
				continue
			else:
				subsmall_G.remove_node(node)

					#if n_node not in new_source_matched:
					#	new_source_matched.append(n_node)
					#if node not in new_source_matched:
					#	new_source_matched.append(node)

		if front_edges == subsmall_G.number_of_edges() and front_nodes == subsmall_G.number_of_nodes():
			break 
		else:
			front_nodes = subsmall_G.number_of_nodes()
			front_edges = subsmall_G.number_of_edges()
	
	print "seed list total len: %d" % front_nodes 
	count = 0
	new_source_matched = subsmall_G.nodes()
	if front_nodes == 0:
		print "real seed: %d" % count
		print "seed rate: 0" 
		return new_source_matched,0,0
	for node in new_source_matched:
		if node in deepwalk_common_nodes_list:
			count += 1
	rate = float(count)/front_nodes
	print "real seed: %d" % count
	print "seed accuracy rate: %.2f" % rate
	return new_source_matched,count,rate
			
def draw_subgraph_after_deepwalk(small_G,long_G,matched_nodes_pairs,deepwalk_common_nodes_list):
	partitions = []	
	partitions.append([node for node in matched_nodes_pairs if node not in deepwalk_common_nodes_list])
	partitions.append([matched_nodes_pairs[node] for node in matched_nodes_pairs if matched_nodes_pairs[node] not in deepwalk_common_nodes_list])
	sub_nodes_list = partitions[0] + partitions[1]
	subgraph_small = small_G.subgraph(sub_nodes_list)
	draw_partitions(subgraph_small,partitions)
	partitions_matched = []
	partitions_matched.append([node for node in matched_nodes_pairs if node not in deepwalk_common_nodes_list])
	partitions_matched.append([matched_nodes_pairs[node] for node in matched_nodes_pairs if matched_nodes_pairs[node] not in deepwalk_common_nodes_list])
	sub_nodes_list_matched = partitions_matched[0] + partitions_matched[1]
	subgraph_big = long_G.subgraph(sub_nodes_list_matched)
	draw_partitions(subgraph_big,partitions_matched)
	return

def obtain_seed_accuracy_rate_with_node_feature(small_G,long_G,matched_nodes_pairs,deepwalk_common_nodes_list):
	'''
	obtain zhe seed nodes and the fraction of the number of the nodes in the list specified by argument @deepwalk_common_nodes_list 
	return the seed nodes array and the accuracy rate between number of the nodes which locate at both   seed nodes array and deepwalk_common_list and the number of the deepwalk_common_nodes_list
	Return type: list for the first one and float for another
	'''

	eligible_seed_nodes_list = []
	seed_nodes_list = []
	if len(deepwalk_common_nodes_list) == 0:
		return 0,seed_nodes_list,eligible_seed_nodes_list
	small_G_edges_list = small_G.edges()
	long_G_edges_list = long_G.edges()
	
	matched_left_nodes_list = matched_nodes_pairs.keys()
	reverse_matched_nodes_pairs = dict(izip(matched_nodes_pairs.itervalues(),matched_nodes_pairs.iterkeys()))
	matched_right_nodes_list = reverse_matched_nodes_pairs.keys()

	while len(matched_left_nodes_list) > 0:
		source_node = matched_left_nodes_list.pop(0)
		dest_node = matched_nodes_pairs[source_node]
		dest_neighbor_nodes_list = long_G.neighbors(dest_node)
		source_neighbor_nodes_list = small_G.neighbors(source_node)

		#obtain the feature of the source node
		source_node_degree = small_G.degree(source_node)
		source_matched_neighbor_nodes_list = []
		edges_uniformity_count = 0
		for node in source_neighbor_nodes_list:
			if node in matched_left_nodes_list:
				temp_matched_node = matched_nodes_pairs[node]
				source_matched_neighbor_nodes_list.append(node)
				if temp_matched_node in matched_right_nodes_list and temp_matched_node in dest_neighbor_nodes_list:
					edges_uniformity_count += 1

		source_matched_neighbor_nodes_count = len(source_matched_neighbor_nodes_list)
		source_neighbor_degree_list = list(small_G.degree(source_neighbor_nodes_list).values())
		#construct the feature oif the source node with the three arguments: the degree,the number of the matched neighbors and the number of the uniformity of the edges
		source_node_feature = [source_node_degree,source_matched_neighbor_nodes_count,edges_uniformity_count]
		triangles_count = obtain_triangles_count(small_G,source_node)
		source_node_feature.append(triangles_count)
		if len(source_neighbor_degree_list) > 0:
			source_node_feature.append(max(source_neighbor_degree_list))
			source_node_feature.append(min(source_neighbor_degree_list))
			source_node_feature.append(float(sum(source_neighbor_degree_list)) / len(source_neighbor_degree_list))
		else:
			source_node_feature.append(0)
			source_node_feature.append(0)
			source_node_feature.append(0)

		#obtain the feature of the dest node
		dest_node_degree = long_G.degree(dest_node)
		dest_matched_neighbor_nodes_list = []
		dest_edges_uniformity_count = 0
		for node in dest_neighbor_nodes_list:
			if node in matched_right_nodes_list:
				temp_matched_node = reverse_matched_nodes_pairs[node]
				dest_matched_neighbor_nodes_list.append(node)
				if temp_matched_node in matched_left_nodes_list and temp_matched_node in source_neighbor_nodes_list:
					dest_edges_uniformity_count += 1
		
		dest_matched_neighbor_nodes_count = len(dest_matched_neighbor_nodes_list)
		#calculate the degree of the macthed neighbor nodes degree
		dest_neighbor_degree_list = list(long_G.degree(dest_neighbor_nodes_list).values())
		#construct the feature oif the dest node with the three arguments: the degree,the number of the matched neighbors and the number of the uniformity of the edges
		dest_node_feature = [dest_node_degree,dest_matched_neighbor_nodes_count,dest_edges_uniformity_count]

		triangles_count = obtain_triangles_count(long_G,dest_node)
		dest_node_feature.append(triangles_count)
		if len(dest_neighbor_degree_list) > 0:
			dest_node_feature.append(max(dest_neighbor_degree_list))
			dest_node_feature.append(min(dest_neighbor_degree_list))
			dest_node_feature.append(float(sum(dest_neighbor_degree_list)) / len(dest_neighbor_degree_list))
		else:
			dest_node_feature.append(0)
			dest_node_feature.append(0)
			dest_node_feature.append(0)
	
		similarity_score = euclidean_distance(source_node_feature,dest_node_feature)
		eligible_seed_nodes_list.append([source_node,dest_node,similarity_score])

	eligible_seed_nodes_list = sorted(eligible_seed_nodes_list,key=lambda x:x[2])
	print eligible_seed_nodes_list[:15]
	predicted_seed_count = int(len(eligible_seed_nodes_list) * 0.15)
	for item in eligible_seed_nodes_list[:predicted_seed_count]:
		if item[0] in deepwalk_common_nodes_list:
			seed_nodes_list.append(item[0])
	seed_node_number = len(seed_nodes_list)
	print "number of the eligible nodes: %d" % predicted_seed_count 
	print "number of common nodes: %d" % seed_node_number
	if predicted_seed_count == 0:
		rate = 0
	else:
		rate = float(seed_node_number) /predicted_seed_count 
	return rate,seed_nodes_list,eligible_seed_nodes_list[:predicted_seed_count]


def main():
	if len(sys.argv) < 7:
		print "usage: ./deepmatching_for_cmty.py [filename] [sample rate]  [community size threshold] [loop count] [distance function = 1] [remarks]"
		return -1
	sample_rate = float(sys.argv[2])
	throd_value = int(sys.argv[3])
	repeated_count = int(sys.argv[4])
	if(sys.argv[5] == "1"):
		method_select = euclidean_distance
	else:
		method_select = euclidean_metric

	filename = "cmty_matching_with_sample_%.2f_repeat_%d_cmty_throd_%d.txt"%(float(sys.argv[2]),repeated_count,throd_value)
	print "result will be recorded in %s"%filename
	df = open(filename,"w")
	df.write("########################################################################\n")
	df.write("date: %s\n" % ti.strftime("%Y-%m-%d %H:%M:%S",ti.localtime(ti.time())))
	df.write("remarks: %s\n" % sys.argv[6])
	df.write("dataset: %s\n" % sys.argv[1])
	df.write("sample: %.2f\n" % float(sys.argv[2]))
	df.write("similarity function: %s" % "euclidean_distance")
	df.write("throd of the community: %d\n" % throd_value)
	df.write("repeated loop count: %d\n" % repeated_count)
	df.flush()

	#load graph form file
	nx_G = load_graph_from_file(sys.argv[1],delimiter = ' ')

	df.write("Graph Infomation: nodes %d edges %d\n" % (len(nx_G.nodes()),len(nx_G.edges())))
	df.flush()

	print "begin to map community....."	
	sum_acc = 0.0
	rate_list = []
	dimensions = 120 
	for i in range(repeated_count):
		print "->%d"%i ,
		sys.stdout.flush()
		G1,G2 = bi_sample_graph(nx_G,sample_rate)

		#obtain the pairs of index of the matched communities
		old_rate,left_graph,right_graph,left_cmty_list,right_cmty_list,matched_index = repeated_eavalute_accuracy_by_feature(G1,G2,limit_cmty_nodes = throd_value)
		#record the small commnuity size
		df.write("%ith loop:\n"%i)
		df.write("G1 cmty length distribution: ")
		for index in left_cmty_list:
			df.write("%i " % (len(index)))
			df.flush()
		df.write("\n")
		df.write("G2 cmty length distribution: ")
		for index in right_cmty_list:
			df.write("%i " % (len(index)))
			df.flush()
		df.write("\n")

		rate_list.append(old_rate)
		sum_acc += old_rate
		
		# the dimensions of feature of the nodes obtained by deepwalk 
		matched_nummber_nodes,seed_rate,seed_nodes_list = obtain_accuracy_rate_in_matched_cmty(left_graph,left_cmty_list,right_graph,right_cmty_list,matched_index,dimensions) 
		df.write("dimensions: %d\n" % dimensions)
		df.write("deepwalk results[cmty-deepwalk-seed]: ")
		for item in matched_nummber_nodes:
			df.write("%d-%d-%d-%d-%d-%d " % (item[0],item[1],item[2],item[3],item[4],item[5]))
			df.flush()
		df.write('\n')
		df.flush()
		#finding the seed nodes and calculating the accuracy rate
		df.write("seed accuracy rate: ")
		for item in seed_rate:
			df.write(" %.5f" % item)
			df.flush()
		df.write('\n')
		df.flush()
			
	df.write("accuracy rate array of community mapping: ")
	df.flush()
	for item in rate_list:
		df.write("%.4f " % item)
		df.flush()
	df.write("\n")
	df.write("arverage: %.5f\n"% (sum_acc / repeated_count))
	df.write("########################################################################\n")
	df.flush()

	print "mapping community finished!!"
	return 0
	

	
if __name__ == "__main__":
    main()

