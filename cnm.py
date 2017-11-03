#!/usr/bin/python
import os
import networkx as nx
import matplotlib.pyplot as plt
import time
import sys
import snap
import community as cmty
#from communityMatching import *

def graph_for_test():
	TG = snap.TUNGraph.New()
	for i in range(0,12):
		TG.AddNode(i)
	TG.AddEdge(0,1)
	TG.AddEdge(0,2)
	TG.AddEdge(0,3)
	TG.AddEdge(1,2)
	TG.AddEdge(1,3)
	TG.AddEdge(4,5)
	TG.AddEdge(4,6)
	TG.AddEdge(5,6)
	TG.AddEdge(7,8)
	TG.AddEdge(7,9)
	TG.AddEdge(7,10)
	TG.AddEdge(8,9)
	TG.AddEdge(8,10)
	TG.AddEdge(9,10)
	TG.AddEdge(9,11)
	TG.AddEdge(10,11)
	TG.AddEdge(3,6)
	TG.AddEdge(8,2)
	TG.AddEdge(5,11)
	return TG

def create_graph_from_file(filename):
	sf = open(filename)
	lines = sf.readlines()
	TG = snap.TUNGraph.New()
	for item in lines:
		if '#' in item:
			continue
		item = item[:-1]
		item = item.split('\t')
		if TG.IsNode(int(item[0])) == False:
			TG.AddNode(int(item[0]))
		if TG.IsNode(int(item[1])) == False:
			TG.AddNode(int(item[1]))
		TG.AddEdge(int(item[1]),int(item[0]))
	print "load nodes : %d" % TG.GetNodes()
	sf.close()
	return TG
		

def generate_graph(G,nodes_list=[]):
	TG = snap.TUNGraph.New()
	length = len(nodes_list)
	for i in range(0,length):
		TG.AddNode(nodes_list[i])
	for i in nodes_list:
		for j in nodes_list:
			if G.IsEdge(i,j):
				TG.AddEdge(i,j)
	return TG

def community_cnm(G):
	CmtyV = snap.TCnComV()
	modularity = snap.CommunityCNM(G,CmtyV)
	#print modularity
	ret_list = []
	for Cmty in CmtyV:
		temp = []
		for NI in Cmty:
			temp.append(NI)
		ret_list.append(temp)
	return ret_list
def community_gn(G):
	CmtyV = snap.TCnComV()
	modularity = snap.CommunityGirvanNewman(G,CmtyV)
	ret_list = []
	for Cmty in CmtyV:
		temp = []
		for NI in Cmty:
			temp.append(NI)
		ret_list.append(temp)
	return ret_list

def community_best_partition_with_limit(G,limit_nodes):
	loop_throd_value = 30
	finished_list = []
	unfinished_list = []
	total_G_nodes_number =  G.number_of_nodes()
	unfinished_list.append(G.nodes())
	print "community detection with limit by using best partition starts"
	sys.stdout.flush()
	last_remain_len = len(unfinished_list) 
	remain_loop_count = 0
	while len(unfinished_list) > 0:
		result_nodes = []
		list_nodes = unfinished_list.pop()
		#create Graph for best partition 
		if len(list_nodes) != total_G_nodes_number:
			subG = G.subgraph(list_nodes)
		else:
			subG = G

		#community detection with best partiton 
		subG_partition = cmty.best_partition(subG)
		#check the number of the nodes in the community
		subG_cmty_lists = []

		for com in set(subG_partition.values()):
			temp = [nodes for nodes in subG_partition.keys() if subG_partition[nodes] == com]
			subG_cmty_lists.append(temp)

		for item in subG_cmty_lists:
			if(len(item) > limit_nodes):
				unfinished_list.append(item)
			else:
				finished_list.append(item)

		print "the size of finished_list : %d " % len(finished_list)
		sys.stdout.flush()
		print "the size of unfinished_list : %d " % len(unfinished_list)
		sys.stdout.flush()
		current_remain_len = len(unfinished_list)
		if current_remain_len == last_remain_len:
			remain_loop_count += 1
			if remain_loop_count >= loop_throd_value:
				break
		else:
			remain_loop_count = 0 
			last_remain_len = current_remain_len
	print "community detection with limit by using best partition has finished!!"
	sys.stdout.flush()
	return finished_list

def community_cnm_with_limit(G,limit_nodes):
	finished_list = []
	unfinished_list = []
	graph_nodes = []
	for i in G.Nodes():
		graph_nodes.append(i.GetId())
	unfinished_list.append(graph_nodes)
	print "detection starts"
	sys.stdout.flush()
	CmtyV = snap.TCnComV()
	while len(unfinished_list) > 0:
		result_nodes = []
		list_nodes = unfinished_list.pop()
		#create Graph for CNM
		if len(list_nodes) != G.GetNodes():
			TG = generate_graph(G,list_nodes)
		else:
			TG = G
		#community detection with CNM
		modularity = snap.CommunityCNM(TG,CmtyV)
		#judge the number of the nodes in the community detected by the CNM
		for Cmty in CmtyV:
			result_nodes = []
			for NI in Cmty:
				result_nodes.append(NI)
			if(len(result_nodes) > limit_nodes):
				unfinished_list.append(result_nodes)
			else:
				finished_list.append(result_nodes)

		print "the size of finished_list : %d " % len(finished_list)
		sys.stdout.flush()
		print "the size of unfinished_list : %d " % len(unfinished_list)
		sys.stdout.flush()
		if TG != G:
			TG.Clr()
	print "detection end"
	sys.stdout.flush()
	return finished_list


def main():

	nx_G = load_graph_from_file(sys.argv[1],delimiter = ' ')
	print ("Graph Infomation: nodes %d edges %d\n" % (len(nx_G.nodes()),len(nx_G.edges())))
	ret_list = community_best_partition_with_limit(nx_G,500)
	print "number of communities: %d" % (len(ret_list))	
	#dataset_source = os.path.abspath(sys.argv[1])
	#log = open("%s.log" % dataset_source,"a")
	#log.write("time: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	#log.write("dataset: %s\n" % dataset_source)
	#log.write("create graph start!!\n")
	#print "create graph start!!"
	#separator = sys.argv[2][0]
	#if sys.argv[2] == "tab":
	#	separator = '\t'
	#log.write("separator: %c\n" % separator)
	#log.flush()
	#print "dataset file: %s " % sys.argv[1]
	#print "separator %c " % separator

	#G = snap.LoadEdgeList(snap.PUNGraph,sys.argv[1],0,1,separator) 
	#print "create graph finish!!"
	#log.write("create graph finish!!\n")
	#run_time = time.time()

	##detecting  with limition of communitt nodes
	##log.write("community nodes limit: %d\n" % int(sys.argv[3]))
	##ret_list = community_cnm(G,int(sys.argv[3]))

	##detecting  without limition of communitt nodes
	#CmtyV = snap.TCnComV()
	#modularity = snap.CommunityCNM(G,CmtyV)
	#ret_list = []
	#for Cmty in CmtyV:
	#	temp = []
	#	for NI in Cmty:
	#		temp.append(NI)
	#	ret_list.append(temp)
	#end_time = time.time()
	#log.write("community number: %d\n" % len(ret_list))
	#log.flush()

	##record the community
	#for item in ret_list:
	#	print item
	#	temp = ','.join(str(node) for node in item)
	#	print temp
	#	log.write("%s\n" % temp)
	#	log.flush()
	#log.write("running time : %d mins %d secs\n" % (int(end_time - run_time)/60,int(end_time - run_time) % 60))
	#log.flush()
	#log.close()
	#print "running time : %d mins %d secs" % (int(end_time - run_time)/60,int(end_time - run_time) % 60)

	return 0


if __name__ == "__main__":
	main()
