#!/usr/bin/python
import numpy as np
import time as ti
import networkx as nx
from functools import partial
from scipy.io import loadmat
import matplotlib.pyplot as plt
from cpdcore import DeformableRegistration, RigidRegistration, AffineRegistration
from sklearn.decomposition import PCA
from node2vec import learn_nodevec
import time
from graph_matching import sample_graph
import sys

def visualize(X, Y, ax=None, words = None):
    pass
    # plt.cla()
    # ax.scatter(X[:,0] ,  X[:,1], color='red')
    # ax.scatter(Y[:,0] ,  Y[:,1], color='blue')
    # if words != None:
    #     index = 0
    #     for x, y in zip(X[:,0],  X[:,1]):
    #         plt.text(x, y, words[index], color = 'red' )
    #         index += 1
    #
    #     index = 0
    #     for x, y in zip(Y[:, 0], Y[:, 1]):
    #         plt.text(x, y, words[index], color='blue')
    #         index += 1
    # plt.draw()
    # plt.pause(0.001)

def nodes_embedding(G, p=1, q=1, dimensions=128):
    # Extract the features for each node using node2vec
    model = learn_nodevec(G, dimensions=dimensions, argp=p, argq=q, num_walks=100)
    nodes = [word for word, vcab in model.vocab.iteritems()]
    inds = [vcab.index for word, vcab in model.vocab.iteritems()]
    X = model.syn0[inds]
    return nodes,X


def map_prob_maxtrix(G1, G2, p=1, q=1, dimensions=128):
    # match the nodes according to the node feature based on Coherent Point Drift
    nodes1, X = nodes_embedding(G1, p = p, q=q, dimensions=dimensions)
    nodes2, Y = nodes_embedding(G2, p = p, q=q, dimensions=dimensions)
    reg = AffineRegistration(Y, X)
    callback = partial(visualize)
    reg.register(callback)
    P = reg.P
    return (nodes1, nodes2, P)

#def repeated_eavalute_accuracy(repeated=10, p=1.0, q=1.0, dimensions=64, sample_rate=0.9):
#    accuracies = []
#    for i in range(repeated):
#        print i, '->',
#        nx_G = nx.barabasi_albert_graph(500, 5)
#        G1 = sample_graph(nx_G, sample_rate)
#        G2 = sample_graph(nx_G, sample_rate)
#        node1, node2, P = map_prob_maxtrix(G1, G2, p=p, q=q, dimensions=dimensions)
#        count = 0
#        for i in range(len(node2)):
#            # print node1[i], node2[np.array(P[:, i]).argmax()], np.array(P[:, i]).max()
#            if node2[i] == node1[np.array(P[:, i]).argmax()]:
#                count += 1
#        accuracies.append(count * 1.0 / nx_G.order())
#    return accuracies



#def repeated_eavalute_accuracy(repeated=10, p=1.0, q=1.0, dimensions=64, sample_rate=0.9,degree = 5):
#    accuracies = []
#    for i in range(repeated):
#        print i, '->',
#        nx_G = nx.barabasi_albert_graph(500, degree)
#        G1 = sample_graph(nx_G, sample_rate)
#        G2 = sample_graph(nx_G, sample_rate)
#        node1, node2, P = map_prob_maxtrix(G1, G2, p=p, q=q, dimensions=dimensions)
#        count = 0
#        for i in range(len(node2)):
#            # print node1[i], node2[np.array(P[:, i]).argmax()], np.array(P[:, i]).max()
#            if node2[i] == node1[np.array(P[:, i]).argmax()]:
#                count += 1
#        accuracies.append(count * 1.0 / nx_G.order())
#    return accuracies


def repeated_eavalute_accuracy(repeated=10, p=1.0, q=1.0, dimensions=64, nodes = 500,sample_rate=0.9,degree = 5,handle_function = nx.barabasi_albert_graph):
	accuracies = []
	for i in range(repeated):
		print i,'->',
		sys.stdout.flush()
		nx_G = handle_function(nodes, degree)
		G1 = sample_graph(nx_G, sample_rate)
		G2 = sample_graph(nx_G, sample_rate)
		node1, node2, P = map_prob_maxtrix(G1, G2, p=p, q=q, dimensions=dimensions)
		count = 0
		for i in range(len(node2)):
			# print node1[i], node2[np.array(P[:, i]).argmax()], np.array(P[:, i]).max()
			if node2[i] == node1[np.array(P[:, i]).argmax()]:
				count += 1
		accuracies.append(count * 1.0 / nx_G.order())
	return accuracies



def main():
	##try:
	##	print "task bara_D_1 start......"
	##	filename = "./results/dimensions60_bara_alb_p1_q1_100_1000_degree5_sample0.85.csv"
	##	resultsfile = open(filename, 'w')
	##	repeated = 10
	##	for node in range(100,1050,50):
	##		print "handle %d ...." % node
	##		runtime = ti.time()
	##		results = repeated_eavalute_accuracy(repeated=repeated, nodes = node,dimensions=60, sample_rate=0.85,degree=5)
	##		runtime = ti.time() - runtime
	##		resultsfile.write("%d,nodes,0.85,sample ratio" % node)
	##		for result in results:
	##			resultsfile.write(",{:.5f}".format(result))
	##		resultsfile.write(",%.5f,run time" % runtime)
	##		resultsfile.write('\n')
	##		resultsfile.flush()
	##	resultsfile.close()
	##	print "task D_1 complet......"
	##except Exception,e:
	##	print e

	#try:
	#	print "task barar_D_2 start,nodes:500--dim:60--degree:1~20--p:1--q:1--smaple:0.85,......"
	#	filename = "./results/dimensions60_bara_alb_500_degree_1_20_p1_q1_sample0.85.csv"
	#	resultsfile = open(filename, 'w')
	#	repeated = 10
	#	for i in range(1,21):
	#		print "(1~21)handle degree: %d ...." % i 
	#		sys.stdout.flush()
	#		runtime = ti.time()
	#		results = repeated_eavalute_accuracy(repeated=repeated,dimensions=60, sample_rate=0.85,degree=i)
	#		runtime = ti.time() - runtime
	#		resultsfile.write("%d,degree,0.85,sample ratio" % i)
	#		for result in results:
	#			resultsfile.write(",{:.5f}".format(result))
	#		resultsfile.write(",%.5f,run time" % runtime)
	#		resultsfile.write('\n')
	#		resultsfile.flush()
	#	resultsfile.close()
	#	print "task barar_D_2 start,nodes:500--dim:60--degree:1~20--p:1--q:1--smaple:0.85, complete!!!"
	#	print "the results of task barar_D_2 is in %s" % filename
	#except Exception,e:
	#	print e



	try:
		print "task erdos_A start nodes:500--degree:5--p:1--q:1--sample:0.9--dim:100 ~ 160......"
		filename = "./results/dimensions_100_160_erdos_renyi_degree_500_0.01_p1_q1_sample0.9.csv" 
		resultsfile = open(filename,'a')
		repeated = 10
		
		#create graph by nodes and degree

		for i in range(148,160,1):
			print "handle dimension %d ..." % i
			results = repeated_eavalute_accuracy(handle_function = nx.erdos_renyi_graph,repeated = repeated, dimensions=i, sample_rate=0.9,degree = 0.01)
			resultsfile.write(str("%d," % i))
			for result in results:
				resultsfile.write(",{:.5f}".format(result))
			resultsfile.write('\n')
			resultsfile.flush()
		resultsfile.close()
		print "task erdos_A  nodes:500--degree:0.01--p:1--q:1--sample:0.9--dim:100 ~ 160 complete!!!!"
		print "the results of task erdos_A is in %s " % filename
	except Exception,e:
		print e

	#try:
	#	print "task erdos_B start nodes:500--degree:0.1--p:1--q:1--dim:50~70--sample:0.6 ~ 0.95......"
	#	smaple = [i / 100. for i in range(60,100,5)]
	#	repeated = 10

	#	for i in smaple:
	#		filename = "./results/dimensions_erdos_renyi_degree_500_5_p1_q1_sample%.2f.csv" % i 
	#		print "handle %s " % filename
	#		sys.stdout.flush()
	#		resultsfile = open(filename,'a')
	#		for dim in range(50,71):
	#			results = repeated_eavalute_accuracy(handle_function = nx.erdos_renyi_graph,repeated = repeated, dimensions=dim, sample_rate=i,degree = 0.01)
	#			resultsfile.write(str("%d" % dim))
	#			for result in results:
	#				resultsfile.write(",{:.5f}".format(result))
	#			resultsfile.write('\n')
	#			resultsfile.flush()

	#		resultsfile.close()
	#		print "results of sample: %.2f--dim: %d in %s complete!!! " %(i,dim,filename)
	#	print "task erdos_B  nodes:500--degree:0.01--p:1--q:1--dim:50~70--sample:0.6 ~ 0.95 complete"
	#except Exception,e:
	#	print e
	#
	#try:
	#	print "task erdos C start......"
	#	repeated = 10
	#	filename = "./results/pq_erdos_renyi_500_0.01_dimensions64_sample0.8.csv"

	#	resultsfile = open(filename,'w')
	#	for p in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
	#		for q in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
	#			print "handle (%.1f,%.1f) ...." % (p,q) 
	#			sys.stdout.flush()
	#			results = repeated_eavalute_accuracy(handle_function = nx.erdos_renyi_graph,degree=0.01,repeated=repeated, dimensions=64, sample_rate=0.8)
	#			resultsfile.write("%.1f,p,%.1f,q" % (p,q))
	#			for result in results:
	#				resultsfile.write(",{:.5f}".format(result))
	#			resultsfile.write('\n')
	#			resultsfile.flush()
	#	resultsfile.close()
	#	print "task erdos C complete!!!"
	#	print "results of task erdos C is in %s" % filename
	#except Exception,e:
	#	print e

	#try:
	#	print "task erdos D_1 start......"
	#	filename = "./results/dimensions60_erdos_renyi_p1_q1_100_1000_degree5_sample0.85.csv"
	#	print filename
	#	resultsfile = open(filename, 'a')
	#	repeated = 10
	#	for node in range(100,1050,50):
	#		degree = 5 / float(node)
	#		print "(100 ~ 1000)handle %d degree %.3f ...." % (node,degree)
	#		sys.stdout.flush()
	#		runtime = ti.time()

	#		results = repeated_eavalute_accuracy(repeated=repeated, nodes = node,dimensions=60, sample_rate=0.85,degree=degree,handle_function  = nx.erdos_renyi_graph)
	#		runtime = ti.time() - runtime
	#		resultsfile.write("%d,nodes,0.85,sample ratio" % node)
	#		for result in results:
	#			resultsfile.write(",{:.5f}".format(result))
	#		resultsfile.write(",%.5f,run time" % runtime)
	#		resultsfile.write('\n')
	#		resultsfile.flush()

	#	resultsfile.close()
	#	print "task erdos D_1 complet......"
	#except Exception,e:
	#	print e

	#try:
	#	print "task erdos D_2 start......"
	#	filename = "dimensions60_erdos_renyi_degree_1_20_p1_q1_sample0.85.csv"
	#	resultsfile = open(filename, 'w')
	#	repeated = 10
	#	for i in range(1,21):
	#		degree = i / float(500)
	#		print "(1~21)handle %d ...." % i 
	#		sys.stdout.flush()
	#		runtime = ti.time()
	#		results = repeated_eavalute_accuracy(repeated=repeated,dimensions=60, sample_rate=0.85,degree=degree,handle_function  = nx.erdos_renyi_graph)
	#		runtime = ti.time() - runtime
	#		resultsfile.write("%d,degree,0.85,sample ratio" % i)
	#		for result in results:
	#			resultsfile.write(",{:.5f}".format(result))
	#		resultsfile.write(",%.5f,run time" % runtime)
	#		resultsfile.write('\n')
	#		resultsfile,flush()

	#	resultsfile.close()
	#	print "task erdos D_2 complet......"
	#except Exception,e:
	#	print e


    ## nx_G = nx.erdos_renyi_graph(500, 0.01)
    # nx_G = node2vec.read_graph('./data/karate.edgelist')
    # nx.draw_spring(nx_G, with_labels=True)

    # fig = plt.figure()
    # fig.add_axes([0, 0, 1, 1])
    # callback = partial(visualize, ax=fig.axes[0], words=words)
    #
    # reg = AffineRegistration(XP, YP)
    # reg.register(callback)
    # # plt.show()
    # np.savetxt(fname='./results/P.txt', X = reg.P, fmt='%10.5f')
    #
    # P = reg.P


if __name__ == "__main__":
    main()

