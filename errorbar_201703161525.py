#!/usr/bin/env python

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  MultipleLocator
import mpl_toolkits.mplot3d 

import sys

def draw_sample_accuracy_rate_errorbar(accuracy_sample_dic):
	'''
	draw errorbar
	x axis: sample range
	y axis: accuracy rate
	'''
	x_range = accuracy_sample_dic.keys()
	x_range = sorted(x_range)
	print x_range
	y_range = []
	y_std = []
	for key in x_range:
		temp = accuracy_sample_dic[key]
		temp = np.array(temp)
		y_std.append(np.sqrt((temp.var() * temp.size) / (temp.size - 1)));
		average = float(sum(temp)) / len(temp) 
		y_range.append(average)
	
	fx,ax = plt.subplots()
	ax.errorbar(x_range,y_range,yerr=y_std,fmt="r-o" ,color = "green",elinewidth=2)
	ax.yaxis.set_major_locator(MultipleLocator(0.05));
	ax.set_xticks([i/100. for i in range(0,120,5)])
	plt.title("Accuract rate in different sample %i - %i" % (x_range[0],x_range[-1]))
	plt.ylabel("Accuracy Rate")
	plt.xlabel("Sample Range")
	plt.show()
	return


def draw_sample_accuracy_rate(filename,accuracy_sample_dic):
	"""
	draw the error bar with the community matching results
	the function will handle the data from the file speacied by @filename and draw the figure
	xaxis: sample rate 
	yaxis:accurate
	Return: void
	"""
	sample_rate = 0.0
	repeated_count = 0
	txt = open(filename)
	ret = txt.readlines()

	accuracy_rate = []
	for line in ret:
		if '#' in line:
			continue
		if 'sample' in line:
			line = line.split(' ')
			sample_rate = float(line[1][:-1])
			print "sample : %.5f" % sample_rate
			continue
		if 'repeated' in line:
			line = line.split(' ')
			repeated_count = int(line[-1][:-1])
			print "repeat count: %d" % repeated_count
			continue
		if 'accuracy' in line:
			line = line.split(' ')
			for i in range(3,repeated_count  +3):
				temp = float(line[i])
				accuracy_rate.append(temp)
			continue
	accuracy_sample_dic[sample_rate] = accuracy_rate

	#x_repeated = [i for i in range(1,repeated_count + 1)]
	#fx,ax = plt.subplots()
	#plt.plot(x_repeated,accuracy_rate,'o-')
	#ax.set_xticks([i for i in range(repeated_count + 2)])
	#ax.set_yticks(accuracy_rate)
	#plt.title("Accuract rate repteated  %d times " % repeated_count)
	#plt.ylabel("Accuracy Rate")
	#plt.xlabel("Repeat Range")
	#plt.show()
	return


def draw_dimensions_accuracy_errorbar(filename):
	"""
	task A
	the function will handle the data from the file speacied by @filename and draw the figure
	xaxis: dimensions range
	yaxis:accurate
	Return: void
	"""
	X = [] 
	Y = []
	Ystd = []
	txt = open(filename)
	ret = txt.readlines()

	ycurrent = []
	ystd = []
	for line in ret:
		line = line.split(',')
		X.append(int(line[0]))
		line = [float(i) for i in line[1:]]
		temp = np.array(line)
		Ystd.append(np.sqrt((temp.var() * temp.size) / (temp.size - 1)));
		Y.append(np.mean(line));

	fx,ax = plt.subplots()
	ax.errorbar(X, Y,yerr=Ystd,fmt="r-o" ,color = "green",elinewidth=2)
	#ax.set_xticks(X)
	ax.xaxis.set_major_locator(MultipleLocator(2));
	ax.axis([min(X) - 2 ,max(X) + 1,min(Y) - max(Ystd),max(Y) + 2*max(Ystd)])

	#plt.tick_params(axis="both",which="both",bottom="off",top="off",labelbottom="on",left="off",right="on",labelleft="on")
	plt.title("Accuract rate in different dimension %i - %i" % (X[0],X[-1]))
	plt.ylabel("Accuracy Rate")
	plt.xlabel("Dimension Range")
	plt.show()
	return
def draw_dimensions_accuracy_errorbar_sample(filelist):
	"""
	task B
	the function will extract the data from the filename array speacied by @filelist and draw the figure with the
	data.
	xaxis: dimensions range
	yaxis: accurate
	Return: void
	"""
	X = [] 
	Y = []
	Ystd = []
	sample_ratio = [str(item/100.) for item in range(55,100,5)]

	fx,ax = plt.subplots()

	for filename in filelist:
		print " %s-> " % filename
		txt = open(filename)
		x_per = []
		y_per = []
		ystd_per = []
		ret = txt.readlines()
		for line in ret:
			line = [float(i) for i in line[:-2].split(',')]
			x_per.append(int(line[0]))
			temp = np.array(line[1:])
			ystd_per.append(np.sqrt((temp.var() * temp.size) / (temp.size - 1)));
			#print np.sqrt((temp.var() * temp.size) / (temp.size - 1))
			y_per.append(np.mean(line[1:]));
		X.append(x_per)
		Y.append(y_per)
		Ystd.append(ystd_per)

	X = np.array(X)
	Y = np.array(Y)
	Ystd = np.array(Ystd)
	
	data_row,data_cul = X.shape
	for i in range(0,data_row):
			ax.errorbar(X[i],Y[i],Ystd[i],fmt = '-s',elinewidth = 2)
	ax.legend(sample_ratio[:len(filelist)],loc="upper right",ncol = 1,numpoints=1,title="Sample Ratio")
	ax.axis([X.min() - 1 ,X.max() + 3,Y.min() - Ystd.max(),1])

	plt.title("Accuract rate in different dimension:%i - %i with sample:%s - %s " % (X.min(),X.max(),sample_ratio[0],sample_ratio[len(filelist) - 1]))
	ax.set_xticks([i for i in range(X.min() - 1,X.max() + 3,1)])
	ax.set_yticks([i/100. for i in range(0,110,5)])
	ax.xaxis.set_major_locator(MultipleLocator(1));
	ax.yaxis.set_major_locator(MultipleLocator(0.05));
	plt.ylabel("Accuracy Rate")
	plt.xlabel("Dimension Range")

	plt.tight_layout()
	plt.show()
	return

def get_p_q_accuracy_data(filename):
	"""
	task C
	the function will handle the file specied by @filename and return the data as an array
	"""
	data = []
	
	file = open(filename)
	for line in file.readlines():
		line = line[:-2].split(',')
		data.append([float(line[0]),float(line[2]),float('%.5f' % np.mean([float(i) for i in line[4:]]))])
	return  data



def draw_p_q_accuracy_3D(data):
	"""
	the function will draw 3D with data specied by @data 
	the format of data must like "[[p1,q1,result1],[p2,q2,result2],[p3,q3,result3],....]"
	zaxis : accuracy
	xaxis : value of p
	yaxis : value of q
	Return : void
	"""
	x = data[:,0]
	y = data[:,1]
	z = data[:,2]
	#x, y = np.meshgrid(x, y)
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.set_xticks(x)
	ax.set_yticks(y)
	#ax.set_zticks([i / 100. for i in range(int(min(z) * 100),int(max(z)*100),1)])
	#ax.zaxis.set_major_locator(MultipleLocator(0.5));
	ax.set_zlim(0.5,1)

	x = np.array(x)
	x = x.reshape(len(x)/10,10)
	y = np.array(y)
	y = y.reshape(len(y)/10,10)
	z = np.array(z)
	z = z.reshape(len(z)/10,10)


	ax.scatter(x,y,z,s = 5,c = 'r')
	#ax.plot_wireframe(x,y,z,rstride=1,cstride=1)
	surf = ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0,antialiased=False,alpha=0.3)
	fig.colorbar(surf, shrink=1, aspect=10)


	ax.set_zlabel('Accuract Mean Ratio')
	ax.set_xlabel('p')
	ax.set_ylabel('q')

	plt.tight_layout()
	plt.show()
	return



def draw_degree_runtime(filename):
	'''
	task D_1
	the function will draw the 2D figure with the data from the file speacied by @filename
	xaxis: degree
	yaxis: runtime
	Return : void
	'''
	X = []
	Y = []
	data = open(filename)
	for line in data.readlines():
		line = line.split(',')
		X.append(int(line[0]))
		Y.append(float('%.5f' % float(line[-2])))


	fx,ax = plt.subplots()
	ax.plot(X,Y)
	ax.set_ylabel('Run Time')
	ax.set_xlabel('Dimensions')
	ax.set_xticks(X)
	ax.set_yticks(Y)
	plt.show()


def draw_edge_runtime(filename):
	'''
	task D_2
	the function will draw the 2D figure with the data from the file speacied by @filename
	xaxis: edges 
	yaxis: runtime
	Return : void
	'''
	X = []
	Y = []
	data = open(filename)
	for line in data.readlines():
		line = line.split(',')
		X.append(500 * int(line[0]))
		Y.append(float('%.5f' % float(line[-2])))


	fx,ax = plt.subplots()
	plt.plot(X,Y,'r-o')
	ax.set_ylabel('Run Time')
	ax.set_xlabel('Edge')
	#ax.xaxis.set_major_locator(MultipleLocator(1));
	#ax.yaxis.set_major_locator(MultipleLocator(1000));
	#ax.set_xticks(X)
	#ax.set_yticks(Y)
	plt.show()



def main():
	
	#draw_dimensions_accuracy_errorbar(sys.argv[1])
	accuracy_sample_dic = {}
	for i in range(1,len(sys.argv)):
		draw_sample_accuracy_rate(sys.argv[i],accuracy_sample_dic)
	draw_sample_accuracy_rate_errorbar(accuracy_sample_dic)
	#draw_edge_runtime(sys.argv[1])



if __name__ == "__main__":
    main()
