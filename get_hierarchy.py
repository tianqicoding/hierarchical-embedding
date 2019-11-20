
import sys
import numpy as np
import networkx as nx
import pandas as pd
import csv
import math
import matplotlib
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, ArrowStyle
from matplotlib.collections import PatchCollection
from matplotlib import patches
import collections
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
import os


def parse_args():

	parser = argparse.ArgumentParser(description='Load Hyperboloid Embeddings and evaluate reconstruction')
	
	parser.add_argument("--nodelist", dest="nodelist", type=str, 
		help="nodelist to load.")
	parser.add_argument("--edgelist", dest="edgelist", type=str, 
		help="edgelist to load.")
	parser.add_argument("--embedding", dest="embedding", type=str, 
		help="neighbor to save.")
	parser.add_argument("--hierarchyfile", dest="test", type=str, 
		help="save name.")
	parser.add_argument("--directed", action="store_true",
		help="whether the input is directed")
	
	
	return parser.parse_args()
def minkowki_dot(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	#print(u.shape, rank)
	euc_dp = u[:, :rank].dot(v[:, :rank].T)
	#print(u, v, euc_dp)
	return euc_dp - u[:, rank, None] * v[:, rank].T

def hyperbolic_distance_hyperboloid(u, v):
	mink_dp = minkowki_dot(u, v)
	mink_dp = np.maximum(-1 - mink_dp, 1e-15)
	return np.arccosh(1 + mink_dp)
def hyperbolic_setup(fig, ax):
    fig.set_size_inches(10.0, 10.0, forward=True)

    # set axes
    ax.set_ylim([-1.2, 1.2])
    ax.set_xlim([-1.2, 1.2])

    # draw Poincare disk boundary
    e = patches.Arc((0,0), 2.0, 2.0,
                     linewidth=2, fill=False, zorder=2)
    ax.add_patch(e)

colors = np.array(["b", "g", "r", "c", "m", "y", "k", "w"])
def draw_graph(graph, embedding, labels, path, s=25):
	assert embedding.shape[1] == 2 

	edges = list(graph.edges())

	idx = np.arange(len(embedding))

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges)

	print ("saving two-dimensional poincare plot to {}".format(path))

	fig = plt.figure()
	title = "Two dimensional poincare plot"
	plt.suptitle(title)
	ax = fig.add_subplot(111)
	hyperbolic_setup(fig, ax)

	pos = {n: emb for n, emb in zip(sorted(graph.nodes()), embedding)}
	node_colours = None

	node_sizes = np.array([graph.degree(n, weight="weight") for n in graph.nodes()])

	node_sizes = node_sizes / node_sizes.max() * 250
	#print(pos)
	nx.draw_networkx_nodes(graph, pos=pos, node_size=node_sizes)
	nx.draw_networkx_edges(graph, pos=pos, width=.05, node_size=node_sizes)
	nx.draw_networkx_labels(graph, pos, labels,font_size=5)
	plt.show()
	#plt.savefig(path)
	#plt.close()
def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperbolic_distance_poincare(X):
	norm_X = np.linalg.norm(X, keepdims=True, axis=-1)
	norm_X = np.minimum(norm_X, np.nextafter(1,0, ))
	uu = euclidean_distances(X) ** 2
	dd = (1 - norm_X**2) * (1 - norm_X**2).T
	return np.arccosh(1 + 2 * uu / dd)

def main():
	args = parse_args()
	V={}
	l=[]
	V_list=[]

	for name in open(args.nodelist):
		name=name.strip('\n')
		l.append(name)
		V[name]=len(V)

	adj=defaultdict(set)
	positive_pairs=np.loadtxt(args.edgelist, delimiter='\t', dtype='i4')
	for i in range(positive_pairs.shape[0]):
		#print(positive_pairs[i])
		adj[l[positive_pairs[i, 0]]].add(l[positive_pairs[i, 1]])
		if not args.directed:
			adj[l[positive_pairs[i, 1]]].add(l[positive_pairs[i, 0]])
	print(positive_pairs.shape)
	
	# adj=defaultdict(set)
	# for line in open(args.edgelist):
	# 	line=line.strip('\n').split('\t')
	# 	h1_index=int(line[0])
	# 	h1=l[h1_index]
	# 	h2_index=int(line[1])
	# 	h2=l[h2_index]

	# 	adj[h1].add(h2)
	# 	if not args.directed:
	# 		adj[h2].add(h1)
	
	norm=[]

	EmbeddingFile=args.embedding
	cnt=0



	node_coors=[]
	for line in open(EmbeddingFile):
		if cnt>0:
			line=line.strip('\n').split(',')
			node_coors.append([float(a) for a in line[1:]])
		cnt+=1
	node_coors=np.array(node_coors)


	dists=hyperbolic_distance_hyperboloid(node_coors, node_coors)
	print('Distance Computed')
	norm=[]

	A=hyperboloid_to_poincare_ball(node_coors)
	for i in range(A.shape[0]):
		norm.append(np.linalg.norm(np.array(A[i]), 2))


	########save

	test=args.test
	t=test.split('/')
	filename=t[-1]
	folder=t[:-1]
	#if folder:
	path=os.path.join(*folder)
	if not os.path.exists(path):
		os.makedirs(path)
	file1 = open(test,"w")
	print('save to', test)
	file1.write('This is the hierarchical structure of the embedding. There are {} hashtags. '.format(len(V)))
	file1.write('For every hashtag, we compute the hyperbolic distances(d) as well as the differences of norms(delta) between it to its neighbors.\n\n')

	for word in l:

		word_id=V[word]
		parents=[]
		sons=[]
		same=[]
		for neighbor in adj[word]:
			
			neighbor_id=V[neighbor]
			if norm[neighbor_id]>norm[word_id]+0.01: #####sons
				sons.append((neighbor, dists[neighbor_id, word_id], norm[neighbor_id]-norm[word_id]))
			elif norm[neighbor_id]<norm[word_id]-0.01:
				parents.append((neighbor, dists[neighbor_id, word_id], norm[word_id]-norm[neighbor_id]))
			else:
				same.append((neighbor, dists[neighbor_id, word_id], norm[neighbor_id]-norm[word_id]))

		sons.sort(key=lambda x: x[1])
		parents.sort(key=lambda x: x[1])
		same.sort(key=lambda x: x[1])
		file1.write(word.split('.')[0]+' norm: '+str(norm[word_id])+'\n')
		file1.write("There are "+str(len(sons))+ ' hashtags more niche than '+word.split('.')[0]+': ')
		for a,b, c in sons:
			file1.write('('+a.split('.')[0]+', d: '+str(b)+', '+'delta: '+str(c)+') ')
		file1.write('\n---\n')
		file1.write("There are "+str(len(parents))+" hashtags more general than "+word.split('.')[0]+": ")
		for a,b, c in parents:
			file1.write('('+a.split('.')[0]+', d: '+str(b)+', '+'delta: '+str(c)+') ')
		file1.write('\n---\n')
		file1.write(str(len(same))+' hashtags parallel with '+word.split('.')[0]+": ")
		for a,b, c in same:
			file1.write('('+a.split('.')[0]+', d: '+str(b)+', '+'delta: '+str(c)+') ')
		file1.write('\n\n')
		file1.write('--------------------------------------------------------------\n')
	print('done', args.test)


	# if node_coors.shape[1]==3:
	# 	print ("projecting to poincare ball")
	# 	embedding = hyperboloid_to_poincare_ball(node_coors)
	# 	graph = nx.read_weighted_edgelist(args.nodelist, delimiter="\t", nodetype=int,
	# 		create_using=nx.Graph())
	# 	#print(graph.nodes)
	# 	V_dic={}
	# 	for i, a in enumerate(l):
	# 		V_dic[i]=a
	# 	draw_graph(graph,#undirected_edges if not args.directed else directed_edges, 
	# 			embedding, V_dic, path="2d-poincare-disk-"+args.test+".png")


if __name__ == "__main__":
	main()