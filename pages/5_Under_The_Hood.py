import streamlit as st
from streamlit.logger import get_logger
from utils import *
from streamlit_plotly_events import plotly_events
from PIL import Image
from sklearn.metrics import adjusted_rand_score
import random
import time
import os

dataset = st.selectbox('Pick a dataset', List_datasets)
graph,pos,X,y,length,y_pred_kshape,y_pred_kmean,all_graphoid_ex,all_graphoid_rep = read_dataset(dataset)


with st.expander("""## In short, how does $k$-graph work?"""):
	st.markdown("""
	$k$-Graph is a method for time series clustering. For a given time series dataset $\mathcal{D}$, 
	the overall $k$-Graph process is divided into three main steps as follows:
	1. **Graph Embedding**: for $M$ different subsequence lengths, we compute $M$ graphs.
	For a given subsequence length $\ell$, The set of nodes represent groups of similar subsequences 
	of length $\ell$ within the dataset $\mathcal{D}$. The edges have weights corresponding to the number of 
	times one subsequence of a given node has been followed by a subsequence of the other node.
	2. **Graph Clustering**: For each graph, we extract a set of features for all time series of the dataset $\mathcal{D}$. 
	These features correspond to the nodes and edges that the time series crossed. Then, we use these features to cluster 
	the time series using the $k$-Mean algorithm for scalability reasons.
	3. **Consensus Clustering**: At this point we have $M$ clustering partitions (i.e., one per graph). We build a consensus matrix $M_C$. 
	We then cluster this matrix using spectral clustering in the objective of grouping time series that are highly connected 
	(i.e., grouped in the same cluster in most of the $M$ partitions). The output of this clustering step is the labels provided by $k$-Graph.
	4. **Interpretability Computation**: after obtaining the clustering partition, we select the most relevant graph (among the $M$ graphs), 
	and we compute the interpretable graphoids. This is what you can see in the graph tab.
	""")
	image = Image.open('data/figures/pipeline.png')
	st.image(image, caption='Overview of kgraph pipeline')
	st.markdown("""
	You may find more details in our [paper]().
	""")
with st.expander("""## How is the graph built?"""):
	st.markdown("""The graph corresponds to a summary of all the subsequences present in the datasets. 
	In theory, the objective is to transform a time series 
	dataset into a sequence of abstract states corresponding to different subsequences 
	within the dataset. These states are represented as nodes, denoted by $\mathcal{N}$, 
	in a directed graph, $\mathcal{G}=(\mathcal{N},\mathcal{E})$. The edges, $\mathcal{E}$, 
	encode the frequency with which one state occurs after another.""")
	st.markdown("""
	In practice, we build the graph as follows:
	""")
	st.markdown("""
	1. **Subsequence Embedding**: For each time series $T \in \mathcal{D}$, we collect all the subsequences 
	of a given length $\ell$ into an array called $Proj(T,\lambda)$. We then concatenate all the computed 
	$Proj(T,\lambda)$ into $Proj$ for all the time series in the dataset.
	We then sample $Proj$ (user-defined parameter $smpl$) and keep only a limited number of subsequences 
	stored in $Proj_{smpl}$. We use the latter to train a Principal Component Analysis (PCA). 
	We then use the trained PCA and a rotation step to project all the subsequences into a two-dimensional 
	space that only preserves the shapes of the subsequences. The result is denoted as $SProj$. 
	We denote the PCA and rotation steps $Reduce(Proj,pca)$, where $pca$ is the trained PCA.""")
	st.markdown("""
	2. **Node Creation**: Create a node for each of the densest parts of the above two-dimensional space. 
	In practice, we perform a radial scan of $SProj_{smpl}$.  %(using a fixed number of radius). 
	For each radius, we collect the intersection with the trajectories of $SProj_{smpl}$, and we apply kernel 
	density estimation on the intersected points: each local maximum of the density estimation curve is 
	assigned to a node. These nodes can be seen as a summarization of all the major patterns of length 
	$\ell$ that occurred in $\mathcal{D}$. For this step, we only consider the sampled collection of 
	subsequences $SProj_{smpl}$.""")
	st.markdown("""
	3. **Edge Creation**: Retrieve all transitions between pairs of subsequences represented by two different 
	nodes: each transition corresponds to a pair of subsequences, where one occurs immediately after the other 
	in a time series $T$ of the dataset $\mathcal{D}$. We represent transitions with an edge between the 
	corresponding nodes.""")

	st.markdown("""
	You may find more details in our [paper]().
	""")

with st.expander("""## Which subsequence length is used for the graph?"""):

	st.markdown("""$k$-Graph is computing $M$ different graphs for $M$ different 
	subsequence lengths. To maximize user interaction and interpretability, only 
	one graph is selected (the one you can see in the graph tab).""")

	st.markdown("""We select the graph using two criteria, the consistency 
	(ARI score for the labels obtained from each graph compared to the final labels 
	of $k$-Graph), and the interpretability factor.""")

	st.markdown("""The length relevance (first plot below) is the product of the two, 
	and the graph computed with the length maximizing this product is selected.""")

	fig_length,fig_feat = show_length_plot(graph)
	st.plotly_chart(fig_length, use_container_width=True,height=800)
	st.markdown("for {}, the optimal length selected is {}".format(dataset,length))

with st.expander("""## How is the graph used to cluster time series?"""):

	st.markdown("""
	To cluster the time series using the graph, we are extracting features. 
	The features corresponds to the number of time a node and an edge have 
	been crossed by one time series. We then use $k$-mean to cluster the 
	time series using the aforementioned extracted features.""")

	st.markdown("""
	The heatmap below show the feature matrix (one time series per row, 
	and one node or edge per column) for {} with the optimal subsequence length {}.
	""".format(dataset,length))

	st.plotly_chart(fig_feat, use_container_width=True,height=800)
with st.expander("""## Is only one graph used to cluster time series?"""):
	st.markdown("""No, we actually use all the graph to generate the final label fo $k$-Graph.
	In total, we have one clustering partition per graph ($M$ in total). We compute a consensus from all
	these partitions. In practice, we build a consensus matrix, which we employ to measure how many times 
	two time series have been grouped in the same cluster for two graphs built with two different lengths.
	Below is the consensus matrix $M_C$ (only for the first 100 time series) for {}.""".format(dataset))

	fig_cons = compute_consensus(graph['all_runs'])
	st.plotly_chart(fig_cons, use_container_width=True,height=800)

	st.markdown("""$M_C$ can be seen as a similarity matrix about the clustering results obtained on each graph. 
	More specifically, for two time series $T_i$ and $T_j$, if $M_C[i,j]$ is high, they have been associated in 
	the same cluster for several subsequence lengths and can be grouped in the same cluster. 
	On the contrary, if $M_C[i,j]$ is low, the two time series were usually grouped in different clusters regardless 
	of the subsequence length. Therefore, the $M_C$ matrix can be seen as the adjacency matrix of a graph. 
	In this graph, nodes are the time series of the dataset and an edge exists if two time series have been clustered together 
	in a same cluster (the weights of these edges are the number of time these two time series have been clustered together). 
	As the objective is to find communities of highly connected nodes (i.e., time series that were grouped multiple times in 
	the same cluster), we use spectral clustering (with $M_C$ used as a pre-computed similarity matrix). 
	The output of the spectral clustering is the final labels $\mathcal{L}$ of $k$-Graph.
	""")

	#TODO
with st.expander("""## Is one graph enough to interpret the clustering?"""):
	st.markdown("""
	Yes and no. It depends on how precise or simple the interpretation needs to be.
	The perfect case is when one graph is enough (i.e., strong consistency and high interpretability factor).
	However, it might be a combination of graphs (i.e., subsequence lengths) that contribute the most to the final $k$-Graph
	labels. In this case, we have two options: (i) we can return multiple graphs such that the consistency reaches a decent level, 
	(ii) or returning only one graph even though the returned graph alone might not be enough to distinguish clusters easily.
	The first option favors accuracy over user interaction, while the second option favors user interaction over accuracy.
	In this app, we chose the second option. Nevertheless, there are no best options, and it is an interesting research direction. 
	""")
	#TODO

with st.expander("""## How can I use $k$-Graph?"""):
	st.markdown("""
	Quite simple, you can install $k$-Graph with the following command:

	```(bash) 
	pip install kgraph-ts
	```

	You may find more details [here](https://github.com/boniolp/kGraph). Here is an example on how to use $k$-Graph:

	```python 
	import sys
	import pandas as pd
	import numpy as np
	import networkx as nx
	import matplotlib.pyplot as plt
	from sklearn.metrics import adjusted_rand_score

	sys.path.insert(1, './utils/')
	from utils import fetch_ucr_dataset

	from kgraph import kGraph


	path = "/Path/to/UCRArchive_2018/"
	data = fetch_ucr_dataset('Trace',path)
	X = np.concatenate([data['data_train'],data['data_test']],axis=0)
	y = np.concatenate([data['target_train'],data['target_test']],axis=0)


	# Executing kGraph
	clf = kGraph(n_clusters=len(set(y)),n_lengths=10,n_jobs=4)
	clf.fit(X)

	print("ARI score: ",adjusted_rand_score(clf.labels_,y))
	``` 
	```
	Running kGraph for the following length: [36, 72, 10, 45, 81, 18, 54, 90, 27, 63] 
	Graphs computation done! (36.71151804924011 s) 
	Consensus done! (0.03878021240234375 s) 
	Ensemble clustering done! (0.0060100555419921875 s) 
	ARI score:  0.986598879940902
	```
	  
	""")
