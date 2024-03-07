# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from utils import *
from streamlit_plotly_events import plotly_events

LOGGER = get_logger(__name__)



def run():
    st.set_page_config(
        page_title="Graphit",
        page_icon="👋",
    )

    
    with st.sidebar:
        dataset = st.selectbox('Pick a dataset', List_datasets)

    st.title("$k$-Graph on {}".format(dataset))
    
    graph,X,y,length = read_dataset(dataset)

    with st.sidebar:
        with st.expander("Advanced settings"):
            lambda_val = st.slider('Lambda', 0.0, 1.0, 0.5)
            gamma_val = st.slider('Gamma', 0.0, 1.0, 0.5)
            options = st.multiselect(
                'Show graphoids for',
                ['Cluster {}'.format(i) for i in set(y)],
                ['Cluster {}'.format(i) for i in set(y)])
        
    
    tab_ts,tab_graph,tab_detail = st.tabs(["Time series", "Graph", "Under the hood"])

    with tab_ts:
        fig_ts,fig_pred = show_ts(X,y,graph)
        st.header("Time series dataset (true labels)")
        
        st.markdown("""Time series grouped based on the true labels 
        (see [UCR-Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) for more details. 
        Only 50 first time series are displayed.)""")
        
        st.plotly_chart(fig_ts, use_container_width=True,height=800)
        st.header("Time series dataset (predicted labels of $k$-Graph)")
        
        st.markdown("""Time series grouped based on the clustering labels of $k$-Graph. You can check 
        the graph on the graph tab for more details. Only 50 first time series are displayed.""")
        
        st.plotly_chart(fig_pred, use_container_width=True,height=800)
    
    with tab_graph:
        fig_graph,node_label = create_graph(graph['graph'])
        #st.plotly_chart(fig_graph, use_container_width=True,height=800)
        selected_node = plotly_events(fig_graph,override_height=800)
        st.markdown("You can click on a node to see its content ({} node selected)".format(len(selected_node)))
    
        if len(selected_node)>0:
            with st.container(border=True):
                node_label = node_label[selected_node[0]['pointIndex']]
                fig_ts,fig_hist,nb_subseq = get_node_ts(graph,X,node_label,length)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Selected node is {} ({} subsequences)".format(node_label,nb_subseq))
                    st.plotly_chart(fig_ts, use_container_width=True)
                with col2:
                    st.markdown("Clusters proportion within node {}".format(node_label))
                    st.plotly_chart(fig_hist, use_container_width=True)

    with tab_detail:
        with st.expander("""## How the graph is built?"""):
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

        with st.expander("""## How the graph is used to cluster time series?"""):
            
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
            Below is the consensus matrix $M_C \in \mathbb{R}^{(|\mathcal{D}|,|\mathcal{D}|)}$ (only for the first 100 time series) for {}.
            """.format(dataset))

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
            st.markdown("Yes and no, It depends on the how precise or simple the interpretation needs to be.")
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
            #TODO
            
        


if __name__ == "__main__":
    run()
