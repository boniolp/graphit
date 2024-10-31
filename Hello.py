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
from PIL import Image
from sklearn.metrics import adjusted_rand_score
import random

LOGGER = get_logger(__name__)



def run():
    st.set_page_config(
        page_title="Graphit",
        page_icon="👋",
    )

    
    with st.sidebar:
        dataset = st.selectbox('Pick a dataset', List_datasets)
        with st.expander("About"):
             st.markdown("""
             Time series clustering poses a significant challenge with diverse applications across domains. 
             A prominent drawback of existing solutions lies in their limited interpretability, often confined 
             to presenting users with centroids. In addressing this gap, our work presents in this demo $k$-Graph, an 
             unsupervised method explicitly crafted to augment interpretability in time series clustering.
             """)

        with st.expander("Contributors"):
            st.markdown("""
            - [Paul Boniol](https://boniolp.github.io/), Inria, DI ENS, ENS, PSL University, CNRS
            - [Donato Tiano](https://it.linkedin.com/in/donatotiano/en), Università degli Studi di Modena e Reggio Emilia
            - [Angela Bonifati](https://perso.liris.cnrs.fr/angela.bonifati/), Lyon 1 University, IUF, Liris CNRS
            - [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/). Université Paris Cité, IUF
            """)
        st.warning("The graph rendering can be slow. We suggest to clone the [repo](https://github.com/boniolp/graphit) and run the app locally for faster interactions.", icon=None)

    st.title("$k$-Graph on {}".format(dataset))
    
    graph,pos,X,y,length,y_pred_kshape,y_pred_kmean,all_graphoid_ex,all_graphoid_rep = read_dataset(dataset)
        
    tab_ts,tab_graph,tab_quiz,tab_detail = st.tabs(["Benchmarking", "k-Graph in action", "Quiz Time!", "Under the hood"])

    with tab_ts:
        st.header("Compare $k$-Graph with $k$-Shape, and $k$-Means")
        st.divider()
        fig_ts,fig_pred,fig_pred_kshape,fig_pred_kmean = show_ts(X,y,graph['kgraph_labels'],y_pred_kshape,y_pred_kmean)
        st.markdown("""You can compare below the clustering performances (using the [Adjusted Rand Index]()) of our porposed approach 
        $k$-Graph, with the state-of-the-art clustering algorithm $k$-Shape, and the usual baseline $k$-Means.""")
        
        with st.expander("$k$-Graph, ARI: {:.3f}".format(adjusted_rand_score(y,graph['kgraph_labels'])),expanded=True):
            st.markdown("""Time series grouped based on the clustering labels of $k$-Graph. You can check 
            the graph on the graph tab for more details. Only 50 first time series are displayed.""")
            st.plotly_chart(fig_pred, use_container_width=True,height=800)

        with st.expander("$k$-Shape, ARI: {:.3f}".format(adjusted_rand_score(y,y_pred_kshape)),expanded=True):
            st.markdown("""Time series grouped based on the clustering labels of $k$-Shape. Only 50 first time series are displayed.""")
            st.plotly_chart(fig_pred_kshape, use_container_width=True,height=800)

        with st.expander("$k$-Means, ARI: {:.3f}".format(adjusted_rand_score(y,y_pred_kmean)),expanded=True):
            st.markdown("""Time series grouped based on the clustering labels of $k$-Means. Only 50 first time series are displayed.""")
            st.plotly_chart(fig_pred_kmean, use_container_width=True,height=800)

        st.markdown("""You can find below the time series grouped using the true labels.""")
        
        with st.expander("Time series dataset (true labels)"):
            st.markdown("""Time series grouped based on the true labels 
            (see [UCR-Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) for more details. 
            Only 50 first time series are displayed.)""")
        
            st.plotly_chart(fig_ts, use_container_width=True,height=800)
        
        st.header("Overall benchmark with 14 baselines")
        st.divider()
        st.markdown("""We conducted an experimental evaluation utilizing real datasets from the UCR-Archive to assess the clustering performance of various methods.""")
        col_metric,col_type,col_param = st.columns([0.15,0.5,0.35])

        method_names = ['kGraph', 'kShape','SPF','BIRCH','Gauss.-Mixt.','MB-KMeans', 'kMeans', 'Time2Feat', 'DTC','OPTICS', 'Reservoir',
            'Agglomerative', 'HDBSCAN','MeanShift','DBSCAN']
        
        with col_metric:
            metric_name = st.selectbox('Accuracy measure', ['ARI','RI','AMI','NMI'])
        with col_type:
            type_dataset = st.multiselect(
                "Dataset types",
                ['AUDIO','DEVICE','ECG','EOG','EPG',
                 'HEMODYNAMICS','IMAGE','MOTION','OTHER','SENSOR',
                 'SIMULATED','SOUND','SPECTRO','TRAFFIC'],
                ['AUDIO','DEVICE','ECG','EOG','EPG',
                 'HEMODYNAMICS','IMAGE','MOTION','OTHER','SENSOR',
                 'SIMULATED','SOUND','SPECTRO','TRAFFIC'],
            )
        with col_param:
            ts_length = st.slider("Time series length", 15, 3000, (15, 3000))
            nb_class = st.slider("Number of clusters", 2, 60, (2, 60))
            nb_time_ts = st.slider("Numberd of time series", 40, 16637, (40, 16637))

        df_performance = pd.read_csv('data/results/{}.csv'.format(metric_name))
        
        list_true_false = [entry in type_dataset for entry in df_performance['Type']]
        df_performance = df_performance.loc[list_true_false]

        df_performance = df_performance.loc[df_performance['Length'] >= ts_length[0]]
        df_performance = df_performance.loc[df_performance['Length'] <= ts_length[1]]

        df_performance = df_performance.loc[df_performance['No. of Classes'] >= nb_class[0]]
        df_performance = df_performance.loc[df_performance['No. of Classes'] <= nb_class[1]]

        df_performance = df_performance.loc[df_performance['Test Size'] + df_performance['Train Size'] >= nb_time_ts[0]]
        df_performance = df_performance.loc[df_performance['Test Size'] + df_performance['Train Size'] <= nb_time_ts[1]]
        
        on = st.toggle("Box plot")

        if on:
            fig = px.box(df_performance[method_names])
            st.plotly_chart(fig, use_container_width=True,boxmean=True)
        else:
            st.dataframe(df_performance[method_names].style.highlight_max(axis=1))
            
    with tab_graph:

        col_graph,col_side = st.columns([0.7,0.3])

        with col_side:
            with st.expander("Advanced settings"):
                lambda_val = st.slider('Lambda (Representativity)', 0.0, 1.0, 0.5)
                gamma_val = st.slider('Gamma (Exclusivity)', 0.0, 1.0, 0.7)
                options = st.multiselect(
                    'Show graphoids for',
                    ['Cluster {}'.format(i) for i in set(graph['kgraph_labels'])],
                    ['Cluster {}'.format(i) for i in set(graph['kgraph_labels'])])
        
        with col_side:
            with st.container(border=True):
                selected_node = st.selectbox('Select a node',graph['graph']['dict_node'].keys())
                if selected_node is not None:
                    fig_ts,fig_hist,nb_subseq,intervals = get_node_ts(graph,X,selected_node,length)
                    st.markdown("Selected node is {} ({} subsequences)".format(selected_node,nb_subseq))
                    st.plotly_chart(fig_ts, use_container_width=True)
                    st.markdown("Clusters proportion within node {}".format(selected_node))
                    st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_graph:
            fig_graph,node_label = create_graph(graph['graph'],pos,graph['kgraph_labels'],graph['feature'],all_graphoid_ex,all_graphoid_rep,lambda_val=lambda_val,gamma_val=gamma_val,list_clusters=[int(val.replace('Cluster ','')) for val in options])
            fig_graph.update_layout(
                height=800,
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                showlegend=False,
                hovermode='closest',
                #margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            
            st.plotly_chart(fig_graph, use_container_width=True,height=800)
            if selected_node is not None:
                fig_pred_node = show_ts_node(X,y,graph['kgraph_labels'],intervals)
                st.markdown("""Node {} in the dataset""".format(selected_node))
                st.plotly_chart(fig_pred_node, use_container_width=True,height=800)
            #st.markdown("You can click on a node to see its content")
            #with st.container(border=True):
            #selected_node = plotly_events(fig_graph,click_event=True, override_height=800, override_width='100%')
            #st.markdown("You can click on a node to see its content")

    with tab_quiz:
        st.markdown("""Will you be able to find the good cluster?""")
        st.markdown("""Here is a time series rondomly selected from the dataset of your choice (tab on the left). Which cluster does the time series belong to?""")
        st.markdown("""Which cluster does the time series belong to?""")

        scorecard_placeholder = st.empty()
        list_question = []
        for i in range(10):
            rand_ts = random.randint(0, len(X)-1)
            quiz_set = {
                "question_number":i,
                "ts": X[rand_ts],
                "options": ["Cluster {}".format(j) for j in list(set(y))],
                "correct_answer": "Cluster {}".format(y[rand_ts])
            }
            list_question.append(quiz_set)
        
        # Activate Session States
        ss = st.session_state
        # Initializing Session States
        if 'counter' not in ss:
            ss['counter'] = 0
        if 'start' not in ss:
            ss['start'] = False
        if 'stop' not in ss:
            ss['stop'] = False
        if 'refresh' not in ss:
            ss['refresh'] = False
        if "button_label" not in ss:
            ss['button_label'] = ['START', 'SUBMIT', 'RELOAD']
        if 'current_quiz' not in ss:
            ss['current_quiz'] = []
        if 'user_answers' not in ss:
            ss['user_answers'] = []
        if 'grade' not in ss:
            ss['grade'] = 0

        def btn_click():
            ss.counter += 1
            if ss.counter > 2: 
                ss.counter = 0
                ss.clear()
            else:
                update_session_state()
                with st.spinner("*this may take a while*"):
                    time.sleep(2)
        
        def update_session_state():
            if ss.counter == 1:
                ss['start'] = True
                ss.current_quiz = list_question
            elif ss.counter == 2:
                # Set start to False
                ss['start'] = True 
                # Set stop to True
                ss['stop'] = True
        
        st.button(label=ss.button_label[ss.counter], 
            key='button_press', on_click= btn_click)
        with st.container():
            if (ss.start):
                for i in range(len(ss.current_quiz)):
                    number_placeholder = st.empty()
                    question_placeholder = st.empty()
                    options_placeholder = st.empty()
                    results_placeholder = st.empty()
                    expander_area = st.empty()                
                    # Add '1' to current_question tracking variable cause python starts counting from 0
                    current_question = i+1
                    # display question_number
                    number_placeholder.write(f"*Question {current_question}*")
                    # display question based on question_number
                    with question_placeholder.container():
                        fig = px.line(ss.current_quiz[i].get('ts'))
                        st.plotly_chart(fig)
                    # question_placeholder.write(f"**{ss.current_quiz[i].get('question')}**") 
                    # list of options
                    options = ss.current_quiz[i].get("options")
                    # track the user selection
                    options_placeholder.radio("", options, index=1, key=f"Q{current_question}")
                    nl(1)
                    # Grade Answers and Return Corrections
                    if ss.stop:
                        # Track length of user_answers
                        if len(ss.user_answers) < 10: 
                            # comparing answers to track score
                            if ss[f'Q{current_question}'] == ss.current_quiz[i].get("correct_answer"):
                                ss.user_answers.append(True)
                            else:
                                ss.user_answers.append(False)
                        else:
                            pass
                        # Results Feedback
                        if ss.user_answers[i] == True:
                            results_placeholder.success("CORRECT")
                        else:
                            results_placeholder.error("INCORRECT")
    
        # calculate score
        if ss.stop:  
            ss['grade'] = ss.user_answers.count(True)           
            scorecard_placeholder.write(f"### **Your Final Score : {ss['grade']} / {len(ss.current_quiz)}**")        
        
    
    with tab_detail:
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
            #TODO
            
        


if __name__ == "__main__":
    run()
