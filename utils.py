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

import inspect
import textwrap
import pickle
import streamlit as st
import plotly
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import networkx as nx
from plotly.subplots import make_subplots
from sklearn.metrics import adjusted_rand_score
import plotly.express as px

List_datasets = ['TwoLeadECG','CBF','Trace','DodgerLoopWeekend',
            'Haptics','SyntheticControl','Worms','Computers','HouseTwenty',
            'GestureMidAirD3', 'Chinatown', 'UWaveGestureLibraryAll', 'Strawberry', 
            'Car', 'GunPointAgeSpan', 'GestureMidAirD2', 'BeetleFly', 'Wafer',
            'Adiac', 'ItalyPowerDemand', 'Yoga', 'AllGestureWiimoteY']

cols = plotly.colors.DEFAULT_PLOTLY_COLORS

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def read_dataset(dataset):
    with open('data/graphs/{}.pickle'.format(dataset),'rb') as handle:
        graph = pickle.load(handle)

    G_nx = nx.DiGraph(graph['graph']['list_edge'])
    pos = nx.nx_agraph.graphviz_layout(G_nx,prog="fdp")

    all_graphoid_ex,all_graphoid_rep = [],[]
    for cluster in set(graph['kgraph_labels']):
        data = []
        for i in range(len(graph['kgraph_labels'])):
            if cluster == graph['kgraph_labels'][i]:
                data.append(list(graph['feature'].values[i]))
        representative_graphoid = np.count_nonzero(data, axis=0)
        all_graphoid_rep.append(representative_graphoid)

    all_graphoid_ex = np.array(all_graphoid_rep)/np.sum(np.array(all_graphoid_rep),0)
    all_graphoid_rep = (np.array(all_graphoid_rep).T/np.array([list(graph['kgraph_labels']).count(i) for i in set(graph['kgraph_labels'])])).T
    
    with open('data/baselines/{}_kshape.pickle'.format(dataset),'rb') as handle:
        y_pred_kshape = pickle.load(handle)

    with open('data/baselines/{}_kmean.pickle'.format(dataset),'rb') as handle:
        y_pred_kmean = pickle.load(handle)

    path = 'data/timeseries/{}/'.format(dataset)
    train_data = pd.read_csv(path + "{}_TRAIN.tsv".format(dataset),sep='\t',header=None)
    target_train = np.array(train_data[0].values)
    train_data = train_data.drop(0,axis=1)
    train_data = train_data.fillna(0)
    data_train = np.array(train_data.values)
    data_train = (data_train - np.mean(data_train,axis=1,keepdims=True))/(np.std(data_train,axis=1,keepdims=True))

    test_data = pd.read_csv(path + "{}_TEST.tsv".format(dataset),sep='\t',header=None)
    target_test = np.array(test_data[0].values)
    test_data = test_data.drop(0,axis=1)
    test_data = test_data.fillna(0)
    data_test = np.array(test_data.values)
    data_test = (data_test - np.mean(data_test,axis=1,keepdims=True))/(np.std(data_test,axis=1,keepdims=True))
    X = np.concatenate([data_train,data_test],axis=0)
    y = np.concatenate([target_train,target_test],axis=0)
    
    length = int(graph['length'])
    return graph,pos,X,y,length,y_pred_kshape,y_pred_kmean,all_graphoid_ex,all_graphoid_rep

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def create_graph(graph,pos,labels,features,all_graphoid_ex,all_graphoid_rep,lambda_val=0.5,gamma_val=0.5,list_clusters=[0,1,2,3,4]):
    G_nx = nx.DiGraph(graph['list_edge'])

    features_name = list(features.columns)
    
    edge_size_0 = [] 
    for edge in G_nx.edges():
        edge_size_0.append(graph['list_edge'].count([edge[0],edge[1]]))
    edge_size_b = [float(1+(e - min(edge_size_0)))/float(1+max(edge_size_0) - min(edge_size_0)) for e in edge_size_0]
    edge_size_0 = [min(e*20,10) for e in edge_size_b]
    dict_node_0 = []
    for node in G_nx.nodes():
        if node != "NULL_NODE":
           dict_node_0.append(max(5,graph['dict_node'][node]*0.01))
        else:
           dict_node_0.append(5)
   
    

    list_edge_trace = []
    for i,edge in enumerate(G_nx.edges()):
        pos_in_feature = features_name.index("['{}', '{}']".format(edge[0],edge[1]))
        cluster_max = np.argmax(all_graphoid_ex[:,pos_in_feature])
        cluster_max_val = max(all_graphoid_ex[:,pos_in_feature])
        cluster_max_rep = np.argmax(all_graphoid_rep[:,pos_in_feature])
        cluster_max_val_rep = max(all_graphoid_rep[:,pos_in_feature])
        if cluster_max in list_clusters:
            if (cluster_max_val > gamma_val) and (cluster_max_val_rep > lambda_val):
                color_edge = (cols[cluster_max][:-1]+",1)").replace('rgb','rgba')
            else:
                color_edge = 'rgba(211, 211, 211,0.5)'
            edge_trace = go.Scattergl(
                x=[pos[edge[0]][0],pos[edge[1]][0]], y=[pos[edge[0]][1],pos[edge[1]][1]],
                line=dict(width=edge_size_0[i], color=color_edge),
                hoverinfo='none',
                mode='lines')
            list_edge_trace.append(edge_trace)

    node_x = []
    node_y = []
    node_text = []
    color_node = []
    for i,node in enumerate(G_nx.nodes()):
        pos_in_feature = features_name.index(node)
        cluster_max = np.argmax(all_graphoid_ex[:,pos_in_feature])
        cluster_max_val = max(all_graphoid_ex[:,pos_in_feature])
        cluster_max_rep = np.argmax(all_graphoid_rep[:,pos_in_feature])
        cluster_max_val_rep = max(all_graphoid_rep[:,pos_in_feature])
        if cluster_max in list_clusters:
            if (cluster_max_val > gamma_val) and (cluster_max_val_rep > lambda_val):
                color_node.append((cols[cluster_max][:-1]+",1)").replace('rgb','rgba'))
                dict_node_0[i] = dict_node_0[i]*1.2
            else:
                color_node.append('rgba(211, 211, 211,0.2)')
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=color_node,
            line_color=color_node,
            size=dict_node_0,
            line_width=1))
    fig = go.Figure(data=list_edge_trace + [node_trace])
    return fig,node_text


@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def show_length_plot(graph):
    
    fig = make_subplots(rows=1, cols=3,subplot_titles=["Length relevance","Consistency", "Interpretability factor"])
    all_length = graph['length_relevance'][:,0]
    length_relevance = graph['relevance'][:,1]
    length_consistency = graph['length_relevance'][:,1]
    length_IF = graph['graph_relevance'][:,1]
    
    fig.add_trace(
        go.Scatter(x=all_length, y=length_relevance),
        row=1, col=1
    )
    fig.add_vline(x=graph['length'], line_dash="dot",line_color='red', row=1, col=1,
        annotation_text="optimal length", 
        annotation_position="bottom left",
        annotation_textangle=90)
    fig.add_trace(
        go.Scatter(x=all_length, y=length_consistency),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=all_length, y=length_IF),
        row=1, col=3
    )
    fig.update_layout(height=500,showlegend=False)

    fig_feat = px.imshow(graph['feature'][:200], color_continuous_scale='RdBu_r', origin='lower')
    
    return fig,fig_feat

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def show_ts_node(X,y,y_pred_kgraph,intervals):
    
    fig_pred = make_subplots(rows=1, cols=len(set(y)),subplot_titles=["Cluster {}".format(i) for i in range(len(set(y)))])
    x_list = list(range(len(X[0])))
    labels = {lab:i for i,lab in enumerate(set(y))}
    labels_pred = {lab:i for i,lab in enumerate(set(y_pred_kgraph))}
    for x,lab,pred in zip(X[:50],y[:50],y_pred_kgraph[:50]):
        fig_pred.add_trace(
            go.Scattergl(x=x_list, y=x, mode='lines', line_color="grey",opacity=0.1),
            row=1, col=labels_pred[pred]+1
        )
    for interval in intervals[:50]:
        x = X[interval[0]]
        lab = y[interval[0]]
        pred = y_pred_kgraph[interval[0]]
        fig_pred.add_trace(
            go.Scattergl(x=x_list[interval[1]:interval[2]], y=x[interval[1]:interval[2]], mode='lines',  line_width=5, line_color=(cols[labels[lab]][:-1]+",0.5)").replace("rgb","rgba")),
            row=1, col=labels_pred[pred]+1
        )
    fig_pred.update_layout(height=400,showlegend=False)

    
    return fig_pred



@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def show_ts(X,y,y_pred_kgraph,y_pred_kshape,y_pred_kmean):
    trace_ts = []
    fig = make_subplots(rows=1, cols=len(set(y)),subplot_titles=["Cluster {}".format(i) for i in range(len(set(y)))])
    x_list = list(range(len(X[0])))
    labels = {lab:i for i,lab in enumerate(set(y))}
    for x,lab in zip(X[:50],y[:50]):
        fig.add_trace(
            go.Scattergl(x=x_list, y=x, mode='lines', line_color=(cols[labels[lab]][:-1]+",0.5)").replace("rgb","rgba")),
            row=1, col=labels[lab]+1
        )
    fig.update_layout(height=400)

    
    fig_pred = make_subplots(rows=1, cols=len(set(y)),subplot_titles=["Cluster {}".format(i) for i in range(len(set(y)))])
    x_list = list(range(len(X[0])))
    labels_pred = {lab:i for i,lab in enumerate(set(y_pred_kgraph))}
    for x,lab,pred in zip(X[:50],y[:50],y_pred_kgraph[:50]):
        fig_pred.add_trace(
            go.Scattergl(x=x_list, y=x, mode='lines', line_color=(cols[labels[lab]][:-1]+",0.5)").replace("rgb","rgba")),
            row=1, col=labels_pred[pred]+1
        )
    fig_pred.update_layout(height=400,title="ARI: {}".format(adjusted_rand_score(y_pred_kgraph,y)))

    fig_pred_kshape = make_subplots(rows=1, cols=len(set(y)),subplot_titles=["Cluster {}".format(i) for i in range(len(set(y)))])
    x_list = list(range(len(X[0])))
    labels_pred = {lab:i for i,lab in enumerate(set(y_pred_kshape))}
    for x,lab,pred in zip(X[:50],y[:50],y_pred_kshape[:50]):
        fig_pred_kshape.add_trace(
            go.Scattergl(x=x_list, y=x, mode='lines', line_color=(cols[labels[lab]][:-1]+",0.5)").replace("rgb","rgba")),
            row=1, col=labels_pred[pred]+1
        )
    fig_pred_kshape.update_layout(height=400,title="ARI: {}".format(adjusted_rand_score(y_pred_kshape,y)))

    fig_pred_kmean = make_subplots(rows=1, cols=len(set(y)),subplot_titles=["Cluster {}".format(i) for i in range(len(set(y)))])
    x_list = list(range(len(X[0])))
    labels_pred = {lab:i for i,lab in enumerate(set(y_pred_kmean))}
    for x,lab,pred in zip(X[:50],y[:50],y_pred_kmean[:50]):
        fig_pred_kmean.add_trace(
            go.Scattergl(x=x_list, y=x, mode='lines', line_color=(cols[labels[lab]][:-1]+",0.5)").replace("rgb","rgba")),
            row=1, col=labels_pred[pred]+1
        )
    fig_pred_kmean.update_layout(height=400,title="ARI: {}".format(adjusted_rand_score(y_pred_kmean,y)))
    return fig,fig_pred,fig_pred_kshape,fig_pred_kmean

@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def compute_consensus(all_runs):
    all_mat = sum([create_membership_matrix(run[:100]) for run in all_runs])
    fig_feat = px.imshow(all_mat/all_mat.diagonal(), color_continuous_scale='RdBu_r', origin='lower')
    return fig_feat

def create_membership_matrix(run):
    mat = np.zeros((len(run),len(run)))
    for i,val_i in enumerate(run):
        for j,val_j in enumerate(run):
            if val_i == val_j:
                mat[i][j] = 1
                mat[j][i] = 1
    return mat


@st.cache_data(ttl=3600, max_entries=1, show_spinner=True)
def get_node_ts(graph,X,node,length):
    result = []
    current_pos = 0
    labels_node = []
    intervals = []
    ts_found = {"Cluster {}".format(j):0 for j in set(graph['kgraph_labels'])}
    edge_in_time = graph['graph']['edge_in_time']
    ts_found_tmp = {"Cluster {}".format(j):False for j in set(graph['kgraph_labels'])}
    for i,edge in enumerate(graph['graph']['list_edge']):
        if node == edge[0]:
            relative_pos = i-graph['graph']['list_edge_pos'][current_pos]
            pos_in_time = min(
                range(len(edge_in_time[current_pos])), 
                key=lambda j: abs(edge_in_time[current_pos][j]-relative_pos))
            ts = X[int(current_pos),int(pos_in_time):int(pos_in_time+length)]
            intervals.append([int(current_pos),int(pos_in_time),int(pos_in_time+length)])
            labels_node.append("Cluster {}".format(graph['kgraph_labels'][int(current_pos)]))
            ts_found_tmp[labels_node[-1]] = True
            ts = ts - np.mean(ts)
            result.append(ts)
        
        if i >= graph['graph']['list_edge_pos'][current_pos+1]:
            current_pos += 1
            for key in ts_found_tmp.keys():
                if ts_found_tmp[key]:
                    ts_found[key] += 1
            ts_found_tmp = {"Cluster {}".format(j):False for j in set(graph['kgraph_labels'])}
                    

    mean = np.mean(result,axis=0)
    dev = np.std(result,axis=0)

    mean_trace = go.Scatter(
            x=list(range(length)), y=mean,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
    lowerbound_trace = go.Scatter(
            x=list(range(length)), y=mean-dev,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')
    upperbound_trace = go.Scatter(
            x=list(range(length)), y=mean+dev,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            fill='tonexty',
            mode='lines')
    
    fig = go.Figure(data=[mean_trace,lowerbound_trace,upperbound_trace],
        layout=go.Layout(
            height=150,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )
    fig_hist = go.Figure(layout=go.Layout(height=300))
    fig_hist.add_trace(go.Bar(name='Exclusivity',x=["Cluster {}".format(i) for i in set(graph['kgraph_labels'])], y=[labels_node.count("Cluster {}".format(i))/len(labels_node) for i in set(graph['kgraph_labels'])]))
    fig_hist.add_trace(go.Bar(name='Representativity',x=["Cluster {}".format(i) for i in set(graph['kgraph_labels'])], y=[ts_found["Cluster {}".format(i)]/list(graph['kgraph_labels']).count(i) for i in set(graph['kgraph_labels'])]))
    fig_hist.update_layout(barmode='group',legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.3))
    #fig_hist.add_trace(go.Histogram(x=labels_node, name="number of subsequences", histnorm='percent', texttemplate="%{y}%", textfont_size=10))
    
    return fig,fig_hist,len(result),intervals

