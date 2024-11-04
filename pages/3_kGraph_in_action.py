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
        height=400,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=False,
        hovermode='closest',
        #margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    st.markdown("""### $k$-Graph for {}""".format(dataset))
    st.plotly_chart(fig_graph, use_container_width=True,height=400)
    if selected_node is not None:
        fig_pred_node = show_ts_node(X,y,graph['kgraph_labels'],intervals)
        fig_pred_node.update_layout(
            height=300,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            showlegend=False,
            hovermode='closest',)
        st.divider()
        st.markdown("""### Node {} in the dataset""".format(selected_node))
        st.plotly_chart(fig_pred_node, use_container_width=True,height=100,key="interactive-graph")
    #st.markdown("You can click on a node to see its content")
    #with st.container(border=True):
    #selected_node = plotly_events(fig_graph,click_event=True, override_height=800, override_width='100%')
    #st.markdown("You can click on a node to see its content")
