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

st.title("$k$-Graph on {}".format(dataset))
    
graph,pos,X,y,length,y_pred_kshape,y_pred_kmean,all_graphoid_ex,all_graphoid_rep = read_dataset(dataset)
    


st.header("Compare $k$-Graph with $k$-Shape, and $k$-Means")
st.divider()
fig_ts,fig_pred,fig_pred_kshape,fig_pred_kmean = show_ts(X,y,graph['kgraph_labels'],y_pred_kshape,y_pred_kmean)
st.markdown("""You can compare below the clustering performances (using the [Adjusted Rand Index]()) of our porposed approach 
$k$-Graph, with the state-of-the-art clustering algorithm $k$-Shape, and the usual baseline $k$-Means.""")

with st.expander("$k$-Graph, ARI: {:.3f}".format(adjusted_rand_score(y,graph['kgraph_labels'])),expanded=True):
    st.markdown("""Time series grouped based on the clustering labels of $k$-Graph. You can check 
    the graph on the graph tab for more details. Only 50 first time series are displayed.""")
    st.plotly_chart(fig_pred, use_container_width=True,height=800,key="kgraph-ARI")

with st.expander("$k$-Shape, ARI: {:.3f}".format(adjusted_rand_score(y,y_pred_kshape)),expanded=True):
    st.markdown("""Time series grouped based on the clustering labels of $k$-Shape. Only 50 first time series are displayed.""")
    st.plotly_chart(fig_pred_kshape, use_container_width=True,height=800,key="kshape-ARI")

with st.expander("$k$-Means, ARI: {:.3f}".format(adjusted_rand_score(y,y_pred_kmean)),expanded=True):
    st.markdown("""Time series grouped based on the clustering labels of $k$-Means. Only 50 first time series are displayed.""")
    st.plotly_chart(fig_pred_kmean, use_container_width=True,height=800,key="kmean-ARI")

st.markdown("""You can find below the time series grouped using the true labels.""")

with st.expander("Time series dataset (true labels)"):
    st.markdown("""Time series grouped based on the true labels 
    (see [UCR-Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) for more details. 
    Only 50 first time series are displayed.)""")

    st.plotly_chart(fig_ts, use_container_width=True,height=800,key="true_label")
