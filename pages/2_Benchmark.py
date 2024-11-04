import streamlit as st
from streamlit.logger import get_logger
from utils import *
from streamlit_plotly_events import plotly_events
from PIL import Image
from sklearn.metrics import adjusted_rand_score
import random
import time
import os

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
    st.plotly_chart(fig, use_container_width=True,boxmean=True,key="boxplot")
else:
    st.dataframe(df_performance[method_names].style.highlight_max(axis=1))
