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

    st.title("kGraph on {}".format(dataset))
    
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
        st.markdown("Time series grouped based on the true labels (see [UCR-Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) for more details. Only 50 first time series are displayed.)")
        st.plotly_chart(fig_ts, use_container_width=True,height=800)
        st.markdown("Time series grouped based on the clustering labels of $k$-Graph. You can check the graph on the graph tab for more details. Only 50 first time series are displayed.")
        st.plotly_chart(fig_pred, use_container_width=True,height=800)
    
    with tab_graph:
        fig_graph,node_label = create_graph(graph)
        #st.plotly_chart(fig_graph, use_container_width=True,height=800)
        selected_node = plotly_events(fig_graph,override_height=800)
    
        if len(selected_node)>0:
            with st.container(border=True):
                node_label = node_label[selected_node[0]['pointIndex']]
                fig_ts,nb_subseq = get_node_ts(graph,X,node_label,length)
                st.markdown("Selected node is {} ({} subsequences)".format(node_label,nb_subseq))
                st.plotly_chart(fig_ts, use_container_width=True)

    with tab_detail:
        st.markdown("show detail")
        


if __name__ == "__main__":
    run()
