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
        st.markdown("Time series grouped based on the true labels (see [UCR-Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) for more details. Only 50 first time series are displayed.)")
        st.plotly_chart(fig_ts, use_container_width=True,height=800)
        st.header("Time series dataset (predicted labels of $k$-Graph)")
        st.markdown("Time series grouped based on the clustering labels of $k$-Graph. You can check the graph on the graph tab for more details. Only 50 first time series are displayed.")
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
        with st.expander("## Which subsequence length is used for the graph?"):
            st.markdown("$k$-Graph is computing $M$ different graphs for $M$ different subsequence lengths. To maximize user interaction and interpretability, only one graph is selected (the one you can see in the graph tab).")
            st.markdown("We select the graph using two criteria, the consistency (ARI score for the labels obtained from each graph compared to the final labels of $k$-Graph), and the interpretability factor.")
            st.markdown("The length relevance (first plot below) is the product of the two, and the graph computed with the length maximizing this product is selected.")
    
            fig_length,fig_feat = show_length_plot(graph)
            st.plotly_chart(fig_length, use_container_width=True,height=800)
            st.markdown("for {}, the optimal length selected is {}".format(dataset,length))

        with st.expander("## How the graph is used to cluster time series?"):
            st.markdown("To cluster the time series using the graph, we are extracting features. The features corresponds to the number of time a node and an edge have been crossed by one time series. We then use $k$-mean to cluster the time series using the aforementioned extracted features.")
            st.markdown("The heatmap below show the feature matrix (one time series per row, and one node or edge per column) for {} with the optimal subsequence length {}.".format(dataset,length))
    
            st.plotly_chart(fig_feat, use_container_width=True,height=800)
        with st.expander("## Is only one graph used to cluster time series?"):
            st.markdown("No, we actually use all the graph to generate the final label fo $k$-Graph.")

        with st.expander("## Is one graph enough to interpret the clustering?"):
            st.markdown("Yes and no, It depends on the how precise or simple the interpretation needs to be.")
            
        


if __name__ == "__main__":
    run()
