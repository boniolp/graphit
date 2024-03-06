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

    st.markdown(
        """
        <style>
        [data-testid="stForm"]{
        border: 2px solid black;
        border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True
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
        
    
    fig_graph,node_label = create_graph(graph)
    #st.plotly_chart(fig_graph, use_container_width=True,height=800)
    selected_node = plotly_events(fig_graph,override_height=800)

    if len(selected_node)>0:
        with st.form("my_form"):
            node_label = node_label[selected_node[0]['pointIndex']]
            st.markdown("Selected node is {}".format(node_label))
            fig_ts = get_node_ts(graph,X,node_label,length)
            st.plotly_chart(fig_ts, use_container_width=True)

    


if __name__ == "__main__":
    run()
