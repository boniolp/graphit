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
    
    graph,X,y = read_dataset(dataset)
    fig_graph = create_graph(graph)
    st.plotly_chart(fig_graph, use_container_width=True,height=800)
    selected_node = plotly_events(fig_graph)

    st.markdown("Selected node is {}".format(selected_node))


if __name__ == "__main__":
    run()
