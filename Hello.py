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
import time
import os

LOGGER = get_logger(__name__)

st.set_page_config(
        page_title="Graphit",
        page_icon="👋",
    )


image = Image.open('figures/graphit_logo.png')
st.image(image,width=300)
st.markdown("## Welcome to Graphit")


st.markdown("""
 Time series clustering poses a significant challenge with diverse applications across domains. 
 A prominent drawback of existing solutions lies in their limited interpretability, often confined 
 to presenting users with centroids. In addressing this gap, our work presents in this demo $k$-Graph, an 
 unsupervised method explicitly crafted to augment interpretability in time series clustering.
 """)

st.markdown("### Contributors")
st.markdown("""
- [Paul Boniol](https://boniolp.github.io/), Inria, DI ENS, ENS, PSL University, CNRS
- [Donato Tiano](https://it.linkedin.com/in/donatotiano/en), Università degli Studi di Modena e Reggio Emilia
- [Angela Bonifati](https://perso.liris.cnrs.fr/angela.bonifati/), Lyon 1 University, IUF, Liris CNRS
- [Themis Palpanas](https://helios2.mi.parisdescartes.fr/~themisp/). Université Paris Cité, IUF
""")
st.warning("The graph rendering can be slow. We suggest to clone the [repo](https://github.com/boniolp/graphit) and run the app locally for faster interactions.", icon=None)
