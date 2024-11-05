import streamlit as st
from streamlit.logger import get_logger
from utils import *
from streamlit_plotly_events import plotly_events
from PIL import Image
from sklearn.metrics import adjusted_rand_score
import random
import time
import os

# Activate Session States
ss = st.session_state

def nl(num_of_lines):
    for i in range(num_of_lines):
        st.write(" ")

def btn_click():
    ss.counter += 1
    if ss.counter > 2: 
        ss.counter = 0
        ss.clear()
    else:
        update_session_state()
        with st.spinner("*this may take a while*"):
            time.sleep(2)


def intitialize_session():
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
    

def intitialize_session_change():
    ss.clear()
    intitialize_session()

st.header("""Can you find the correct cluster?""")

st.markdown("""Here are time series randomly selected from the dataset of your choice. Which cluster does these time series belong to?""")


st.markdown("""Please note that we assess here the ability of the methods to let the user anderstand why the time seires belong to a given cluster, we do not evaluate the cludtering accuracy. If one time series is assigned to the correct cluster (even though it does not belongs to the correct class), the answer is marked as correct.""")


col_dts,col_method = st.columns(2)

with col_dts:
    dataset = st.selectbox('Pick a dataset', List_datasets, on_change=intitialize_session_change)
with col_method:
    method = st.selectbox('Clustering method', ['k-Means','k-Shapes','k-Graph'], on_change=intitialize_session_change)


graph,pos,X,y,length,y_pred_kshape,y_pred_kmean,all_graphoid_ex,all_graphoid_rep = read_dataset(dataset)
correspondance_dict = {'k-Means':'kmean','k-Shapes':'kshape','k-Graph':'kgraph'}

if (correspondance_dict[method] == 'kmean') or (correspondance_dict[method] == 'kshape'):
    with open('data/graphs/{}_{}_centroid.pickle'.format(dataset,correspondance_dict[method]), 'rb') as handle:
        centroids = pickle.load(handle)
scorecard_placeholder = st.empty()


list_question = []
for i in range(5):
    rand_ts = random.randint(0, len(X)-1)
    if correspondance_dict[method] == 'kmean':
        y_temp = y_pred_kmean
    elif correspondance_dict[method] == 'kshape':
        y_temp = y_pred_kshape
    else:
        y_temp = graph['kgraph_labels']
    quiz_set = {
        "question_number":i,
        "ts": X[rand_ts],
        "options": ["Cluster {}".format(j) for j in list(set(y_temp))],
        "correct_answer": "Cluster {}".format(y_temp[rand_ts]),
        "id_ts": rand_ts,
    }
    print("Cluster {}".format(y_temp[rand_ts]))
    list_question.append(quiz_set)

def update_session_state():
    if ss.counter == 1:
        ss['start'] = True
        ss.current_quiz = list_question
    elif ss.counter == 2:
        # Set start to False
        ss['start'] = True 
        # Set stop to True
        ss['stop'] = True
intitialize_session()
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
            #with st.expander(f"*Question {current_question}*"):
            number_placeholder.markdown("""### Question {}""".format(current_question))
            # display question based on question_number
            with question_placeholder.container():
                fig = px.line(ss.current_quiz[i].get('ts'))
                if correspondance_dict[method] == 'kgraph':
                    
                    col_graph_quiz, col_ts_quiz = st.columns(2)
                    with col_graph_quiz:
                        st.markdown('Subgraph of the time series')
                        start_edge_ts = graph['graph']['list_edge_pos'][ss.current_quiz[i].get('id_ts')]
                        end_edge_ts = graph['graph']['list_edge_pos'][ss.current_quiz[i].get('id_ts')+1]
                        list_edge_ts = graph['graph']['list_edge'][start_edge_ts:end_edge_ts]
                        #st.markdown(list_edge_ts[0])
                        #st.markdown(list_edge_ts[0][0])
                        #st.markdown(list_edge_ts[0][1])
                        fig_graph_quiz = create_subgraph(list_edge_ts,graph['graph'],pos,graph['kgraph_labels'],graph['feature'],all_graphoid_ex,all_graphoid_rep,lambda_val=0.5,gamma_val=0.7,list_clusters=[i for i in set(graph['kgraph_labels'])])
                        
                        min_pos_x = min([pos[key_node][0] for key_node in pos.keys()])
                        max_pos_y = max([pos[key_node][1] for key_node in pos.keys()])
                        min_pos_y = min([pos[key_node][1] for key_node in pos.keys()])

                        
                        for pos_text_incr,lab_c in enumerate(set(y_temp)):
                            fig_graph_quiz.add_annotation(
                                x=min_pos_x,
                                y=max_pos_y - (max_pos_y - min_pos_y)*(pos_text_incr/(len(set(y_temp))-1)),
                                text="Cluster {}".format(lab_c),
                                showarrow=False,
                                #xanchor="right",
                                xshift=-min_pos_x,
                                bgcolor=(cols[lab_c][:-1]+",1)").replace('rgb','rgba'),
                                #font=dict(
                                #    family="sans serif",
                                #    size=18,
                                #    color=(cols[lab_c][:-1]+",1)").replace('rgb','rgba'),
                                #)
                            )

                        fig_graph_quiz.update_layout(
                            height=300,
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            showlegend=False,
                            hovermode='closest',
                            #margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        
                        st.plotly_chart(fig_graph_quiz, use_container_width=True,height=300,key="question_graph_{}".format(current_question))
                    with col_ts_quiz:
                        st.plotly_chart(fig,height=300,key="question_ts_{}".format(current_question),use_container_width=True)
                    
                    
                elif (correspondance_dict[method] == 'kmean') or (correspondance_dict[method] == 'kshape'):
                    
                    for id_c,centroid in enumerate(centroids):
                        fig.add_scatter(x=[val for val in range(len(centroid))], y=centroid, mode='lines', name="Centroid {}".format(id_c), visible='legendonly',line_color=cols[id_c])
                    st.plotly_chart(fig,height=300,use_container_width=True)
            # question_placeholder.write(f"**{ss.current_quiz[i].get('question')}**") 
            # list of options
            options = ss.current_quiz[i].get("options")
            # track the user selection
            options_placeholder.radio("Your answer:", options, index=1, key=f"Q{current_question}",horizontal=True)
            #nl(1)
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
    scorecard_placeholder.progress(ss['grade']/len(ss.current_quiz), text="Your Final Score : {} / {}".format(ss['grade'],len(ss.current_quiz)))
    #scorecard_placeholder.write(f"### **Your Final Score : {ss['grade']} / {len(ss.current_quiz)}**")        
