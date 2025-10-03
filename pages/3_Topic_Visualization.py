from utility import get_json_files
from model import docModel

import streamlit as st
import streamlit.components.v1 as v1

# Text for widget help

json_help = "The json file containing the word list for processing. If more than one json file is present in the current working directory, the first file found is used as the default selection."
topic_help = "The number of topics to be extracted from the corpus"
term_help = "The number of terms returned for each topic"
chunk_help = "The number of documents to be used in each training chunk"
passes_help = "The number of passes through the corpus during training"
random_help = "The seed used to generate a randomState object"
visualization_help = "LDA visualization methods"


#  callback functions

def on_submit():
    # Create the LDA model
    if not st.session_state.p_model:
        st.session_state.p_model = docModel(st.session_state.p_json_file)

    if st.session_state.p_model:
        try:
            # Generate a trained model on first use. A new model is subsequently created only if one or more
            # of the input parameters have changed.
            if st.session_state.p_param_changed:
                st.session_state.p_model.lda_model(num_topics=st.session_state.p_topics, 
                                                    chunks=st.session_state.p_chunk_size,
                                                    passes=st.session_state.p_num_passes
                                                    )
                st.session_state.p_param_changed = False
            
            st.session_state.p_vis_plot = st.session_state.p_model.vis_data(st.session_state.p_visualization)
        except Exception as e:
            st.error("Model creation failed with error: {}".format(e))

def json_select():
    st.session_state.p_param_changed = True
    st.session_state.p_json_file = st.session_state._json_file
    st.session_state.p_model = None

def topic_change():
    st.session_state.p_param_changed = True
    st.session_state.p_topics = st.session_state._topics

def terms_change():
    st.session_state.p_param_changed = True
    st.session_state.p_terms = st.session_state._terms

def chunk_change():
    st.session_state.p_param_changed = True
    st.session_state.p_chunk_size = st.session_state._chunk_size

def passes_change():
    st.session_state.p_param_changed = True
    st.session_state.p_num_passes = st.session_state._num_passes

def visualization_change():
    st.session_state.p_visualization = st.session_state._visualization

# Page configuration

st.set_page_config(
    page_title="Topic Modeling",
    layout="wide",
    initial_sidebar_state="collapsed")

# Page layout

col1, col2 = st.columns([1,6])

with col1:
    visualizations = ["topic map", 
                      "topic similarity",
                      "topic barchart",
                      "topic clouds",
                      "topic sunburst",
                      "topic treemap",
                      "document topics",
                      "documents", 
                      "3D document topics", 
                      "cluster map"]

    files = get_json_files()
    alpha_list = ["symmetric","asymmetric", "auto"]
    disable = not files
    st.markdown("##### :orange[JSON]")
    if files:
        if not st.session_state.p_json_file:
            st.session_state.p_json_file = files[0]
        file = st.selectbox("JSON File",
                            options=[file for file in files],
                            index= files.index(st.session_state.p_json_file),
                            key='_json_file', 
                            on_change=json_select,
                            help=json_help, 
                            disabled=disable)
    else:
        st.error("No JSON files found")
    
    st.markdown("##### :orange[LDA Parameters]")
    topics = st.number_input("Number of Topics", 
                                min_value=1, 
                                max_value=30, 
                                value=st.session_state.p_topics,
                                key='_topics', 
                                on_change=topic_change,
                                help=topic_help, 
                                disabled=disable)
    terms = st.number_input("Number of Terms", 
                                min_value=1, 
                                max_value=30, 
                                value=st.session_state.p_terms,
                                key='_terms', 
                                on_change=terms_change,
                                help=term_help, 
                                disabled=disable)
    chunk = st.number_input("Chunk Size", 
                            min_value=10, 
                            max_value=500, 
                            value=st.session_state.p_chunk_size,
                            key='_chunk_size', 
                            on_change=chunk_change,
                            help=chunk_help, 
                            disabled=disable)
    passes = st.number_input("Data Passes", 
                                min_value=1, 
                                max_value=50, 
                                value=st.session_state.p_num_passes,
                                key='_num_passes', 
                                on_change=passes_change,
                                help=passes_help, 
                                disabled=disable)

    st.markdown("##### :orange[LDA Visualizations]")
    select = st.selectbox("LDA Visualizations",
                        options=visualizations,
                        index=0,
                        key='_visualization', 
                        on_change=visualization_change,
                        help=visualization_help, 
                        disabled=disable)
    st.session_state.p_visualization = select

    st.button("Display", on_click=on_submit, disabled=disable)

with col2:
    try:
        if st.session_state.p_vis_html:
            v1.html(st.session_state.p_vis_html, width=1200,height=800)
        elif st.session_state.p_fig_plotly == True:
            st.plotly_chart(st.session_state.p_vis_plot, use_container_width=True, theme=None)
        elif st.session_state.p_fig_plotly == False:
            st.pyplot(st.session_state.p_vis_plot, use_container_width=True)

        st.session_state.p_vis_html = None
        st.session_state.p_fig_plotly = None
    except Exception as e:
            st.error("Visualization failed with error: {}".format(e))
