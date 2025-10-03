import streamlit as st
from pathlib import Path

# Session state variables for the Topic Visualization page
if ('p_topics' not in st.session_state 
    or 'p_terms' not in st.session_state 
    or 'p_random_state' not in st.session_state 
    or 'p_chunk_size' not in st.session_state 
    or 'p_num_passes' not in st.session_state 
    or 'p_vis_html' not in st.session_state 
    or 'p_model' not in st.session_state or 
    'p_json_file' not in st.session_state or 
    'p_model' not in st.session_state
    or 'p_vis_plot' not in st.session_state
    or 'p_visualization' not in st.session_state
    or 'p_fig_plotly' not in st.session_state
    or 'p_param_changed' not in st.session_state):
    st.session_state.p_topics = 10
    st.session_state.p_terms = 15
    st.session_state.p_chunk_size = 100
    st.session_state.p_num_passes = 10
    st.session_state.p_vis_html = None
    st.session_state.p_model = None
    st.session_state.p_json_file = None
    st.session_state.p_model = None
    st.session_state.v_vis_plot = None
    st.session_state.p_visualization = None
    st.session_state.p_fig_plotly = None
    st.session_state.p_param_changed = True

# Session state variables for the File Selection page
if ('p_first_page' not in st.session_state 
    or 'p_last_page' not in st.session_state 
    or 'p_current_first' not in st.session_state 
    or 'p_current_last' not in st.session_state 
    or 'p_file_loaded' not in st.session_state 
    or 'p_range_set' not in st.session_state 
    or 'p_display_file' not in st.session_state 
    or 'p_selected_pdf' not in st.session_state 
    or 'p_pdf' not in st.session_state 
    or 'p_selected_pdf' not in st.session_state 
    or 'p_suggest_list' not in st.session_state):
    st.session_state.p_first_page = 1
    st.session_state.p_last_page = 1
    st.session_state.p_current_first = 1
    st.session_state.p_current_last = 1
    st.session_state.p_range_set = False
    st.session_state.p_display_file = False
    st.session_state.p_pdf = None
    st.session_state.p_selected_pdf = None
    st.session_state.p_file_loaded = False
    st.session_state.p_suggest_list = []

# Session state variables for the Text Exploration page
if ('p_dataexp_json_file' not in st.session_state 
    or 'p_dataexp_docstruc_bins' not in st.session_state 
    or 'p_explore' not in st.session_state 
    or 'p_common_words_bins' not in st.session_state 
    or 'p_ngram' not in st.session_state 
    or 'p_ngram_cnt' not in st.session_state 
    or 'p_entity' not in st.session_state 
    or 'p_entity_cnt' not in st.session_state):
    st.session_state.p_dataexp_json_file = None
    st.session_state.p_dataexp_docstruc_bins = 50
    st.session_state.p_explore = None
    st.session_state.p_common_words_bins = 30
    st.session_state.p_ngram = 2
    st.session_state.p_ngram_cnt = 40
    st.session_state.p_entity = "GPE"
    st.session_state.p_entity_cnt = 20

def read_markdown_file(markdown_file):
    path = Path(markdown_file)
    return path.read_text()

st.set_page_config(
    page_title="Home",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''
        ### LDA Topic Modelling for Indexers
        ##### Created by [Don Howes](https://dhindexing.ca)
        ##### Github repo:
        Application released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license
        ''',
        "Get Help": "https://github.com/DWHowes/LDA-Topic-Modeling/blob/main/readme.md"
    }
    )

intro_markdown = read_markdown_file("Home.md")
st.markdown(intro_markdown, unsafe_allow_html=True)