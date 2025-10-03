import streamlit as st

from Data_Explorer import DataExplorer
from utility import get_json_files

json_help = "The json file containing the word list for text data exploration. If more than one json file is present in the current working directory, the first file found is used as the default selection"
struc_bins_help = "The number of aggregation bins for document character and word counts"
common_words_help = "The number of common words to be displayed"
ngram_help = "Display word n-grams. A value of 2 shows bigrams, 3 shows trigrams, etc. Range is capped at 5 words."
count_ngram_help = "The number of n-grams to be returned"
sentiment_help = "Display the sentiment analysis"
entity_help = "Display the selected Named Entity"
count_entity_help = " The number of the selected Named Entity to be returned"

# Page configuration
st.set_page_config(
    page_title="Text Exploration",
    layout="wide",
    initial_sidebar_state="collapsed")

# Page layout
col1, col2, col3 = st.columns([1,1, 3], vertical_alignment="top")

def json_select():
    st.session_state.p_dataexp_json_file = st.session_state._p_dataexp_json_file
    st.session_state.p_explore = DataExplorer(st.session_state.p_dataexp_json_file)

def struc_bins_change():
    st.session_state.p_dataexp_docstruc_bins = st.session_state._p_dataexp_docstruc_bins

def common_bins_change():
    st.session_state.p_common_words_bins = st.session_state._p_common_words_bins

def ngram_change():
    st.session_state.p_ngram = st.session_state._p_ngram

def ngram_cnt_change():
    st.session_state.p_ngram_cnt = st.session_state._p_ngram_cnt

def entity_change():
    st.session_state.p_entity = st.session_state._p_entity

def entity_cnt_change():
    st.session_state.p_entity_cnt = st.session_state._p_entity_cnt

def on_show_docstruct():
    if st.session_state.p_explore:
        with col3:
            st.session_state.p_explore.show_char_word(st.session_state.p_dataexp_docstruc_bins)
def on_show_common():
    if st.session_state.p_explore:
        with col3:
            st.session_state.p_explore.show_common_words(st.session_state.p_common_words_bins)

def on_show_ngram():
    if st.session_state.p_explore:
        with col3:
            st.session_state.p_explore.show_ngrams(st.session_state.p_ngram, st.session_state.p_ngram_cnt)

def on_show_entity():
    if st.session_state.p_explore:
        with col3:
            try:
                st.session_state.p_explore.show_entity(st.session_state.p_entity, st.session_state.p_entity_cnt)
            except:
                st.markdown("## No Items To Display")

# Controls for exploratory data analysis techniques
with col1:
    files = get_json_files()
    disable = not files
    entities_list = ["EVENT", "FAC", "GPE", "LAW", "LOC", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]
    st.markdown("##### Data File")
    if files:
        if not st.session_state.p_dataexp_json_file:
            st.session_state.p_dataexp_json_file = files[0]
            if not st.session_state.p_explore:
                st.session_state.p_explore = DataExplorer(st.session_state.p_dataexp_json_file)
        file = st.selectbox("JSON File",
                            options=[file for file in files],
                            index= files.index(st.session_state.p_dataexp_json_file),
                            key='_p_dataexp_json_file', 
                            on_change=json_select,
                            help=json_help, 
                            disabled=disable)   
    else:
        st.error("No JSON files found")
    
    st.markdown("##### Document Structure")
    # Display the number of characters and words per document
    st.number_input("Bins", 
                    min_value=1, 
                    max_value=100, 
                    value=st.session_state.p_dataexp_docstruc_bins,
                    key='_p_dataexp_docstruc_bins',
                    on_change=struc_bins_change,
                    help=struc_bins_help,
                    disabled=disable)
    st.markdown('##### Text Exploration')
    # Display common words in the corpus
    st.number_input("Common Words", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.p_common_words_bins,
                key='_p_common_words_bins',
                on_change=common_bins_change,
                help=common_words_help,
                disabled=disable)
    # Display n-grams
    st.number_input("N-Grams", 
                min_value=2, 
                max_value=5, 
                value=st.session_state.p_ngram,
                key='_p_ngram',
                on_change=ngram_change,
                help=ngram_help,
                disabled=disable)
    st.number_input("Number of N-Grams", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.p_ngram_cnt,
                key='_p_ngram_cnt',
                on_change=ngram_cnt_change,
                help=count_ngram_help,
                disabled=disable)
    st.selectbox("Named Entities", 
                 entities_list, 
                 index=entities_list.index(st.session_state.p_entity), 
                 key='_p_entity', 
                 on_change=entity_change, 
                 help=entity_help, 
                 disabled=disable)
    st.number_input("Number of Entities", 
                min_value=1, 
                max_value=100, 
                value=st.session_state.p_entity_cnt,
                key='_p_entity_cnt',
                on_change=entity_cnt_change,
                help=count_entity_help,
                disabled=disable)

# Execution buttons for the controls in col1. Padding where required so that the buttons line up with the controls
with col2:
    st.write('<div style="height: 192px;"> </div>', unsafe_allow_html=True)
    st.button("Show Text", on_click=on_show_docstruct, disabled=disable)
    st.write('<div style="height: 69px;"> </div>', unsafe_allow_html=True)
    st.button("Show Common", on_click=on_show_common, disabled=disable)
    st.write('<div style="height: 27px;"> </div>', unsafe_allow_html=True)
    st.button("Show N-Gram", on_click=on_show_ngram, disabled=disable)
    st.write('<div style="height: 113px;"> </div>', unsafe_allow_html=True)
    st.button("Named Entity", on_click=on_show_entity, disabled=disable)
