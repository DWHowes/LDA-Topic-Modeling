import streamlit as st

import time

from pdfdoc import PDFdoc

# Text for widget help

first_help = "The first page of the PDF included for processing. This should be the first page following the fore matter"
last_help = "The last page of the PDF included for processing. This should be the last page preceding the end matter"
view_help = "Enable if you wish to see the PDF file in the embedded PDF file viewer"
file_help = "Select the PDF file for processing. Parsing the layout of the PDF file can take 5-10 minutes."
csv_help = "CSV file containing custom stop word list"

# Callback functions

def on_submit():
    st.session_state.p_pdf.process_pdf(st.session_state.p_current_first, 
                                       st.session_state.p_current_last)
    
def first_change():
    st.session_state.p_current_first = st.session_state._first_min

def last_change():
    st.session_state.p_current_last = st.session_state._last_max

def file_select():
    if st.session_state.uploaded_file:
        try:
            start_time = 0
            end_time = 0
            st.session_state.p_selected_pdf = st.session_state.uploaded_file
            start_time = time.time()
            with st.spinner("Parsing Layout for File: {} ...".format(st.session_state.p_selected_pdf.name), show_time=True):
                st.session_state.p_pdf = PDFdoc(st.session_state.p_selected_pdf)
            end_time = time.time()
            elapsed_time = round(end_time - start_time)
            st.write("Elapsed Time: {} seconds.".format(elapsed_time))
            st.session_state.p_file_loaded = True
            st.session_state.p_range_set = False
        except Exception as e:
            st.error("An Unexpected Error Occurred: {}".format(e))
    else:
        st.session_state.p_file_loaded = False

# Page configuration

st.set_page_config(
    page_title="Select File",
    layout="wide",
    initial_sidebar_state="collapsed")

# Page layout

col1, col2 = st.columns([2,5])

with col1:
    st.markdown("### Process File")
    if st.session_state.p_file_loaded:
        if not st.session_state.p_range_set:
            st.session_state.p_first_page = st.session_state.p_current_first = st.session_state.p_pdf.get_first_page()
            st.session_state.p_last_page = st.session_state.p_current_last = st.session_state.p_pdf.get_last_page()
            st.session_state.p_range_set = True

    first = st.number_input("First Page to Process", 
                            min_value=st.session_state.p_first_page, 
                            max_value=st.session_state.p_last_page, 
                            disabled= not st.session_state.p_file_loaded, 
                            key='_first_min',
                            on_change=first_change,
                            help=first_help)
    last = st.number_input("Last Page to Process", 
                            min_value=st.session_state.p_first_page, 
                            max_value=st.session_state.p_last_page, 
                            value=st.session_state.p_last_page,
                            disabled= not st.session_state.p_file_loaded, 
                            key='_last_max',
                            on_change=last_change,
                            help=last_help)
    submitted = st.button("Process File", disabled= not st.session_state.p_file_loaded, on_click=on_submit)

with col2:
    if st.session_state.p_selected_pdf:
        st.write("Currently Selected File: ", st.session_state.p_selected_pdf.name)
    st.file_uploader("Select a PDF file", type=['pdf'], key='uploaded_file', on_change=file_select, help=file_help)
