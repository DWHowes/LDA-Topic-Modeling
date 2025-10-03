import spacy as sp
from spacy_layout import spaCyLayout

import pandas as pd
import streamlit as st

import os
import json

import utility as utils
from preprocess import process

MIN_SPAN_SIZE = 100

class PDFdoc():
    def __init__(self, fname:str) -> None:
        self.fname = fname
        self.doc_list = []
        self.pp = process()
        self.stop_words = None
        self.nlp = None

        self.nlp = sp.load("en_core_web_sm")
        st.toast("Creating SpaCy Layout")
        layout = spaCyLayout(self.nlp)
        # Parsing the layout of the PDF file can take some time (5-10 minutes)
        st.toast("Parsing Document Layout")
        self.doc = layout(self.fname.read())

        self.first_process_page = 1
        self.last_process_page = len(self.doc._.layout.pages)

        if 'json_file' not in st.session_state:
            st.session_state.json_file = None


## PRIVATE METHODS

    def __tokenize(self, para:str)->list:
        lst = []

        lst = self.pp.remove_url(para)
        lst = self.pp.remove_html(lst)
        lst = self.pp.remove_cit(lst)
        lst = self.pp.tokenize(self.nlp(lst))
        lst = self.pp.pos(self.nlp(lst))
        lst = self.pp.lower(self.nlp(lst))
        lst = self.pp.del_punct(self.nlp(lst))
        lst = self.pp.lemma(self.nlp(lst))
        lst = self.pp.del_stop(self.nlp(lst))

        return lst
    
    # Save the pandas dataframe containing the processed data to disk
    def __save_df(self):
        name = self.fname.name
        json_file = utils.get_fname(name)+".json"
        st.session_state.json_file = json_file
        df = pd.DataFrame(self.doc_list)

        # Delete the JSON file if it already exists
        if os.path.exists(json_file):
            os.remove(json_file)

        # Name the text output
        df.columns = ['paragraphs']
        # Convert DataFrame to a dictionary with lists as values
        df_dict = df.to_dict(orient='list')
        # Save the dictionary to a JSON file
        with open(json_file, 'w') as f:
            json.dump(df_dict, f, indent=2) # indent for pretty-printing 
   
    # Create the list of documents (paragraphs) using the text extracted from the PDF
    def __create_doc_list(self, first:int, last:int)->None:
        paraList = []

        for i in utils.Interval(first, last):
            # Page count is zero-based
            page = (self.doc._.pages[i-1])
            # Save paragraph spans that exceed the minimum span length to a list
            for section in page[1]:
                if(section.label_ == "text" and len(section.text) > MIN_SPAN_SIZE):
                    paraList.append(section.text)

        # Preprocess paragraphs and create the word list
        para_len = len(paraList)
        para_cnt = 1
        prog_text = "Percent Completed"
        status_text = st.empty()
        prog_bar = st.progress(0, prog_text)
        for para in paraList:
            percent = int((para_cnt/para_len)*100)
            prog_bar.progress(percent)
            status_text.text("Progress: {}, span {} of {}".format(percent, para_cnt, para_len))
            tokens = self.__tokenize(para)
            self.doc_list.append(tokens)
            para_cnt += 1
            prog_bar.empty()

        # Save the word list as a JSON file
        self.__save_df()

    # PUBLIC METHODS
    def get_first_page(self)->int:
        return self.first_process_page
        
    def get_last_page(self)->int:
        return self.last_process_page

    def process_pdf(self, first:int, last:int)->None:
        self.__create_doc_list(first, last)
