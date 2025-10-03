import json
from pathlib import Path
import os
import glob

def load_json_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json_data(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def stopwords(nlp, custom_stop):
    if custom_stop:
        nlp.Defaults.stop_words |= custom_stop
    return nlp.Defaults.stop_words

def get_fname(fpath:str)->str:
    return(Path(fpath).stem)

# Return all json files present in the current working directory
def get_json_files()->list:
    # absolute path to search all text files inside a specific folder
    cur_dir = os.getcwd()
    path = cur_dir + '/' + '*.json'
    return glob.glob(path)

# Traverse a number range from start to end inclusive. This is different from
# range(), which only traverses to end-1
def Interval(start:int, end:int, step:int=1):
    i = start
    while i < end:
        yield i
        i += step
    yield end        

# Calculates the maximum number of columns in a csv file and returns a generated
# list of column names. Use the returned value in the 'names=' parameter of
# a Pandas read_csv call.
def max_csv_columns(fname:str, delimiter)->list:
    # The max column count a line in the file could have
    largest_column_count = 0

    # Loop the data lines
    with open(fname, 'r') as temp_f:
        # Read the lines
        lines = temp_f.readlines()

        for l in lines:
            # Count the column count for the current line
            column_count = len(l.split(delimiter)) + 1
            
            # Set the new most column count
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count

    # Generate column names (will be 0, 1, 2, ..., largest_column_count - 1)
    column_names = [i for i in range(0, largest_column_count)]

    return column_names    