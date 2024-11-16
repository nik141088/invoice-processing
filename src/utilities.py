import os
import pickle
import glob
import pandas as pd
import streamlit as st

from src.constants import ROOT_DIR, INVOICE_DIR, IMAGE_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_DIR, WORDS_AND_BOXES_DIR, DUPLICATES_DIR, INV_DONE_DIR, PREV_LOOP_ITERATION_FILE


def create_dirs():
    """
    This function creates directories if they don't exist
    :param: None
    :return: None
    """
    # all directories are defined relative to ROOT_DIR. If in future the app is moved to a different folder, then only ROOT_DIR needs to be changed!
    if not os.path.exists(ROOT_DIR):
        raise Exception('Root directory not found!')

    for d in [INVOICE_DIR, IMAGE_EMBEDDINGS_DIR, TEXT_EMBEDDINGS_DIR, WORDS_AND_BOXES_DIR, DUPLICATES_DIR, INV_DONE_DIR]:
        if not os.path.exists(d):
            os.mkdir(d)


def initialize_loop_stack():
    """
    This function initializes the loop stack file
    :param: None
    :return: None
    """
    # PREV_LOOP_ITERATION_FILE is a keyed stack, i.e. it is a stack which also checks for entered key.
    # This is used to keep track of sequence of invoices that the user has checked. We use this to go back to earlier invoice!
    if not os.path.exists(PREV_LOOP_ITERATION_FILE):
        with open(PREV_LOOP_ITERATION_FILE, 'wb') as fp:
            curr_inv = ''
            curr_data = list()
            pickle.dump([curr_inv, curr_data], fp)


def list_files(directory):
    """
    This function lists all files in a directory
    :param directory: directory name
    :return: list of files
    """
    return list(glob.glob(os.path.join(directory, '*')))


# @st.cache
@st.experimental_memo
def convert_df_to_csv(df):
    """
    This function converts a pandas dataframe to csv
    :param df: pandas dataframe
    :return: csv file
    """
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


# @st.cache
@st.experimental_memo
def download_all_image_duplicates(ret_csv=True):
    """
    This function downloads all image duplicates
    :param ret_csv: whether to return csv or the dataframe
    :return: csv file or dataframe
    """
    dups = list_files(DUPLICATES_DIR)
    dt_list = [pickle.load(open(dup_file, 'rb')) for dup_file in dups]
    df = pd.concat(dt_list)
    df = df.sort_values(["org", "dup"])
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    if ret_csv:
        return df.to_csv().encode('utf-8')
    else:
        return df


def delete_all_files_in_dir(dir_name):
    """
    This function deletes all files in a directory
    :param dir_name: directory name
    :return: None
    """
    file_names = list_files(dir_name)
    for f in file_names:
        os.remove(f)


def get_short_names(file_list, suffix):
    """Extract short names by removing directory path and suffix."""
    return [os.path.basename(f).rstrip(suffix) for f in file_list]


def get_duplicate_files(ret_short_names=False):
    """Retrieve all duplicate files."""
    duplicates_names = list_files(DUPLICATES_DIR)
    if not ret_short_names:
        return duplicates_names
    duplicates_short_names = get_short_names(duplicates_names, ".dup")
    return duplicates_names, duplicates_short_names
