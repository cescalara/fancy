import pickle
import os


def get_nucleartable():
    '''Read dictionary from nuclear_table.pkl'''
    this_fpath = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_fpath, "nuclear_table.pkl"), "rb") as f:
        nuc_table = pickle.load(f)

    return nuc_table