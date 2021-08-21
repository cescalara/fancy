import pickle
import os
import numpy as np


def get_nucleartable():
    '''Read dictionary from nuclear_table.pkl'''
    this_fpath = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_fpath, "nuclear_table.pkl"), "rb") as f:
        nuc_table = pickle.load(f)

    return nuc_table

'''Integral of Fischer distribution used to evaluate kappa_d'''
def fischer_int(kappa, cos_thetaP):
    '''Integral of vMF function over all angles'''
    return (1. - np.exp(-kappa * (1 - cos_thetaP))) / (1. - np.exp(-2.*kappa))

def fischer_int_eq_P(kappa, cos_thetaP, P):
    '''Equation to find roots for'''
    return fischer_int(kappa, cos_thetaP) - P