import pickle
import os
import numpy as np


def get_nucleartable():
    '''Read dictionary of nuclear tables'''
    nuc_table = {
        "p": (1, 1),
        "H": (1, 1),
        "He": (4, 2),
        "Li": (7, 3),
        "C": (12, 6),
        "N": (14, 7),
        "O": (16, 8),
        "Si": (28, 14),
        "Fe": (56, 26),
    }

    return nuc_table

'''Integral of Fischer distribution used to evaluate kappa_d'''
def fischer_int(kappa, cos_thetaP):
    '''Integral of vMF function over all angles'''
    return (1. - np.exp(-kappa * (1 - cos_thetaP))) / (1. - np.exp(-2.*kappa))

def fischer_int_eq_P(kappa, cos_thetaP, P):
    '''Equation to find roots for'''
    return fischer_int(kappa, cos_thetaP) - P