#
# This module contains I/O helper classes for this project.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np


def load_training_density(filedir):
    """
    # Load the MD density data of the classification.
    #
    # Input
    # =====
    # `filedir`: (str) The directory containing the data.
    #
    # Return
    # =====
    # `densities`: (list) A list of 2-D arrays of the densities.
    # `labels`: (list) A list of arrays of the labels arranged as
    #           (is_begnin, is_pathogenic).
    """
    import os
    import glob
    d_shape = (32, 32, 1)  # Density shape
    bp = os.path.join(filedir, 'Benign/density')
    pp = os.path.join(filedir, 'Pathogenic/density')
    bs = glob.glob(bp + '/*_density.csv')
    ps = glob.glob(pp + '/*_density.csv')

    # Load
    densities = []
    labels = []
    for b in bs:
        bb = np.loadtxt(b, delimiter=',', skiprows=1)[:, 1]  # Get 2nd col.
        densities.append(np.reshape(bb, d_shape))
        labels.append([[[1, 0]]])
    for p in ps:
        pp = np.loadtxt(p, delimiter=',', skiprows=1)[:, 1]  # Get 2nd col.
        densities.append(np.reshape(pp, d_shape))
        labels.append([[[0, 1]]])
    return np.asarray(densities), np.asarray(labels)


def load_training_rama(filedir, postfix='', extra=False):
    """
    # Load the MD rama data of the classification.
    #
    # Input
    # =====
    # `filedir`: (str) The directory containing the data.
    #
    # Return
    # =====
    # `densities`: (list) A list of 2-D arrays of the densities.
    # `labels`: (list) A list of arrays of the labels arranged as
    #           (is_begnin, is_pathogenic).
    # `mutants`: (list) A list of mutants.
    """
    import os
    import glob
    import re
    # rama shape: (time_frame, protein_size, phi_psi)
    if 'TP53' in filedir:
        d_shape = (334, 217, 2)
    elif 'MLH1' in filedir:
        d_shape = (334, 346, 2)
        skip = ['M1I', 'M1K', 'M1R', 'M1T', 'M1V']
    bp = os.path.join(filedir, 'Benign/rama_csv' + postfix)
    pp = os.path.join(filedir, 'Pathogenic/rama_csv' + postfix)
    bs = glob.glob(bp + '/*_rama.csv')
    ps = glob.glob(pp + '/*_rama.csv')

    # Load
    densities = []
    labels = []
    mutants = []
    for b in bs:
        bb = np.loadtxt(b, delimiter=',', usecols=[0, 1])  # skip last col.
        densities.append(np.reshape(bb, d_shape).reshape(-1, d_shape[1] * d_shape[2]))
        labels.append([[[1, 0]]])
        mutants.append(re.findall('rama\_csv%s\/(.*)\_rama\.csv' % postfix, b)[0])
    for p in ps:
        m = re.findall('rama\_csv%s\/(.*)\_rama\.csv' % postfix, p)[0]
        if m in skip:
            print('Skipping', m)
            continue
        pp = np.loadtxt(p, delimiter=',', usecols=[0, 1])  # skip last col.
        densities.append(np.reshape(pp, d_shape).reshape(-1, d_shape[1] * d_shape[2]))
        labels.append([[[0, 1]]])
        mutants.append(re.findall('rama\_csv%s\/(.*)\_rama\.csv' % postfix, p)[0])
    if extra:
        wp = os.path.join(filedir, 'Benign')
        ws = glob.glob(wp + '/wildtype*_rama.csv')
        for b in ws:
            if 'None' in b:
                print('Skipping', b)
                continue
            bb = np.loadtxt(b, delimiter=',', usecols=[0, 1])  # skip last col.
            densities.append(np.reshape(bb, d_shape).reshape(-1, d_shape[1] * d_shape[2]))
            labels.append([[[1, 0]]])
            mutants.append(re.findall('Benign\/(.*)\_.*\_.*ns\_rama\.csv', b)[0])
    return np.asarray(densities), np.asarray(labels), mutants


def load_vus_rama(filedir, postfix=''):
    """
    # Load the MD rama data of the classification.
    #
    # Input
    # =====
    # `filedir`: (str) The directory containing the data.
    #
    # Return
    # =====
    # `densities`: (list) A list of 2-D arrays of the densities.
    # `mutants`: (list) A list of mutants.
    """
    import os
    import glob
    import re
    # rama shape: (time_frame, protein_size, phi_psi)
    if 'TP53' in filedir:
        d_shape = (334, 217, 2)
    elif 'MLH1' in filedir:
        d_shape = (334, 346, 2)
    bp = os.path.join(filedir, 'VUS/rama_csv' + postfix)
    bs = glob.glob(bp + '/*_rama.csv')
    if 'TP53' in filedir:
        skip = ['C242G', 'V216G']  # ['K101Q']
    elif 'MLH1' in filedir:
        skip = [] # ['G181D', 'V326M', 'I50F', 'G98D', 'V16G']

    # Load
    densities = []
    mutants = []
    for b in bs:
        m = re.findall('rama\_csv%s\/(.*)\_rama\.csv' % postfix, b)[0]
        if m in skip:
            print('Skipping', m)
            continue
        bb = np.loadtxt(b, delimiter=',', usecols=[0, 1])  # skip last col.
        try:
            densities.append(np.reshape(bb, d_shape).reshape(-1, d_shape[1] * d_shape[2]))
            mutants.append(m)
        except ValueError:
            print('Cannot load', m)
    return np.asarray(densities), mutants
