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
