'''
File: correlation_dimension.py
Project: fractal_analysis
File Created: Thursday, 30th June 2022 7:22:10 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Thursday, 30th June 2022 8:06:07 pm
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2020 - 2021 Amru, University of Pennsylvania

Summary: correlation_dimension calculation using 3N particle positions

Example shown below: Data for https://doi.org/10.48550/arXiv.2204.00587
'''

#%%
from os import popen, makedirs, system, walk
from os.path import join, isfile, isdir, basename, dirname, exists
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import os
from os.path import join
from cycler import cycler
import random
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import FixedFormatter
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator
from mpl_toolkits import mplot3d
from sklearn.metrics import pairwise_distances
from statsmodels.graphics import tsaplots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

input_foldername = "input/"
output_foldername = "output/"
graphs_foldername = "graphs/"

d = 3 # dimensionality of the system

N_init = 342 # number of d-dimensional positions; N = 1 for a 3D system

binning_type = 1 # 1 for unique binning, 2 for logarithmic binning, 3 for linear binning

PATH = join(input_foldername)
dpos_filename = "positions.txt" # file name for the D-dimensional positions

length_scale = 3  # length scale relevant tot he problem

dpos_all = np.array([])

dpos = pd.read_csv(join(PATH, dpos_filename), sep=r",",
                   header=None, skiprows=1, dtype="float")
dpos.columns = ["t", "dt"]+[("x_" + str(k+1) + "_" + str(i))
                            for k in range(d) for i in range(N_init)]
dpos = dpos.iloc[:50, 2:]
dpos = np.array(dpos)

clustering = AgglomerativeClustering(
    n_clusters=1, compute_full_tree=True, linkage="single", compute_distances=True).fit(dpos)
labels = clustering.labels_ + 1
# print(dpos.shape, np.size(clustering.distances_))

x_decorr = np.max(clustering.distances_)

dpos = pairwise_distances(dpos)
# dpos = dpos[np.triu_indices(dpos.shape[0], 1)]
# dpos = dpos[dpos>1e-3]

dpos_mod = dpos
dpos = np.array([])
t_decorr = 0
random.seed(1)
n = 3
for n_steps in range(n):
    index_list = np.array([])
    for index, x in np.ndenumerate(dpos_mod):
        if (index[0] < index[1]) & (index[1] - index[0] >= t_decorr) & (index[0] not in index_list) & (index[1] not in index_list):
            if (x < x_decorr):
                # dpos = np.append(dpos, x)
                remove_index = random.choice(index)
                # random.choice(index))
                index_list = np.append(index_list, remove_index)
                # for i in range(dpos_mod.shape[0]):
                #     if i != remove_index:
                #         if dpos_mod[remove_index][i] < x_decorr:
                #             index_list = np.append(index_list, i)
    indecies = np.arange(0, dpos_mod.shape[0])
    random.shuffle(indecies)
    for index in indecies:
        if index not in index_list:
            for i in range(dpos_mod.shape[0]):
                if (i != index) & (i not in index_list):
                    if (dpos_mod[i][index] < x_decorr):
                        index_list = np.append(index_list, i)

    for index, x in np.ndenumerate(dpos_mod):
        if (index[0] < index[1]) & (index[1] - index[0] >= t_decorr) & (index[0] not in index_list) & (index[1] not in index_list):
            if (x >= x_decorr):
                dpos = np.append(dpos, x)
dpos_all = np.append(dpos_all, dpos)

dpos = dpos_all
dpos /= length_scale

if binning_type == 1:
    bins = np.unique(dpos)
elif binning_type == 2:
    _, bins_hist = np.histogram(np.log10(dpos), bins="auto")
    bins_hist = 10**bins_hist
    bins = bins_hist
    bins = 10**(0.5 * (np.log10(bins[1:])+np.log10(bins[:-1])))
elif binning_type == 3:
    _, bins_hist = np.histogram((dpos), bins="auto")
    bins = bins_hist
    bins = (0.5 * ((bins[1:])+(bins[:-1])))

N_r = np.sum(dpos[:, None] < bins, axis=0)
N = len(dpos)
x = bins
y = N_r/N
alpha = 0.05
epsilon = np.sqrt(np.log(2/alpha)/(2*N))
# epsilon = 1.96*y4*(1-y4)/N
x_cutoff = 3.2e0
y = y[x >= x_cutoff]
x = x[x >= x_cutoff]
# upper = np.clip(y41+epsilon, 0, 1)-y41
# lower = -np.clip(y41-epsilon, 0, 1)+y41
# plt.errorbar(x4, y4, yerr=[lower, upper], fmt='o', linewidth=1, capsize=1, elinewidth=0.75, markersize=2)
plt.scatter(x, y)
plt.ylabel(r"$CDF(||\Delta {\mathbf{r}}||)$")
plt.xlabel(r"$||\Delta {\mathbf{r}}||$")
plt.xscale("log")
plt.yscale("log")
plt.savefig(output_foldername + graphs_foldername +
            "CDF.jpg", dpi=1000, bbox_inches='tight')