'''
File: dat_to_lammpstrj.py
Project: fractal_analysis
File Created: Saturday, 20th March 2021 3:22:02 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Saturday, 20th March 2021 6:17:16 pm
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2018 - 2019 Amru, University of Pennsylvania

Summary: Fill In
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join, isfile
from os import popen, makedirs, walk
import os
import itertools as it
from scipy import stats
import numpy_indexed as npi

#plt.style.use('https://gist.githubusercontent.com/amruthesht/bd3febe2544cb96699b127a3e1f7d7a8/raw/439de31c30b68784519d8d7e81b1e31f8135b404/matplotlibstyle')

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

f = pd.read_csv(join(output_foldername, run_filename), sep="\n",
                header=None, skip_blank_lines=False)
f = f[0].str.split(r"\s{2,}", expand=True)
f.columns = ["value", "name"]

binary_extension = ".dat"
text_extension = ".lammpstrj"

utils_filename = str(f.loc[f["name"] == "utils_filename"]["value"].values[0])

frame_filename = str(f.loc[f["name"] == "frame_filename"]["value"].values[0])

output_mode = float(
    f.loc[f["name"] == "output_mode"]["value"].values[0])

d = 3
x = ["x", "y", "z"]

files = [files for root, folders, files in walk(output_foldername)]

for filename in files[0]:
    if utils_filename in filename and binary_extension in filename:

        dtype = np.dtype([("system_counter", 'i4'), ("t", 'f8'), ("counter", 'i4'), ("counter_time", 'f8'), ("state_flag", 'f8'), ("N", 'i4'), ("vol_frac", 'f8')] +  [("L_box" + str(k+1), 'f8') for k in range(d)] + [("p_idx", 'i4'), ("r_idx", 'i4'), ("flag", 'i4'), ("R", 'f8')] + [("x_" + str(k+1), 'f8') for k in range(d)] + [("dx_" + str(k+1), 'f8') for k in range(d)] + [("deltax_" + str(k+1), 'f8') for k in range(d)] +
                                [("s", 'f8'), ("ds", 'f8'), ("contour", 'f8'), ("dcontour", 'f8'), ("U", 'f8'), ("dU", 'f8'), ("dU_U", 'f8'), ("Z", 'f8')] + [("Tau_" + str(k+1), 'f8') for k in range(d * d)])

        f_in = open(join(output_foldername, filename), "rb")
        f_in.seek(0)

        data = np.fromfile(f_in, dtype=dtype)

        filename = filename.replace(utils_filename, frame_filename)

        filename = filename.replace(".dat", ".txt")

        f_out = open(join(output_foldername, filename), "w")

        f_out.write("ITEM: TIMESTEP\n")
        f_out.write(str(data["t"][0]) + '\n')
        f_out.write("ITEM: NUMBER OF ATOMS\n")
        f_out.write(str(data["N"][0]) + '\n')
        f_out.write("ITEM: BOX BOUNDS pp pp pp\n")
        f_out.write("ITEM: ATOMS id type mol")
        for k in range(d):
            f_out.write(" " + x[k])
        f_out.write('\n')
        for k in range(d):
            f_out.write("0.0 " + str(data["L_box" + str(k+1)][0]) + '\n')
        for i in range(data.shape[0]):
            if data["flag"][i] == 1:
                if data["r_idx"][i] > 1:
                    f_out.write('\n')
                f_out.write(str(data["r_idx"][i]) + " " + "1" + " " + str(data["p_idx"][i]))
                for k in range(d):
                    f_out.write(" " + str(data["x_" + str(k+1)][i]))
                f_out.write(" " + str(data["R"][i]))