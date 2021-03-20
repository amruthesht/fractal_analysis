'''
File: dr2ds_creator_old.py
Project: Q_analysis
File Created: Friday, 3rd May 2019 4:28:03 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Saturday, 20th March 2021 3:19:09 pm
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2018 - 2019 Amru, University of Pennsylvania

Summary: Fill In
'''
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join, isfile
from os import popen, makedirs, walk
import os
import itertools as it
from scipy import stats
import numpy_indexed as npi
import time

start_time = time.time()

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"


def periodic_BC(x1, x2, L):
    return (x1 - x2) - np.multiply(L, np.round(np.divide((x1 - x2), L)))

def d_mean(x, axis=0):
    return np.sum(np.nanmean(x, axis=axis))

def d_var(x, axis=0):
    return np.sum(np.nanvar(x, axis=axis))

f = pd.read_csv(join(output_foldername, run_filename), sep="\n",
                header=None, skip_blank_lines=False)
f = f[0].str.split(r"\s{2,}", expand=True)
f.columns = ["value", "name"]

utils_filename = str(f.loc[f["name"] == "utils_filename"]["value"].values[0])

U_filename = str(f.loc[f["name"] == "U_filename"]["value"].values[0])

type_quenching = int(f.loc[f["name"] == "type_quenching"]["value"].values[0])

output_mode = float(
    f.loc[f["name"] == "output_mode"]["value"].values[0])

dt_reset_system = float(
    f.loc[f["name"] == "dt_reset_system"]["value"].values[0])

d = 3

COARSEN_STEP_MAX = int(
    f.loc[f["name"] == "COARSEN_STEP_MAX"]["value"].values[0])
utils_print_frequency = int(
    f.loc[f["name"] == "utils_print_frequency"]["value"].values[0])

U = pd.read_csv(join(output_foldername, U_filename),
                sep=r"\s+", header=None, skiprows=1)

NUMBER_OF_COUNTERS = 1

if (type_quenching == 3):
    if (output_mode == 5):
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max", "kbT"]
    else:
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max"]
elif (type_quenching >= 4):
    if (output_mode == 5):
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU",
                     "dU/U", "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max",  "bias_U", "number of biases"]
    else:
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU",
                     "dU/U", "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max",  "bias_U", "number of biases"]
else:
    if (output_mode == 5):
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max"]
    else:
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max"]

NUMBER_OF_COUNTERS = int(np.max(U["counter"]))

utils_collection = {}

i_file_counter = 0

dt_range = [1, 10, 100]

dynamical_scaling_time = 400

files = [files for root, folders, files in walk(output_foldername)]

for filename in files[0]:
    if utils_filename in filename:

        i_file_counter += 1

        dtype = np.dtype([("system_counter", 'i4'), ("t", 'f8'), ("counter", 'i4'), ("counter_time", 'f8'), ("state_flag", 'f8'), ("N", 'i4'), ("vol_frac", 'f8')] +  [("L_box" + str(k+1), 'f8') for k in range(d)] + [("p_idx", 'i4'), ("r_idx", 'i4'), ("flag", 'i4'), ("R", 'f8')] + [("x_" + str(k+1), 'f8') for k in range(d)] + [("dx_" + str(k+1), 'f8') for k in range(d)] + [("deltax_" + str(k+1), 'f8') for k in range(d)] +
                                [("s", 'f8'), ("ds", 'f8'), ("contour", 'f8'), ("dcontour", 'f8'), ("U", 'f8'), ("dU", 'f8'), ("dU_U", 'f8'), ("Z", 'f8')] + [("Tau_" + str(k+1), 'f8') for k in range(d * d)])

        f = open(join(output_foldername, filename), "rb")
        f.seek(0)

        data = np.fromfile(f, dtype=dtype)

        utils_collection[i_file_counter] = pd.DataFrame(
        data, columns=data.dtype.names)

dt = np.array([])
dcounter = np.array([])
ds = np.array([])
dcontour = np.array([])
ddisplacement2 = np.array([])
dr2 = np.array([])
dTau2 = np.array([])

ddisplacement_dN = []
ddisplacement2_i = {}
ddisplacement2_t_i = []

dTau_d = []

ddisplacement2_i_dt = {}

for k in range(d):
    ddisplacement2_i["dx_i_" + str(k+1)] = []
    ddisplacement2_i_dt["dx_i_" + str(k+1)] = []

for index, (i, j) in enumerate(list(it.combinations(utils_collection.keys(), 2))):
    if (float(utils_collection[i]["t"][0]) > dynamical_scaling_time) & (float(utils_collection[j]["t"][0]) > dynamical_scaling_time):
        delt = abs(float(utils_collection[j]["t"][0]) -
                                    float(utils_collection[i]["t"][0]))

        delc = abs(float(utils_collection[j]["counter"][0]) -
                                float(utils_collection[i]["counter"][0]))

        if (output_mode == 5) & (delt == 0.0) & (delc > 0):
            if (float(utils_collection[i]["state_flag"][0]) > 1.0) & (float(utils_collection[j]["state_flag"][0]) > 1.0):

                dcounter = np.append(dcounter, delc)

                dcontour = np.append(dcontour, abs(float(utils_collection[j]["contour"][0]) -
                                float(utils_collection[i]["contour"][0])))

                dd = np.array([utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                    k+1)] for k in range(d)])[(utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1)]

                ddisplacement_dN.append(dd.flatten("F"))

                ddisplacement2 = np.append(ddisplacement2, np.nanmean(np.nanvar(dd, axis=1)))

                dr2 = np.append(dr2, sum(sum([np.power(np.multiply(periodic_BC(utils_collection[j]["x_"+str(k+1)], utils_collection[i]["x_"+str(k+1)], np.mean([utils_collection[j]["L_box"+str(k+1)][0], utils_collection[i]["L_box"+str(k+1)][0]])),
                                                        np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in range(d)])))

                dTau2 = np.append(dTau2, sum(sum([np.power(np.multiply(utils_collection[j]["Tau_"+str(k+1)] - utils_collection[i]["Tau_"+str(k+1)],
                                                            np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in [d, 2*d, 2*d+1]])))

        elif (output_mode == 1) & (delt > 0) & (delc == 0):
            if (float(utils_collection[i]["state_flag"][0]) == 0) & (float(utils_collection[j]["state_flag"][0]) == 0):

                dt = np.append(dt, delt)

                dcounter = np.append(dcounter, abs(float(utils_collection[j]["counter"][0]) -
                                    float(utils_collection[i]["counter"][0])))

                dcontour = np.append(dcontour, abs(float(utils_collection[j]["contour"][0]) -
                                float(utils_collection[i]["contour"][0])))

                ds = np.append(ds, abs(float(utils_collection[j]["s"][0]) -
                                float(utils_collection[i]["s"][0])))

                if (np.around(delt, 12) in dt_range):
                    for k in range(d):
                        ddisplacement2_i_dt["dx_i_" + str(k+1)] = [np.array(np.power((utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                            k+1)])[((utils_collection[i]["flag"] == 1) & (utils_collection[j]["flag"] == 1) & (utils_collection[i]["Z"] >= d+1) & (utils_collection[j]["Z"] >= d+1))], 2))]

                    ddisplacement2_t_i_dt = [np.array(sum([np.power((utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                        k+1)])[((utils_collection[i]["flag"] == 1) & (utils_collection[j]["flag"] == 1) & (utils_collection[i]["Z"] >= d+1) & (utils_collection[j]["Z"] >= d+1))], 2) for k in range(d)]))]
                    R = [np.array((utils_collection[j]["R"] + utils_collection[i]["R"])[((utils_collection[i]["flag"] == 1) & (
                        utils_collection[j]["flag"] == 1) & (utils_collection[i]["Z"] >= d+1) & (utils_collection[j]["Z"] >= d+1))]) / 2.0]
                    for k in range(d):
                        ddisplacement2_i_dt["dx_i_" + str(k+1)] = (
                            np.array(ddisplacement2_i_dt["dx_i_" + str(k+1)]) / np.power(np.nanmean(R), 2)).tolist()
                        ddisplacement2_i["dx_i_" + str(k+1)] += ddisplacement2_i_dt["dx_i_" + str(k+1)]
                    R = (np.array(R) / np.nanmean(R)).tolist()
                    ddisplacement2_t_i_dt = (
                        np.array(ddisplacement2_t_i_dt) / np.nanmean(R)).tolist()


                    ddisplacement2_t_i += ddisplacement2_t_i_dt

                dd = np.array([utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                    k+1)] for k in range(d)])[(utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1)]

                ddisplacement_dN.append(dd.flatten("F"))

                ddisplacement2 = np.append(ddisplacement2, np.nanmean(np.nanvar(dd, axis=1)))

                dr2 = np.append(dr2, sum(sum([np.power(np.multiply(periodic_BC(utils_collection[j]["x_"+str(k+1)], utils_collection[i]["x_"+str(k+1)], np.mean([utils_collection[j]["L_box"+str(k+1)][0], utils_collection[i]["L_box"+str(k+1)][0]])),
                                                        np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in range(d)])))

                dT = np.array([utils_collection[j]["Tau_"+str(k+1)] - utils_collection[i]["Tau_"+str(
                    k+1)] for k in range(d)])[(utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1)]
                
                dTau_d.append((np.sum(dT, axis=1)/ 2).flatten("F"))

                dTau2 = np.append(dTau2, np.sum((np.sum(dT, axis=1)/ 2)**2))



if (output_mode == 5):
    dt = dcounter
    #ds = dcontour

ddisplacement_dN = np.array(ddisplacement_dN)
dTau_d = np.array(dTau_d)

""" dr2ds_x, dr2ds_y = ds, dr2
_, bins = np.histogram(np.log10(dr2ds_x), bins='auto')
dr2ds_y, dr2ds_x, bnumbr = stats.binned_statistic(
    dr2ds_x, np.log10(dr2ds_y), statistic=np.nanmean, bins=10**bins)
dr2ds_y = 10**dr2ds_y
dr2ds_x = 10**(0.5*(np.log10(dr2ds_x[1:])+np.log10(dr2ds_x[:-1])))

# npi.group_by(dcontour).mean(dr2)
dr2dcontour_x, dr2dcontour_y = dcontour, dr2
_, bins = np.histogram(np.log10(dr2dcontour_x), bins='auto')
dr2dcontour_y, dr2dcontour_x, bnumbr = stats.binned_statistic(
    dr2dcontour_x, np.log10(dr2dcontour_y), statistic=np.nanmean, bins=10**bins)
dr2dcontour_y = 10**dr2dcontour_y
dr2dcontour_x = 10**(0.5 *
                     (np.log10(dr2dcontour_x[1:])+np.log10(dr2dcontour_x[:-1]))) """

dTau2dt_x, dTau2dt_y = npi.group_by(dt).mean(dTau2)  # dt, dTau2
dTau2dt_x_f, dTau2dt_y_f = dTau2dt_x, dTau2dt_y
_, bins = np.histogram(np.log10(dTau2dt_x), bins='auto')
dTau2dt_y, dTau2dt_x, bnumbr = stats.binned_statistic(
    dTau2dt_x, np.log10(dTau2dt_y), statistic=np.nanmean, bins=10**bins)
dTau2dt_y = 10**dTau2dt_y
dTau2dt_x = 10**(0.5*(np.log10(dTau2dt_x[1:])+np.log10(dTau2dt_x[:-1])))

for k in range(dTau_d.shape[1]):
    dTau_d_dt_x, y = npi.group_by(dt).var(dTau_d[:,k])  # dt, dTau2
    try:
        dTau_d_dt_y
    except NameError:
        dTau_d_dt_y = y
    else:
        dTau_d_dt_y += y
dTau_d_dt_x_f, dTau_d_dt_y_f = dTau_d_dt_x, dTau_d_dt_y
_, bins = np.histogram(np.log10(dTau_d_dt_x), bins='auto')
dTau_d_dt_y, dTau_d_dt_x, bnumbr = stats.binned_statistic(
    dTau_d_dt_x, np.log10(dTau_d_dt_y), statistic=np.nanmean, bins=10**bins)
dTau_d_dt_y = 10**dTau_d_dt_y
dTau_d_dt_x = 10**(0.5*(np.log10(dTau_d_dt_x[1:])+np.log10(dTau_d_dt_x[:-1])))

""" dr2dt_x, dr2dt_y = npi.group_by(dt).mean(dr2)  # dt, dr2
dr2dt_x, dr2dt_y = dr2dt_x, dr2dt_y
_, bins = np.histogram(np.log10(dr2dt_x), bins='auto')
dr2dt_y, dr2dt_x, bnumbr = stats.binned_statistic(
    dr2dt_x, np.log10(dr2dt_y), statistic=np.nanmean, bins=10**bins)
dr2dt_y = 10**dr2dt_y
dr2dt_x = 10**(0.5*(np.log10(dr2dt_x[1:])+np.log10(dr2dt_x[:-1]))) """

dsdt_x, dsdt_y = npi.group_by(dt).mean(ds)  # dt, ds
dsdt_x, dsdt_y = dsdt_x, dsdt_y
_, bins = np.histogram(np.log10(dsdt_x), bins='auto')
dsdt_y, dsdt_x, bnumbr = stats.binned_statistic(
    dsdt_x, np.log10(dsdt_y), statistic=np.nanmean, bins=10**bins)
dsdt_y = 10**dsdt_y
dsdt_x = 10**(0.5*(np.log10(dsdt_x[1:])+np.log10(dsdt_x[:-1])))

dcontourdt_x, dcontourdt_y = npi.group_by(dt).mean(dcontour)
dcontourdt_x, dcontourdt_y = dcontourdt_x, dcontourdt_y
_, bins = np.histogram(np.log10(dcontourdt_x), bins='auto')
dcontourdt_y, dcontourdt_x, bnumbr = stats.binned_statistic(
    dcontourdt_x, np.log10(dcontourdt_y), statistic=np.nanmean, bins=10**bins)
dcontourdt_y = 10**dcontourdt_y
dcontourdt_x = 10**(0.5 *
                    (np.log10(dcontourdt_x[1:])+np.log10(dcontourdt_x[:-1])))

ddisplacement2dt_x, ddisplacement2dt_y = npi.group_by(
    dt).mean(ddisplacement2)
# dt, ddisplacement2
ddisplacement2dt_x_f, ddisplacement2dt_y_f = ddisplacement2dt_x, ddisplacement2dt_y
_, bins = np.histogram(np.log10(ddisplacement2dt_x), bins='auto')
ddisplacement2dt_y, ddisplacement2dt_x, bnumbr = stats.binned_statistic(
    ddisplacement2dt_x, np.log10(ddisplacement2dt_y), statistic=np.nanmean, bins=10**bins)
ddisplacement2dt_y = 10**ddisplacement2dt_y
ddisplacement2dt_x = 10**(0.5 *
                          (np.log10(ddisplacement2dt_x[1:])+np.log10(ddisplacement2dt_x[:-1])))

# npi.group_by(ds).mean(ddisplacement2)
ddisplacement2ds_x, ddisplacement2ds_y = ds, ddisplacement2
_, bins = np.histogram(np.log10(ddisplacement2ds_x), bins='auto')
ddisplacement2ds_y, ddisplacement2ds_x, bnumbr = stats.binned_statistic(
    ddisplacement2ds_x, np.log10(ddisplacement2ds_y), statistic=np.nanmean, bins=10**bins)
ddisplacement2ds_y = 10**ddisplacement2ds_y
ddisplacement2ds_x = 10**(0.5 *
                          (np.log10(ddisplacement2ds_x[1:])+np.log10(ddisplacement2ds_x[:-1])))

# npi.group_by(dcontour).mean(ddisplacement2)
ddisplacement2dcontour_x, ddisplacement2dcontour_y = dcontour, ddisplacement2
_, bins = np.histogram(np.log10(ddisplacement2dcontour_x), bins='auto')
ddisplacement2dcontour_y, ddisplacement2dcontour_x, bnumbr = stats.binned_statistic(
    ddisplacement2dcontour_x, np.log10(ddisplacement2dcontour_y), statistic=np.nanmean, bins=10**bins)
ddisplacement2dcontour_y = 10**ddisplacement2dcontour_y
ddisplacement2dcontour_x = 0.5 * \
    (ddisplacement2dcontour_x[1:]+ddisplacement2dcontour_x[:-1])

ddisplacement2dcontour_dN_x, ddisplacement2dcontour_dN_y = dcontour, ddisplacement_dN
_, bins = np.histogram(ddisplacement2dcontour_dN_x, bins='auto')
ddisplacement2dcontour_dN_y, ddisplacement2dcontour_dN_x, bnumbr = stats.binned_statistic(
    ddisplacement2dcontour_dN_x, ddisplacement2dcontour_dN_y, statistic=d_var, bins=10**bins)
ddisplacement2dcontour_dN_x = 0.5 * \
    (ddisplacement2dcontour_dN_x[1:]+ddisplacement2dcontour_dN_x[:-1])
_, bins = np.histogram(np.log10(ddisplacement2dcontour_dN_x), bins='auto')
ddisplacement2dcontour_dN_y, ddisplacement2dcontour_dN_x, bnumbr = stats.binned_statistic(
    ddisplacement2dcontour_dN_x, np.log10(ddisplacement2dcontour_dN_y), statistic=np.nanmean, bins=10**bins)
ddisplacement2dcontour_dN_y = 10**ddisplacement2dcontour_dN_y
ddisplacement2dcontour_dN_x = 10**(0.5 * \
    (np.log10(ddisplacement2dcontour_dN_x[1:])+np.log10(ddisplacement2dcontour_dN_x[:-1])))

ddisplacement2ds_dN_x, ddisplacement2ds_dN_y = ds, ddisplacement_dN
_, bins = np.histogram(np.log10(ddisplacement2ds_dN_x), bins='auto')
ddisplacement2ds_dN_y, ddisplacement2ds_dN_x, bnumbr = stats.binned_statistic(
    ddisplacement2ds_dN_x, np.log10(ddisplacement2ds_dN_y), statistic=d_var, bins=10**bins)
ddisplacement2ds_dN_x = 0.5 * \
    (ddisplacement2ds_dN_x[1:]+ddisplacement2ds_dN_x[:-1])
_, bins = np.histogram(np.log10(ddisplacement2ds_dN_x), bins='auto')
ddisplacement2ds_dN_y, ddisplacement2ds_dN_x, bnumbr = stats.binned_statistic(
    ddisplacement2ds_dN_x, np.log10(ddisplacement2ds_dN_y), statistic=np.nanmean, bins=10**bins)
ddisplacement2ds_dN_y = 10**ddisplacement2ds_dN_y
ddisplacement2ds_dN_x = 10**(0.5 * \
    (np.log10(ddisplacement2ds_dN_x[1:])+np.log10(ddisplacement2ds_dN_x[:-1])))

for k in range(ddisplacement_dN.shape[1]):
    ddisplacement2dt_dN_x, y = npi.group_by(dt).var(ddisplacement_dN[:,k])  # dt, dTau2
    try:
        ddisplacement2dt_dN_y
    except NameError:
        ddisplacement2dt_dN_y = y
    else:
        ddisplacement2dt_dN_y += y
ddisplacement2dt_dN_x_f, ddisplacement2dt_dN_y_f = ddisplacement2dt_dN_x, ddisplacement2dt_dN_y
_, bins = np.histogram(np.log10(ddisplacement2dt_dN_x), bins='auto')
ddisplacement2dt_dN_y, ddisplacement2dt_dN_x, bnumbr = stats.binned_statistic(
    ddisplacement2dt_dN_x, np.log10(ddisplacement2dt_dN_y), statistic=np.nanmean, bins=10**bins)
ddisplacement2dt_dN_y = 10**ddisplacement2dt_dN_y
ddisplacement2dt_dN_x = 0.5 * \
    (ddisplacement2dt_dN_x[1:]+ddisplacement2dt_dN_x[:-1])

f = pd.DataFrame()
""" f = pd.DataFrame({"dr2ds_ds": dr2ds_x, "dr2ds_dr2": dr2ds_y})
f_1 = pd.DataFrame({"dr2dcontour_dcontour": dr2dcontour_x,
                    "dr2dcontour_dr2": dr2dcontour_y})
f = pd.concat([f, f_1], axis=1) """
f_1 = pd.DataFrame({"dTau2dt_dt_f": dTau2dt_x_f,
                    "dTau2dt_dTau2_f": dTau2dt_y_f})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dTau2dt_dt": dTau2dt_x, "dTau2dt_dTau2": dTau2dt_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dTau_d_dt_dt_f": dTau_d_dt_x_f,
                    "dTau_d_dt_dTau_d_f": dTau_d_dt_y_f})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dTau_d_dt_dt": dTau_d_dt_x, "dTau_d_dt_dTau_d_": dTau_d_dt_y})
f = pd.concat([f, f_1], axis=1)
""" f_1 = pd.DataFrame({"dr2dt_dt": dr2dt_x, "dr2dt_dr2": dr2dt_y})
f = pd.concat([f, f_1], axis=1) """
f_1 = pd.DataFrame({"dsdt_dt": dsdt_x, "dsdt_ds": dsdt_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dcontourdt_dt": dcontourdt_x,
                    "dcontourdt_dcontour": dcontourdt_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dt_dt_f": ddisplacement2dt_x_f,
                    "ddisplacement2dt_ddisplacement2_f": ddisplacement2dt_y_f})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dt_dt": ddisplacement2dt_x,
                    "ddisplacement2dt_ddisplacement2": ddisplacement2dt_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2ds_ds": ddisplacement2ds_x,
                    "ddisplacement2ds_ddisplacement2": ddisplacement2ds_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dcontour_dcontour": ddisplacement2dcontour_x,
                    "ddisplacement2dcontour_ddisplacement2": ddisplacement2dcontour_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dcontour_dN_dcontour": ddisplacement2dcontour_dN_x,
                    "ddisplacement2dcontour_dN_ddisplacement2": ddisplacement2dcontour_dN_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2ds_dN_ds": ddisplacement2ds_dN_x,
                    "ddisplacement2ds_dN_ddisplacement2": ddisplacement2ds_dN_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dcontour_dN_dcontour": ddisplacement2dcontour_dN_x,
                    "ddisplacement2dcontour_dN_ddisplacement2": ddisplacement2dcontour_dN_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dt_dN_dt": ddisplacement2dt_dN_x,
                    "ddisplacement2dt_dN_ddisplacement2": ddisplacement2dt_dN_y})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dt_dN_dt_f": ddisplacement2dt_dN_x_f,
                    "ddisplacement2dt_dN_ddisplacement2_f": ddisplacement2dt_dN_y_f})
f = pd.concat([f, f_1], axis=1)

f.to_csv(join(output_foldername,
              "dr2ds.txt"), index=False, sep='\t', na_rep='nan')

plt.scatter(dt, ds, color='b')
plt.plot(dsdt_x, dsdt_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta s$ [simulation units]")
plt.title(r"$\Delta s$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dsdt.png")
plt.close()

plt.scatter(dt, dcontour, color='b')
plt.plot(dcontourdt_x, dcontourdt_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour$ [simulation units]")
plt.title(r"$\Delta contour$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dcontourdt.png")
plt.close()

plt.scatter(dt, ddisplacement2, color='b')
plt.plot(ddisplacement2dt_x, ddisplacement2dt_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour_c$ [simulation units]")
plt.title(r"$\Delta contour_c$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt.png")
plt.close()

plt.scatter(dt, dTau2, color='b')
plt.plot(dTau2dt_x, dTau2dt_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2dt.png")
plt.close()

""" plt.scatter(ds, dr2, color='b')
plt.plot(dr2ds_x, dr2ds_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2ds.png")
plt.close()

plt.scatter(dcontour, dr2, color='b')
plt.plot(dr2dcontour_x, dr2dcontour_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2dcontour.png")
plt.close()

plt.scatter(dt, dr2, color='b')
plt.plot(dr2dt_x, dr2dt_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2dt.png")
plt.close() """

plt.scatter(ds, ddisplacement2, color='b')
plt.plot(ddisplacement2ds_x, ddisplacement2ds_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2ds.png")
plt.close()

plt.scatter(dcontour, ddisplacement2, color='b')
plt.plot(ddisplacement2dcontour_x, ddisplacement2dcontour_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dcontour.png")
plt.close()

""" plt.scatter(dt, ddisplacement2, color='b')
plt.plot(dr2dt_x, ddisplacement2dt_y, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt.png")
plt.close() """

_, bins = np.histogram(np.log10(ds), bins='auto')
plt.hist(ds, bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\P(\Delta s)$ [simulation units]")
plt.title(r"$\P(\Delta s)$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(s).png")
plt.close()

_, bins = np.histogram(np.log10(dcontour), bins='auto')
plt.hist(dcontour, bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta contour$ [simulation units]")
plt.ylabel(r"$\P(\Delta contour)$ [simulation units]")
plt.title(r"$\P(\Delta contour)$ v/s $\Delta contour$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(contour).png")
plt.close()

_, bins = np.histogram(np.log10(dr2), bins='auto')
plt.hist(dr2, bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta r^2$ [simulation units]")
plt.ylabel(r"$\P(\Delta r^2)$ [simulation units]")
plt.title(r"$\P(\Delta r^2)$ v/s $\Delta r^2$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(r2).png")
plt.close()

_, bins = np.histogram(np.log10(ddisplacement2), bins='auto')
plt.hist(ddisplacement2, bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.ylabel(r"$\P(\Delta {r_c}^2)$ [simulation units]")
plt.title(r"$\P(\Delta {r_c}^2)$ v/s $\Delta {r_c}^2$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(displacement2).png")
plt.close()


dd2_t_i = np.array([item for sublist in ddisplacement2_t_i
                               [np.around(dt, 12) == 1].to_list() for item in sublist])
R = np.array([item for sublist in R
              [np.around(dt, 12) == 1].to_list() for item in sublist])

R = R[~np.isnan(R)]
dd2_t_i = dd2_t_i[~np.isnan(dd2_t_i)]

R = R[np.nonzero(dd2_t_i)]
dd2_t_i = dd2_t_i[np.nonzero(dd2_t_i)]

_, bins = np.histogram(np.log10(R), bins='auto')
y, x, _ = stats.binned_statistic(
    R, np.log10(dd2_t_i), statistic=np.mean, bins=10**bins)
y = 10**y
x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))

plt.plot(x, y)
plt.xscale('log')
plt.yscale('log')
#plt.ylim((ymin, ymax))
plt.xlabel("R (radius) [simulation units]")
plt.ylabel(r"${\Delta r}^ 2$ [simulation units]")
plt.title(r"${\Delta r}^ 2$ v/s R [simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2_rv_sR_1.png")
plt.close()

dd2_i = []
for k in range(d):
    dd2_i.append([item for sublist in ddisplacement2_i["dx_i_" + str(k+1)]
                             [np.around(dt, 12) == 1].to_list() for item in sublist])

dd2_i = np.array(
    [item for sublist in dd2_i for item in sublist])
R = np.array(d*[item for sublist in R
                [np.around(dt, 12) == 1].to_list() for item in sublist])

R = R[~np.isnan(R)]
dd2_i = dd2_i[~np.isnan(dd2_i)]

R = R[np.nonzero(dd2_i)]
dd2_i = dd2_i[np.nonzero(dd2_i)]

_, bins = np.histogram(np.log10(R), bins='auto')
y, x, _ = stats.binned_statistic(
    R, np.log10(dd2_i), statistic=np.mean, bins=10**bins)
y = 10**y
x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))

plt.plot(x, y)
plt.xscale('log')
plt.yscale('log')
#plt.ylim((ymin, ymax))
plt.xlabel("R (radius) [simulation units]")
plt.ylabel(r"${\Delta r}^ 2$ [simulation units]")
plt.title(r"${\Delta r}^ 2$ v/s R [simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2_rv_sR.png")
plt.close()

for delt in dt_range:
    dd2_i = []
    for k in range(d):
        dd2_i.append([item for sublist in ddisplacement2_i["dx_i_" + str(k+1)]
                                 [np.around(dt, 12) == delt].to_list() for item in sublist])
    dd2_i = np.array(
        [item for sublist in ddisplacement2_i for item in sublist])
    R = np.array(d*[item for sublist in R
                    [np.around(dt, 12) == delt].to_list() for item in sublist])
    R = R[~np.isnan(R)]
    dd2_i = dd2_i[~np.isnan(dd2_i)]

    R = R[np.nonzero(dd2_i)]
    dd2_i = dd2_i[np.nonzero(dd2_i)]

    _, bins = np.histogram(np.log10(R), bins='auto')
    y, x, _ = stats.binned_statistic(
        R, np.log10(dd2_i), statistic=np.mean, bins=10**bins)
    y = 10**y
    x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))
    plt.scatter(x, y, label=r"$\tau$ = " + str(dt))
plt.xlabel("R (radius) [simulation units]")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="best")
plt.ylabel(r"${\Delta r}^ 2$ [simulation units]")
plt.title(r"${\Delta r}^ 2$ v/s R [simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2_rv_sR_dt.png")
plt.close()

dd2_i = []
for k in range(d):
    dd2_i.append(
        [item for sublist in ddisplacement2_i["dx_i_" + str(k+1)].to_list() for item in sublist])
dd2_i = np.array(
    [item for sublist in dd2_i for item in sublist])
R = np.array(d*[item for sublist in R.to_list()
                for item in sublist])

R = R[~np.isnan(R)]
dd2_i = dd2_i[~np.isnan(dd2_i)]

R = R[np.nonzero(dd2_i)]
dd2_i = dd2_i[np.nonzero(dd2_i)]

_, bins = np.histogram(np.log10(R), bins='auto')
y, x, _ = stats.binned_statistic(
    R, np.log10(dd2_i), statistic=np.mean, bins=10**bins)
y = 10**y
x = 10**(0.5*(np.log10(x[1:])+np.log10(x[:-1])))

plt.plot(x, y)
plt.xscale('log')
plt.yscale('log')
#plt.ylim((ymin, ymax))
plt.xlabel("R (radius) [simulation units]")
plt.ylabel(r"${\Delta r}^ 2$ [simulation units]")
plt.title(r"${\Delta r}^ 2$ v/s R [simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2_rv_sR.png")
plt.close()

print("--- %s seconds ---" % (time.time() - start_time))