'''
File: dr2ds_creator.py
Project: Q_analysis
File Created: Friday, 3rd May 2019 4:28:03 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Saturday, 20th March 2021 6:17:33 pm
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

NUMBER_OF_QUENCHES = int(
    f.loc[f["name"] == "NUMBER_OF_QUENCHES"]["value"].values[0])
utils_print_frequency = int(
    f.loc[f["name"] == "utils_print_frequency"]["value"].values[0])

U = pd.read_csv(join(output_foldername, U_filename), sep="\n",
                header=None, skip_blank_lines=False, skiprows=1)
U = U[0].str.split("\s+", expand=True)

if (type_quenching == 3):
    U.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU/U",
                 "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max", "U_max", "F_max", "kbT"]
elif (type_quenching == 4):
    U.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU/U",
                 "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max", "U_max", "F_max",  "bias_U", "number of biases"]
else:
    U.columns = ["counter_system", "t", "counter", "counter_time", "status_flag", "U", "dU", "dU/U",
                 "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max", "U_max"]

utils_collection = {}

dr2ds_data = pd.DataFrame(columns=[
                          "dt", "dcounter", "R", "dcontour", "ds"] + ["dx_i_" + str(k+1) for k in range(d)] + ["ddisplacement2", "dr2", "dTau2l", "dTau2u"], dtype='f8')

i_file_counter = 0

dt_range = [1, 10, 100]

dynamic_scaling_time = 400

files = [files for root, folders, files in walk(output_foldername)]

for filename in files[0]:
    if utils_filename in filename:

        i_file_counter += 1

        dtype = np.dtype([("system_counter", 'i4'), ("t", 'f8'), ("counter", 'i4'), ("counter_time", 'f8'), ("state_flag", 'f8'), ("N", 'i4'), ("vol_frac", 'f8')] + [("L_box" + str(k+1), 'f8') for k in range(d)] + [("p_idx", 'i4'), ("r_idx", 'i4'), ("flag", 'i4'), ("R", 'f8')] + [("x_" + str(k+1), 'f8') for k in range(d)] + [("dx_" + str(k+1), 'f8') for k in range(d)] + [("deltax_" + str(k+1), 'f8') for k in range(d)] +
                         [("s", 'f8'), ("ds", 'f8'), ("contour", 'f8'), ("dcontour", 'f8'), ("U", 'f8'), ("dU", 'f8'), ("dU_U", 'f8'), ("Z", 'f8')] + [("Tau_" + str(k+1), 'f8') for k in range(d * d)])

        f = open(join(output_foldername, filename), "rb")
        f.seek(0)

        data = np.fromfile(f, dtype=dtype)

        utils_collection[i_file_counter] = pd.DataFrame(
            data, columns=data.dtype.names)

for index, (i, j) in enumerate(list(it.combinations(utils_collection.keys(), 2))):
    if (float(utils_collection[i]["t"][0]) > dynamic_scaling_time) & (float(utils_collection[j]["t"][0]) > dynamic_scaling_time):
        dt = abs(float(utils_collection[j]["t"][0]) -
                 float(utils_collection[i]["t"][0]))

        dcounter = abs(float(utils_collection[j]["counter"][0]) -
                       float(utils_collection[i]["counter"][0]))

        if (output_mode == 5) & (dt == 0.0) & (dcounter > 0):
            if (float(utils_collection[i]["state_flag"][0]) > 1.0) & (float(utils_collection[j]["state_flag"][0]) > 1.0):

                dcontour = abs(float(utils_collection[j]["contour"][0]) -
                               float(utils_collection[i]["contour"][0]))

                ds = dcontour

                ddisplacement2_i = [[np.nan] for k in range(d)]
                R = [[np.nan]]

                ddisplacement2 = sum(sum([np.power(np.multiply(utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                    k+1)], ((utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1))), 2) for k in range(d)]))

                dr2 = sum(sum([np.power(np.multiply(periodic_BC(utils_collection[j]["x_"+str(k+1)], utils_collection[i]["x_"+str(k+1)], np.nanmean([utils_collection[j]["L_box"+str(k+1)][0], utils_collection[i]["L_box"+str(k+1)][0]])),
                                                    ((utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1))), 2) for k in range(d)]))

                dTau2l = sum(sum([np.power(np.multiply(utils_collection[j]["Tau_"+str(k+1)] - utils_collection[i]["Tau_"+str(k+1)],
                                                       ((utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1))), 2) for k in [d, 2*d, 2*d+1]]))

                dTau2u = sum(np.power([sum(np.multiply(utils_collection[j]["Tau_"+str(k+1)], (utils_collection[j]["flag"] != -1))) - sum(
                    np.multiply(utils_collection[i]["Tau_"+str(k+1)], (utils_collection[i]["flag"] != -1))) for k in [1, d-1, 2*d-1]], 2))

                dr2ds_data.loc[dr2ds_data.shape[0]] = pd.DataFrame(
                    [dt, dcounter] + R + [dcontour] + [ds] + ddisplacement2_i + [ddisplacement2, dr2, dTau2l, dTau2u])

        elif (output_mode == 1) & (dt > 0) & (dcounter == 0):
            if (float(utils_collection[i]["state_flag"][0]) == 0) & (float(utils_collection[j]["state_flag"][0]) == 0):

                dcounter = abs(float(utils_collection[j]["counter"][0]) -
                               float(utils_collection[i]["counter"][0]))

                dcontour = abs(float(utils_collection[j]["contour"][0]) -
                               float(utils_collection[i]["contour"][0]))

                ds = abs(float(utils_collection[j]["s"][0]) -
                         float(utils_collection[i]["s"][0]))

                ddisplacement2_i = [[np.nan] for k in range(d)]
                R = [[np.nan]]

                if (type_quenching == 1) & (dt in dt_range):
                    ddisplacement2_i = []
                    for k in range(d):
                        ddisplacement2_i.append(np.array(np.power((utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                            k+1)])[((utils_collection[i]["flag"] == 1) & (utils_collection[j]["flag"] == 1))], 2)))

                    R = [np.array((utils_collection[j]["R"] + utils_collection[i]["R"])[
                                  ((utils_collection[i]["flag"] == 1) & (utils_collection[j]["flag"] == 1))]) / 2.0]
                    ddisplacement2_i = (
                        np.array(ddisplacement2_i) / np.power(np.nanmean(R), 2)).tolist()
                    R = (np.array(R) / np.nanmean(R)).tolist()

                ddisplacement2 = sum(sum([np.power(np.multiply(utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                    k+1)], ((utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1))), 2) for k in range(d)]))

                dr2 = sum(sum([np.power(np.multiply(periodic_BC(utils_collection[j]["x_"+str(k+1)], utils_collection[i]["x_"+str(k+1)], np.nanmean([utils_collection[j]["L_box"+str(k+1)][0], utils_collection[i]["L_box"+str(k+1)][0]])),
                                                    ((utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1))), 2) for k in range(d)]))

                dTau2l = sum(sum([np.power(np.multiply(utils_collection[j]["Tau_"+str(k+1)] - utils_collection[i]["Tau_"+str(k+1)],
                                                       ((utils_collection[i]["flag"] != -1) & (utils_collection[j]["flag"] != -1))), 2) for k in [d, 2*d, 2*d+1]]))

                dTau2u = sum(np.power([sum(np.multiply(utils_collection[j]["Tau_"+str(k+1)], (utils_collection[j]["flag"] != -1))) - sum(
                    np.multiply(utils_collection[i]["Tau_"+str(k+1)], (utils_collection[i]["flag"] != -1))) for k in [1, d-1, 2*d-1]], 2))

                dr2ds_data.loc[dr2ds_data.shape[0]] = [dt, dcounter] + R + [dcontour] + [
                    ds] + ddisplacement2_i + [ddisplacement2, dr2, dTau2l, dTau2u]

dTau2ldt_x, dTau2ldt_y = npi.group_by(dr2ds_data["dt"]).mean(
    dr2ds_data["dTau2l"])  # dt, dTau2l
_, bins = np.histogram(np.log10(dTau2ldt_x), bins='auto')
dTau2ldt_y, dTau2ldt_x, _ = stats.binned_statistic(
    dTau2ldt_x, np.log10(dTau2ldt_y), statistic=np.mean, bins=10**bins)
dTau2ldt_y = 10**dTau2ldt_y
dTau2ldt_x = 10**(0.5*(np.log10(dTau2ldt_x[1:])+np.log10(dTau2ldt_x[:-1])))

dTau2udt_x, dTau2udt_y = npi.group_by(dr2ds_data["dt"]).mean(
    dr2ds_data["dTau2u"])  # dt, dTau2u
_, bins = np.histogram(np.log10(dTau2udt_x), bins='auto')
dTau2udt_y, dTau2udt_x, _ = stats.binned_statistic(
    dTau2udt_x, np.log10(dTau2udt_y), statistic=np.mean, bins=10**bins)
dTau2udt_y = 10**dTau2udt_y
dTau2udt_x = 10**(0.5*(np.log10(dTau2udt_x[1:])+np.log10(dTau2udt_x[:-1])))

dsdt_x, dsdt_y = npi.group_by(dr2ds_data["dt"]).mean(
    dr2ds_data["ds"])  # dt, ds
_, bins = np.histogram(np.log10(dsdt_x), bins='auto')
dsdt_y, dsdt_x, _ = stats.binned_statistic(
    dsdt_x, np.log10(dsdt_y), statistic=np.mean, bins=10**bins)
dsdt_y = 10**dsdt_y
dsdt_x = 10**(0.5*(np.log10(dsdt_x[1:])+np.log10(dsdt_x[:-1])))

dcontourdt_x, dcontourdt_y = npi.group_by(
    dr2ds_data["dt"]).mean(dr2ds_data["dcontour"])
_, bins = np.histogram(np.log10(dcontourdt_x), bins='auto')
dcontourdt_y, dcontourdt_x, _ = stats.binned_statistic(
    dcontourdt_x, np.log10(dcontourdt_y), statistic=np.mean, bins=10**bins)
dcontourdt_y = 10**dcontourdt_y
dcontourdt_x = 10**(0.5 *
                    (np.log10(dcontourdt_x[1:])+np.log10(dcontourdt_x[:-1])))

dr2dt_x, dr2dt_y = npi.group_by(dr2ds_data["dt"]).mean(
    dr2ds_data["dr2"])  # dt, dr2
_, bins = np.histogram(np.log10(dr2dt_x), bins='auto')
dr2dt_y, dr2dt_x, _ = stats.binned_statistic(
    dr2dt_x, np.log10(dr2dt_y), statistic=np.mean, bins=10**bins)
dr2dt_y = 10**dr2dt_y
dr2dt_x = 10**(0.5*(np.log10(dr2dt_x[1:])+np.log10(dr2dt_x[:-1])))

ddisplacement2dt_x, ddisplacement2dt_y = npi.group_by(
    dr2ds_data["dt"]).mean(dr2ds_data["ddisplacement2"])  # dt, ddisplacement2
_, bins = np.histogram(np.log10(ddisplacement2dt_x), bins='auto')
ddisplacement2dt_y, ddisplacement2dt_x, _ = stats.binned_statistic(
    ddisplacement2dt_x, np.log10(ddisplacement2dt_y), statistic=np.mean, bins=10**bins)
ddisplacement2dt_y = 10**ddisplacement2dt_y
ddisplacement2dt_x = 10**(0.5 *
                          (np.log10(ddisplacement2dt_x[1:])+np.log10(ddisplacement2dt_x[:-1])))

# npi.group_by(ds).mean(dr2)
dr2ds_x, dr2ds_y = dr2ds_data["ds"], dr2ds_data["dr2"]
_, bins = np.histogram(np.log10(dr2ds_x), bins='auto')
dr2ds_y, dr2ds_x, _ = stats.binned_statistic(
    dr2ds_x, np.log10(dr2ds_y), statistic=np.mean, bins=10**bins)
dr2ds_y = 10**dr2ds_y
dr2ds_x = 10**(0.5*(np.log10(dr2ds_x[1:])+np.log10(dr2ds_x[:-1])))

# npi.group_by(dcontour).mean(dr2)
dr2dcontour_x, dr2dcontour_y = dr2ds_data["dcontour"], dr2ds_data["dr2"]
_, bins = np.histogram(np.log10(dr2dcontour_x), bins='auto')
dr2dcontour_y, dr2dcontour_x, _ = stats.binned_statistic(
    dr2dcontour_x, np.log10(dr2dcontour_y), statistic=np.mean, bins=10**bins)
dr2dcontour_y = 10**dr2dcontour_y
dr2dcontour_x = 10**(0.5 *
                     (np.log10(dr2dcontour_x[1:])+np.log10(dr2dcontour_x[:-1])))

# npi.group_by(ds).mean(ddisplacement2)
ddisplacement2ds_x, ddisplacement2ds_y = dr2ds_data["ds"], dr2ds_data["ddisplacement2"]
_, bins = np.histogram(np.log10(ddisplacement2ds_x), bins='auto')
ddisplacement2ds_y, ddisplacement2ds_x, _ = stats.binned_statistic(
    ddisplacement2ds_x, np.log10(ddisplacement2ds_y), statistic=np.mean, bins=10**bins)
ddisplacement2ds_y = 10**ddisplacement2ds_y
ddisplacement2ds_x = 10**(0.5 *
                          (np.log10(ddisplacement2ds_x[1:])+np.log10(ddisplacement2ds_x[:-1])))

# npi.group_by(dcontour).mean(ddisplacement2)
ddisplacement2dcontour_x, ddisplacement2dcontour_y = dr2ds_data[
    "dcontour"], dr2ds_data["ddisplacement2"]
_, bins = np.histogram(np.log10(ddisplacement2dcontour_x), bins='auto')
ddisplacement2dcontour_y, ddisplacement2dcontour_x, _ = stats.binned_statistic(
    ddisplacement2dcontour_x, np.log10(ddisplacement2dcontour_y), statistic=np.mean, bins=10**bins)
ddisplacement2dcontour_y = 10**ddisplacement2dcontour_y
ddisplacement2dcontour_x = 0.5 * \
    (ddisplacement2dcontour_x[1:]+ddisplacement2dcontour_x[:-1])

dr2ds = pd.DataFrame()
dr2ds_temp = pd.DataFrame(
    {"dTau2ldt_dt": dTau2ldt_x, "dTau2ldt_dTau2l": dTau2ldt_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame(
    {"dTau2udt_dt": dTau2udt_x, "dTau2udt_dTau2u": dTau2udt_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"dr2dt_dt": dr2dt_x, "dr2dt_dr2": dr2dt_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"dsdt_dt": dsdt_x, "dsdt_ds": dsdt_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"dcontourdt_dt": dcontourdt_x,
                           "dcontourdt_dcontour": dcontourdt_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"ddisplacement2dt_dt": ddisplacement2dt_x,
                           "ddisplacement2dt_ddisplacement2": ddisplacement2dt_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"dr2ds_ds": dr2ds_x, "dr2ds_dr2": dr2ds_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"dr2dcontour_dcontour": dr2dcontour_x,
                           "dr2dcontour_dr2": dr2dcontour_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"ddisplacement2ds_ds": ddisplacement2ds_x,
                           "ddisplacement2ds_ddisplacement2": ddisplacement2ds_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)
dr2ds_temp = pd.DataFrame({"ddisplacement2dcontour_dcontour": ddisplacement2dcontour_x,
                           "ddisplacement2dcontour_ddisplacement2": ddisplacement2dcontour_y})
dr2ds = pd.concat([dr2ds, dr2ds_temp], axis=1)

dr2ds.to_csv(join(output_foldername,
                  "dr2ds.txt"), index=False, sep='\t', na_rep='nan')

plt.scatter(dr2ds_data["dt"], dr2ds_data["ds"], color='b')
plt.plot(dr2ds["dsdt_dt"], dr2ds["dsdt_ds"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta s$ [simulation units]")
plt.title(r"$\Delta s$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dsdt.png")
plt.show()

plt.scatter(dr2ds_data["dt"], dr2ds_data["dcontour"], color='b')
plt.plot(dr2ds["dcontourdt_dt"], dr2ds["dcontourdt_dcontour"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour$ [simulation units]")
plt.title(r"$\Delta contour$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dcontourdt.png")
plt.show()

plt.scatter(dr2ds_data["dt"], dr2ds_data["ddisplacement2"], color='b')
plt.plot(dr2ds["ddisplacement2dt_dt"],
         dr2ds["ddisplacement2dt_ddisplacement2"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour_c$ [simulation units]")
plt.title(r"$\Delta contour_c$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt.png")
plt.show()

plt.scatter(dr2ds_data["dt"], dr2ds_data["dTau2l"], color='b')
plt.plot(dr2ds["dTau2ldt_dt"], dr2ds["dTau2ldt_dTau2l"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2ldt.png")
plt.show()

plt.scatter(dr2ds_data["dt"], dr2ds_data["dTau2u"], color='b')
plt.plot(dr2ds["dTau2udt_dt"], dr2ds["dTau2udt_dTau2u"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2udt.png")
plt.show()

plt.scatter(dr2ds_data["ds"], dr2ds_data["dr2"], color='b')
plt.plot(dr2ds["dr2ds_ds"], dr2ds["dr2ds_dr2"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2ds.png")
plt.show()

plt.scatter(dr2ds_data["dcontour"], dr2ds_data["dr2"], color='b')
plt.plot(dr2ds["dr2dcontour_dcontour"], dr2ds["dr2dcontour_dr2"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2dcontour.png")
plt.show()

plt.scatter(dr2ds_data["dt"], dr2ds_data["dr2"], color='b')
plt.plot(dr2ds["dr2dt_dt"], dr2ds["dr2dt_dr2"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2dt.png")
plt.show()

plt.scatter(dr2ds_data["ds"], dr2ds_data["ddisplacement2"], color='b')
plt.plot(dr2ds["ddisplacement2ds_ds"],
         dr2ds["ddisplacement2ds_ddisplacement2"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2ds.png")
plt.show()

plt.scatter(dr2ds_data["dcontour"], dr2ds_data["ddisplacement2"], color='b')
plt.plot(dr2ds["ddisplacement2dcontour_dcontour"],
         dr2ds["ddisplacement2dcontour_ddisplacement2"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dcontour.png")
plt.show()

plt.scatter(dr2ds_data["dt"], dr2ds_data["ddisplacement2"], color='b')
plt.plot(dr2ds["dr2dt_dt"], dr2ds["dr2dt_dr2"], color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt.png")
plt.show()

_, bins = np.histogram(np.log10(dr2ds_data["ds"]), bins='auto')
plt.hist(dr2ds_data["ds"], bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\P(\Delta s)$ [simulation units]")
plt.title(r"$\P(\Delta s)$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(s).png")
plt.show()

_, bins = np.histogram(np.log10(dr2ds_data["dcontour"]), bins='auto')
plt.hist(dr2ds_data["dcontour"], bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta contour$ [simulation units]")
plt.ylabel(r"$\P(\Delta contour)$ [simulation units]")
plt.title(r"$\P(\Delta contour)$ v/s $\Delta contour$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(contour).png")
plt.show()

_, bins = np.histogram(np.log10(dr2ds_data["dr2"]), bins='auto')
plt.hist(dr2ds_data["dr2"], bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta r^2$ [simulation units]")
plt.ylabel(r"$\P(\Delta r^2)$ [simulation units]")
plt.title(r"$\P(\Delta r^2)$ v/s $\Delta r^2$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(r2).png")
plt.show()

_, bins = np.histogram(np.log10(dr2ds_data["ddisplacement2"]), bins='auto')
plt.hist(dr2ds_data["ddisplacement2"], bins=10**bins, density=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.ylabel(r"$\P(\Delta {r_c}^2)$ [simulation units]")
plt.title(r"$\P(\Delta {r_c}^2)$ v/s $\Delta {r_c}^2$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(displacement2).png")
plt.show()


ddisplacement2_i = []
for k in range(d):
    ddisplacement2_i.append([item for sublist in dr2ds_data["dx_i_" + str(k+1)]
                             [dr2ds_data["dt"] == 1].to_list() for item in sublist])
ddisplacement2_i = np.array(
    [item for sublist in ddisplacement2_i for item in sublist])
R = np.array(d*[item for sublist in dr2ds_data["R"]
                [dr2ds_data["dt"] == 1].to_list() for item in sublist])

R = R[~np.isnan(R)]
ddisplacement2_i = ddisplacement2_i[~np.isnan(ddisplacement2_i)]

R = R[np.nonzero(ddisplacement2_i)]
ddisplacement2_i = ddisplacement2_i[np.nonzero(ddisplacement2_i)]

_, bins = np.histogram(np.log10(R), bins='auto')
ddisplacement2_i, R, _ = stats.binned_statistic(
    R, np.log10(ddisplacement2_i), statistic=np.mean, bins=10**bins)
ddisplacement2_i = 10**ddisplacement2_i
R = 10**(0.5*(np.log10(R[1:])+np.log10(R[:-1])))

plt.plot(R, ddisplacement2_i)
plt.xscale('log')
plt.yscale('log')
#plt.ylim((ymin, ymax))
plt.xlabel("R (radius) [simulation units]")
plt.ylabel(r"${\Delta r}^ 2$ [simulation units]")
plt.title(r"${\Delta r}^ 2$ v/s R [simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2_rv_sR_dt=1.png")
plt.show()

for dt in dt_range:
    ddisplacement2_i = []
    for k in range(d):
        ddisplacement2_i.append([item for sublist in dr2ds_data["dx_i_" + str(k+1)]
                                 [dr2ds_data["dt"] == dt].to_list() for item in sublist])
    ddisplacement2_i = np.array(
        [item for sublist in ddisplacement2_i for item in sublist])
    R = np.array(d*[item for sublist in dr2ds_data["R"]
                    [dr2ds_data["dt"] == dt].to_list() for item in sublist])
    R = R[~np.isnan(R)]
    ddisplacement2_i = ddisplacement2_i[~np.isnan(ddisplacement2_i)]

    R = R[np.nonzero(ddisplacement2_i)]
    ddisplacement2_i = ddisplacement2_i[np.nonzero(ddisplacement2_i)]

    _, bins = np.histogram(np.log10(R), bins='auto')
    ddisplacement2_i, R, _ = stats.binned_statistic(
        R, np.log10(ddisplacement2_i), statistic=np.mean, bins=10**bins)
    ddisplacement2_i = 10**ddisplacement2_i
    R = 10**(0.5*(np.log10(R[1:])+np.log10(R[:-1])))
    plt.scatter(R, ddisplacement2_i, label=r"$\tau$ = " + str(dt))
plt.xlabel("R (radius) [simulation units]")
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="best")
plt.ylabel(r"${\Delta r}^ 2$ [simulation units]")
plt.title(r"${\Delta r}^ 2$ v/s R [simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2_rv_sR_dt.png")
plt.show()

ddisplacement2_i = []
for k in range(d):
    ddisplacement2_i.append(
        [item for sublist in dr2ds_data["dx_i_" + str(k+1)].to_list() for item in sublist])
ddisplacement2_i = np.array(
    [item for sublist in ddisplacement2_i for item in sublist])
R = np.array(d*[item for sublist in dr2ds_data["R"].to_list()
                for item in sublist])

R = R[~np.isnan(R)]
ddisplacement2_i = ddisplacement2_i[~np.isnan(ddisplacement2_i)]

R = R[np.nonzero(ddisplacement2_i)]
ddisplacement2_i = ddisplacement2_i[np.nonzero(ddisplacement2_i)]

_, bins = np.histogram(np.log10(R), bins='auto')
ddisplacement2_i, R, _ = stats.binned_statistic(
    R, np.log10(ddisplacement2_i), statistic=np.mean, bins=10**bins)
ddisplacement2_i = 10**ddisplacement2_i
R = 10**(0.5*(np.log10(R[1:])+np.log10(R[:-1])))

plt.plot(R, ddisplacement2_i)
plt.xscale('log')
plt.yscale('log')
#plt.ylim((ymin, ymax))
plt.xlabel("R (radius) [simulation units]")
plt.ylabel(r"${\Delta r}^ 2$ [simulation units]")
plt.title(r"${\Delta r}^ 2$ v/s R [simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2_rv_sR_all.png")
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
