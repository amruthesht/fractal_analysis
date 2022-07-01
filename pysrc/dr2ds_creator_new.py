'''
File: dr2ds_creator_new.py
Project: fractal_analysis
File Created: Friday, 3rd May 2019 4:28:03 pm
Author: Amruthesh T (amru@seas.upenn.edu)
-----
Last Modified: Thursday, 30th June 2022 7:22:10 pm
Modified By: Amruthesh T (amru@seas.upenn.edu)
-----
Copyright (c) 2018 - 2019 Amru, University of Pennsylvania

Summary: Fractal path analysis - \Delta r^2 vs. \Delta s (dr2ds) for the Quenching module - new version
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

output_foldername = "output/"
graphs_foldername = "graphs/"
input_foldername = "input/"
input_filename = "init.input"
run_filename = "run_config.txt"

lower_fit_lower_limit = 1e-2
lower_fit_upper_limit = 1e0

upper_fit_lower_limit = 1e-2
upper_fit_upper_limit = np.inf

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

NUMBER_OF_EQB_STEPS = int(
    float(f.loc[f["name"] == "NUMBER_OF_EQB_STEPS"]["value"].values[0]))

NUMBER_OF_QUENCHES = int(
    f.loc[f["name"] == "NUMBER_OF_QUENCHES"]["value"].values[0])
utils_print_frequency = int(
    f.loc[f["name"] == "utils_print_frequency"]["value"].values[0])

U = pd.read_csv(join(output_foldername, U_filename), sep="\n",
                header=None, skip_blank_lines=False, skiprows=1 + 2*NUMBER_OF_EQB_STEPS)
U = U[0].str.split("\s+", expand=True)

NUMBER_OF_COUNTERS = 1

if (type_quenching == 3):
    if (output_mode == 5):
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max", "kbT"]
    else:
        U.columns = ["counter_system", "t", "counter", "counter_time", "state_flag", "U", "dU", "dU/U",
                     "N", "min_counter_SUCCESSFUL", "min_counter", "U_max", "F_max", "kbT"]
elif (type_quenching == 4):
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

        if (type_quenching == 4):
            utils_collection[i_file_counter]["number of biases"] = U.loc[(U["t"] == utils_collection[i_file_counter]["t"][0]) & (
                U["counter"] == utils_collection[i_file_counter]["counter"][0]) & (U["state_flag"].astype(float).astype(int) == 0)]["number of biases"][0]

dt = np.array([])
dcounter = np.array([])
ds = np.array([])
dcontour = np.array([])
ddisplacement2 = np.array([])
dr2 = np.array([])
dTau2 = np.array([])

for index, (i, j) in enumerate(list(it.combinations(utils_collection.keys(), 2))):
    delt = abs(float(utils_collection[j]["t"][0]) -
                                float(utils_collection[i]["t"][0]))

    delc = abs(abs(float(utils_collection[j]["counter"][0])) -
                            abs(float(utils_collection[i]["counter"][0])))

    if (output_mode == 5) & (delt == 0.0) & (delc > 0):
        if (float(utils_collection[i]["state_flag"][0]) > 1.0) & (float(utils_collection[j]["state_flag"][0]) > 1.0):

            dcounter = np.append(dcounter, delc)

            dcontour = np.append(dcontour, abs(abs(float(utils_collection[j]["contour"][0])) -
                            abs(float(utils_collection[i]["contour"][0]))))

            ddisplacement2 = np.append(ddisplacement2, sum(sum([np.power(np.multiply(utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                k+1)], np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in range(d)])))

            dr2 = np.append(dr2, sum(sum([np.power(np.multiply(periodic_BC(utils_collection[j]["x_"+str(k+1)], utils_collection[i]["x_"+str(k+1)], np.mean([utils_collection[j]["L_box"+str(k+1)][0], utils_collection[i]["L_box"+str(k+1)][0]])),
                                                    np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in range(d)])))

            dTau2 = np.append(dTau2, sum(sum([np.power(np.multiply(utils_collection[j]["Tau_"+str(k+1)] - utils_collection[i]["Tau_"+str(k+1)],
                                                        np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in [d, 2*d, 2*d+1]])))

    elif (output_mode == 1) & (delt > 0) & (delc == 0):
        if (float(utils_collection[i]["state_flag"][0]) < 2.0) & (float(utils_collection[j]["state_flag"][0]) < 2.0):

            dt = np.append(dt, delt)

            dcounter = np.append(dcounter, abs(float(utils_collection[j]["counter"][0]) -
                                float(utils_collection[i]["counter"][0])))

            dcontour = np.append(dcontour, abs(abs(float(utils_collection[j]["contour"][0])) -
                            abs(float(utils_collection[i]["contour"][0]))))

            ds = np.append(ds, abs(abs(float(utils_collection[j]["s"][0])) -
                            abs(float(utils_collection[i]["s"][0]))))

            ddisplacement2 = np.append(ddisplacement2, sum(sum([np.power(np.multiply(utils_collection[j]["dx_"+str(k+1)] - utils_collection[i]["dx_"+str(
                k+1)], np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in range(d)])))

            dr2 = np.append(dr2, sum(sum([np.power(np.multiply(periodic_BC(utils_collection[j]["x_"+str(k+1)], utils_collection[i]["x_"+str(k+1)], np.mean([utils_collection[j]["L_box"+str(k+1)][0], utils_collection[i]["L_box"+str(k+1)][0]])),
                                                    np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in range(d)])))

            dTau2 = np.append(dTau2, sum(sum([np.power(np.multiply(utils_collection[j]["Tau_"+str(k+1)] - utils_collection[i]["Tau_"+str(k+1)],
                                                        np.multiply(utils_collection[i]["flag"] != -1, utils_collection[j]["flag"] != -1)), 2) for k in [d, 2*d, 2*d+1]])))

if (output_mode == 5):
    dt = dcounter
    ds = dcontour

dr2ds_x, dr2ds_y = ds, dr2 #npi.group_by(ds).mean(dr2)
_, bins = np.histogram(np.log10(ds), bins='auto')
dr2dsmean, dr2dsbin, bnumbr = stats.binned_statistic(
    dr2ds_x, dr2ds_y, statistic='mean', bins=10**bins)
dr2dscenter = 0.5*(dr2dsbin[1:]+dr2dsbin[:-1])

dr2dcontour_x, dr2dcontour_y = dcontour, dr2 #npi.group_by(dcontour).mean(dr2)
_, bins = np.histogram(np.log10(dcontour), bins='auto')
dr2dcontourmean, dr2dcontourbin, bnumbr = stats.binned_statistic(
    dr2dcontour_x, dr2dcontour_y, statistic='mean', bins=10**bins)
dr2dcontourcenter = 0.5*(dr2dcontourbin[1:]+dr2dcontourbin[:-1])

dTau2dt_x, dTau2dt_y = dt, dTau2 #npi.group_by(dt).mean(dTau2)
_, bins = np.histogram(np.log10(dt), bins='auto')
dTau2dtmean, dTau2dtbin, bnumbr = stats.binned_statistic(
    dTau2dt_x, dTau2dt_y, statistic='mean', bins=10**bins)
dTau2dtcenter = 0.5*(dTau2dtbin[1:]+dTau2dtbin[:-1])

dr2dt_x, dr2dt_y = dt, dr2 #npi.group_by(dt).mean(dr2)
_, bins = np.histogram(np.log10(dt), bins='auto')
dr2dtmean, dr2dtbin, bnumbr = stats.binned_statistic(
    dr2dt_x, dr2dt_y, statistic='mean', bins=10**bins)
dr2dtcenter = 0.5*(dr2dtbin[1:]+dr2dtbin[:-1])

dsdt_x, dsdt_y = dt, ds #npi.group_by(dt).mean(ds)
_, bins = np.histogram(np.log10(dt), bins='auto')
dsdtmean, dsdtbin, bnumbr = stats.binned_statistic(
    dsdt_x, dsdt_y, statistic='mean', bins=10**bins)
dsdtcenter = 0.5*(dsdtbin[1:]+dsdtbin[:-1])

dcontourdt_x, dcontourdt_y = dt, dcontour #npi.group_by(dt).mean(dcontour)
_, bins = np.histogram(np.log10(dt), bins='auto')
dcontourdtmean, dcontourdtbin, bnumbr = stats.binned_statistic(
    dcontourdt_x, dcontourdt_y, statistic='mean', bins=10**bins)
dcontourdtcenter = 0.5*(dcontourdtbin[1:]+dcontourdtbin[:-1])

ddisplacement2dt_x, ddisplacement2dt_y = dt, ddisplacement2 #npi.group_by(dt).mean(ddisplacement2)
_, bins = np.histogram(np.log10(dt), bins='auto')
ddisplacement2dtmean, ddisplacement2dtbin, bnumbr = stats.binned_statistic(
    ddisplacement2dt_x, ddisplacement2dt_y, statistic='mean', bins=10**bins)
ddisplacement2dtcenter = 0.5*(ddisplacement2dtbin[1:]+ddisplacement2dtbin[:-1])

ddisplacement2ds_x, ddisplacement2ds_y = ds, ddisplacement2 #npi.group_by(ds).mean(ddisplacement2)
_, bins = np.histogram(np.log10(ds), bins='auto')
ddisplacement2dsmean, ddisplacement2dsbin, bnumbr = stats.binned_statistic(
    ddisplacement2ds_x, ddisplacement2ds_y, statistic='mean', bins=10**bins)
ddisplacement2dscenter = 0.5*(ddisplacement2dsbin[1:]+ddisplacement2dsbin[:-1])

ddisplacement2dcontour_x, ddisplacement2dcontour_y = dcontour, ddisplacement2 #npi.group_by(dcontour).mean(ddisplacement2)
_, bins = np.histogram(np.log10(dcontour), bins='auto')
ddisplacement2dcontourmean, ddisplacement2dcontourbin, bnumbr = stats.binned_statistic(
    ddisplacement2dcontour_x, ddisplacement2dcontour_y, statistic='mean', bins=10**bins)
ddisplacement2dcontourcenter = 0.5 * \
    (ddisplacement2dcontourbin[1:]+ddisplacement2dcontourbin[:-1])

f = pd.DataFrame({"dr2ds_ds": dr2dscenter, "dr2ds_dr2": dr2dsmean})
f_1 = pd.DataFrame({"dr2dcontour_dcontour": dr2dcontourcenter,
                    "dr2dcontour_dr2": dr2dcontourmean})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dTau2dt_dt": dTau2dtcenter, "dTau2dt_dTau2": dTau2dtmean})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dr2dt_dt": dr2dtcenter, "dr2dt_dr2": dr2dtmean})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dsdt_dt": dsdtcenter, "dsdt_ds": dsdtmean})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dcontourdt_dt": dcontourdtcenter,
                    "dcontourdt_dcontour": dcontourdtmean})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dt_dt": ddisplacement2dtcenter,
                    "ddisplacement2dt_ddisplacement2": ddisplacement2dtmean})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"dr2ds_ds": dr2dscenter, "dr2ds_dr2": dr2dsmean})
f = pd.concat([f, f_1], axis=1)
f_1 = pd.DataFrame({"ddisplacement2dcontour_dcontour": ddisplacement2dcontourcenter,
                    "ddisplacement2dcontour_ddisplacement2": ddisplacement2dcontourmean})
f = pd.concat([f, f_1], axis=1)

f.to_csv(join(output_foldername,
              "dr2ds.txt"), index=False, sep='\t', na_rep='nan')

plt.scatter(dt, ds, color='b')
plt.plot(dsdtcenter, dsdtmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta s$ [simulation units]")
plt.title(r"$\Delta s$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dsdt.png")
plt.show()
plt.close()

plt.scatter(dt, dcontour, color='b')
plt.plot(dcontourdtcenter, dcontourdtmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour$ [simulation units]")
plt.title(r"$\Delta contour$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dcontourdt.png")
plt.show()
plt.close()

plt.scatter(dt, ddisplacement2, color='b')
plt.plot(ddisplacement2dtcenter, ddisplacement2dtmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta contour_c$ [simulation units]")
plt.title(r"$\Delta contour_c$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt.png")
plt.show()
plt.close()

plt.scatter(dt, dTau2, color='b')
plt.plot(dTau2dtcenter, dTau2dtmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta \sigma^2$ [simulation units]")
plt.title(r"$\Delta \sigma^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dTau2dt.png")
plt.show()
plt.close()

plt.scatter(ds, dr2, color='b')
plt.plot(dr2dscenter, dr2dsmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2ds.png")
plt.show()
plt.close()

plt.scatter(dcontour, dr2, color='b')
plt.plot(dr2dcontourcenter, dr2dcontourmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2dcontour.png")
plt.show()
plt.close()

plt.scatter(dt, dr2, color='b')
plt.plot(dr2dtcenter, dr2dtmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta r^2$ [simulation units]")
plt.title(r"$\Delta r^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "dr2dt.png")
plt.show()
plt.close()

plt.scatter(ds, ddisplacement2, color='b')
plt.plot(ddisplacement2dscenter, ddisplacement2dsmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2ds.png")
plt.show()
plt.close()

x1 = ddisplacement2dcontourcenter[(ddisplacement2dcontourcenter > lower_fit_lower_limit) & (ddisplacement2dcontourcenter < lower_fit_upper_limit)]
y1 = ddisplacement2dcontourmean[(ddisplacement2dcontourcenter > lower_fit_lower_limit) & (ddisplacement2dcontourcenter < lower_fit_upper_limit)]

p1, r1, _, _, _ = np.polyfit(np.log10(x1), np.log10(y1), 1, full=True)
print(p1)
print(r1)
p1 = np.poly1d(p1)

slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y1)
print(r_value**2)

x2 = ddisplacement2dcontourcenter[(ddisplacement2dcontourcenter > lower_fit_upper_limit) & (ddisplacement2dcontourcenter > upper_fit_upper_limit)]
y2 = ddisplacement2dcontourmean[(ddisplacement2dcontourcenter > lower_fit_upper_limit) & (ddisplacement2dcontourcenter > upper_fit_upper_limit)]

p2, r2, _, _, _ = np.polyfit(np.log10(x2), np.log10(y2), 1, full=True)
print(p2)
print(r2)
p2 = np.poly1d(p2)

slope, intercept, r_value, p_value, std_err = stats.linregress(x2, y2)
print(r_value**2)

plt.scatter(dcontour, ddisplacement2, label = 'simulation data')
plt.plot(ddisplacement2dcontourcenter, ddisplacement2dcontourmean, color='r', label = 'smoothened data')
plt.plot(x1, 10**p1(np.log10(x1)), ls='--', color='k', label = 'fit')
plt.plot(x2, 10**p2(np.log10(x2)), ls='--', color='k')
plt.legend(loc='best')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\Delta {r}^2$ [simulation units]")
plt.title(r"$\Delta {r}^2$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dcontour.png")
plt.show()
plt.close()

plt.scatter(dt, ddisplacement2, color='b')
plt.plot(dr2dtcenter, ddisplacement2dtmean, color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta t$ [simulation units]")
plt.ylabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.title(r"$\Delta {r_c}^2$ v/s $\Delta t$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "ddisplacement2dt.png")
plt.show()
plt.close()

_, bins = np.histogram(np.log10(ds), bins='auto')
plt.hist(ds, bins=10**bins)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta s$ [simulation units]")
plt.ylabel(r"$\P(\Delta s)$ [simulation units]")
plt.title(r"$\P(\Delta s)$ v/s $\Delta s$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(s).png")
plt.show()
plt.close()

_, bins = np.histogram(np.log10(dcontour), bins='auto')
plt.hist(dcontour, bins=10**bins)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta contour$ [simulation units]")
plt.ylabel(r"$\P(\Delta contour)$ [simulation units]")
plt.title(r"$\P(\Delta contour)$ v/s $\Delta contour$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(contour).png")
plt.show()
plt.close()

_, bins = np.histogram(np.log10(dr2), bins='auto')
plt.hist(dr2, bins=10**bins)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta r^2$ [simulation units]")
plt.ylabel(r"$\P(\Delta r^2)$ [simulation units]")
plt.title(r"$\P(\Delta r^2)$ v/s $\Delta r^2$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(r2).png")
plt.show()
plt.close()

_, bins = np.histogram(np.log10(ddisplacement2), bins='auto')
plt.hist(ddisplacement2, bins=10**bins)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\Delta {r_c}^2$ [simulation units]")
plt.ylabel(r"$\P(\Delta {r_c}^2)$ [simulation units]")
plt.title(r"$\P(\Delta {r_c}^2)$ v/s $\Delta {r_c}^2$[simulation units]")
plt.savefig(output_foldername + graphs_foldername +
            "P(displacement2).png")
plt.show()
plt.close()