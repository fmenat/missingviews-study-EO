import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
matplotlib.rc('font', **{"size":14})

from src.datasets.views_structure import DataViews, load_structure
from src.metrics.metrics import RobustnessMetrics
from src.visualizations.utils import save_results, gt_mask
from src.evaluate.utils import load_data_sup, average_ensemble

def robustness_metrics(
                preds_p_run,
                indexs_p_run,
                data_ground_truth,
                preds_p_run_te_ref,
                indexs_p_run_te_ref ,
                ind_save,
                plot_runs = False,
                dir_folder = "",
                task_type="classification",
                **kwargs
                ):
    R = len(preds_p_run)

    df_runs = []
    df_runs_diss = []
    for r in range(R):
        
        indx_to_position = {value: i for i,value in enumerate(indexs_p_run[r])}
        mask_index = np.asarray([ indx_to_position[i] for i in indexs_p_run_te_ref[r]])
        y_pred_prob, y_pred_noise_prob, y_true= preds_p_run[r], preds_p_run_te_ref[r][mask_index], gt_mask(data_ground_truth, indexs_p_run[r])
        
        if task_type == "nonneg_regression" or task_type == "classification":
            y_true = np.squeeze(y_true)
            y_pred_prob = y_pred_prob[y_true != -1]
            y_pred_noise_prob = y_pred_noise_prob[y_true != -1]
            y_true = y_true[y_true != -1]

        if task_type=="classification" and len(y_true.shape) == 1:
            y_true = OneHotEncoder(sparse=False).fit_transform(y_true.reshape(-1,1))

        d_me = RobustnessMetrics(task_type=task_type)
        dic_res = d_me(y_pred_prob, y_pred_noise_prob, y_true)
        df_res = pd.DataFrame(dic_res, index=["test"])

        d_me = RobustnessMetrics(["PRS none", "PRS_MAE none", "PRS_diff none", "DRS none", "DRS_MAE none", "DRS_diff none"])
        dic_des = d_me(y_pred_prob, y_pred_noise_prob, y_true)
        df_des = pd.DataFrame(dic_des, index=["label-"+str(i) for i in range(len(dic_des["PRS"]))])
        df_runs_diss.append(df_des)

        if plot_runs:
            print(f"Run {r} being shown")
            print(df_res.round(4).to_markdown())
        df_runs.append(df_res)

    df_concat = pd.concat(df_runs).groupby(level=0)
    df_mean = df_concat.mean()
    df_std = df_concat.std()

    save_results(f"{dir_folder}/plots/{ind_save}/robustness_mean", df_mean)
    save_results(f"{dir_folder}/plots/{ind_save}/robustness_std", df_std)
    print(f"################ Showing the {ind_save} ################")
    print(df_mean.round(4).to_markdown())
    print(df_std.round(4).to_markdown())

    df_concat_diss = pd.concat(df_runs_diss).groupby(level=0)
    df_mean_diss = df_concat_diss.mean()
    df_std_diss = df_concat_diss.std()

    save_results(f"{dir_folder}/plots/{ind_save}/robustness_ind_mean", df_mean_diss)
    save_results(f"{dir_folder}/plots/{ind_save}/robustness_ind_std", df_std_diss)
    print(df_mean_diss.round(4).to_markdown())

    return df_mean,df_std

def calculate_metrics(df_summary, df_std, data_te,data_name, method, missview_methods, task_type, **args):
    preds_p_run_te, indexs_p_run_te = load_data_sup(data_name+"/test", method, **args)

    for missview_method in missview_methods[method]:
        print("with respect to",missview_method)
        preds_p_run_te_ref, indexs_p_run_te_ref = load_data_sup(data_name+"/test", missview_method, **args)

        df_aux, df_aux2= robustness_metrics(
                            preds_p_run_te,
                            indexs_p_run_te,
                            data_te,
                            preds_p_run_te_ref,
                            indexs_p_run_te_ref,
                            ind_save=f"{data_name}/{method}_x_{missview_method}/",
                            task_type=task_type,
                            **args
                            )
        df_summary[method+"_x_"+missview_method.split("-Forw_")[-1].split("-")[0]] = df_aux.loc["test"]
        df_std[method+"_x_"+missview_method.split("-Forw_")[-1].split("-")[0]] = df_aux2.loc["test"]

def ensemble_avg(method_names, df_summary, df_std, data_te,data_name, method="EnsembleAVG", view_experiments=[], task_type="classification", **args):
    preds_p_run_te,indexs_p_run_te= average_ensemble(data_name, method_names, **args)
    
    for view_experiment in view_experiments:
        preds_p_run_te_ref, _= average_ensemble(data_name, method_names, pivot=view_experiment, **args)
        print("with respect to",view_experiment)
        
        df_aux, df_aux2= robustness_metrics(
                            preds_p_run_te,
                            indexs_p_run_te,
                            data_te,
                            preds_p_run_te_ref,
                            indexs_p_run_te,
                            ind_save=f"{data_name}/{method}_x_{view_experiment}/",
                            task_type=task_type,
                            **args
                            )
        df_summary[method+f"_x_{view_experiment}"] = df_aux.loc["test"]
        df_std[method+f"_x_{view_experiment}"] = df_aux2.loc["test"]

def main_evaluation(config_file):
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]

    data_tr = load_structure(f"{input_dir_folder}/{data_name}.nc")

    if config_file.get("methods_to_plot"):
        methods_to_plot = config_file["methods_to_plot"]
    else:
        methods_to_plot = sorted(os.listdir(f"{output_dir_folder}/pred/{data_name}/test"))

    fullview_methods = []
    missview_methods = {}
    pool_names = {}
    for m in methods_to_plot:
        if "-Forw" in m:
            m_fullview = "-".join([i for i in m.split("-") if "Forw" not in i])
            if m_fullview not in fullview_methods:
                missview_methods[m_fullview] = []
                fullview_methods.append(m_fullview)
            missview_methods[m_fullview].append(m)

        if m.lower().startswith("pool"):
            key_pool = m.lower().split("_")[0].split("pool")[-1]
            if key_pool =="":
                key_pool = "_"
            if key_pool in pool_names:
                pool_names[key_pool].append(m)
            else:
                pool_names[key_pool] = [m]
    
    df_summary_sup, df_summary_sup_s = pd.DataFrame(), pd.DataFrame()
    for method in fullview_methods:
        print(f"Evaluating method {method}")
        calculate_metrics(df_summary_sup, df_summary_sup_s,
                        data_tr, 
                        data_name=data_name,
                        method=method, 
                        missview_methods=missview_methods,
                        dir_folder=output_dir_folder,
                        task_type = config_file.get("task_type", "classification"),
                        )
        
    view_experiments = [v.split("-Forw_")[-1].split("-")[0] for v in list(missview_methods.values())[0]]
    if len(pool_names) != 0:
        for key_pool in pool_names:
            print(f"Evaluating Ensemble with {key_pool}")
            ensemble_avg(pool_names[key_pool], df_summary_sup, df_summary_sup_s, data_tr,
                         data_name=data_name,
                    plot_runs=config_file.get("plot_runs"),
                    dir_folder=output_dir_folder,
                    method="EnsembleAVG"+key_pool,
                    view_experiments=view_experiments,
                    task_type = config_file.get("task_type", "classification"),
                    )

    #all figures were saved in output_dir_folder/plots
    print(">>>>>>>>>>>>>>>>> Mean across runs on test set")
    print((df_summary_sup.T).round(4).to_markdown())
    print(">>>>>>>>>>>>>>>>> Std across runs on test set")
    print((df_summary_sup_s.T).round(4).to_markdown())
    df_summary_sup.T.to_csv(f"{output_dir_folder}/plots/{data_name}/robustness_mean.csv")
    df_summary_sup_s.T.to_csv(f"{output_dir_folder}/plots/{data_name}/robustness_std.csv")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)

    main_evaluation(config_file)
