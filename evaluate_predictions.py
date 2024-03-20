import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
matplotlib.rc('font', **{"size":14})

from src.datasets.views_structure import DataViews, load_structure
from src.metrics.metrics import ClassificationMetrics, SoftClassificationMetrics, RegressionMetrics
from src.visualizations.utils import save_results, gt_mask
from src.visualizations.tools import plot_prob_dist_bin, plot_conf_matrix, plot_dist_bin,plot_true_vs_pred
from src.evaluate.utils import load_data_sup, average_ensemble

def classification_metric(
                preds_p_run,
                indexs_p_run,
                data_ground_truth,
                ind_save,
                plot_runs = False,
                include_metrics = [],
                dir_folder = "",
                task_type="classification",
                ):
    R = len(preds_p_run)

    df_runs = []
    df_runs_diss = []
    for r in range(R):
        y_true, y_pred = gt_mask(data_ground_truth, indexs_p_run[r]), preds_p_run[r]
        y_true = np.squeeze(y_true)
        y_pred_cont = np.squeeze(y_pred)

        if task_type == "classification":
            y_pred_cont = y_pred_cont[y_true != -1]
            y_true = y_true[y_true != -1]

            y_pred_no_missing = np.argmax(y_pred_cont, axis = -1)

            d_me = ClassificationMetrics()
            dic_res = d_me(y_pred_no_missing, y_true)
        else:
            d_me = RegressionMetrics()
            dic_res = d_me(y_pred_cont, y_true)


        if task_type == "classification":
            d_me_aux = SoftClassificationMetrics()
            dic_res.update(d_me_aux(y_pred_cont, y_true))
            
            d_me = ClassificationMetrics(["F1 none", "R none", "P none", "ntrue", 'npred'])
            dic_des = d_me(y_pred_no_missing, y_true)
            df_des = pd.DataFrame(dic_des)
            df_des.index = ["label-"+str(i) for i in range(len(dic_des["N TRUE"]))]
            df_runs_diss.append(df_des)

            d_me = ClassificationMetrics(["confusion"])
            cf_matrix = d_me(y_pred_no_missing, y_true)["MATRIX"]        
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), squeeze=False)
            plot_conf_matrix(ax[0,0], cf_matrix, "test set")
            save_results(f"{dir_folder}/plots/{ind_save}/preds_r{r:02d}_conf", plt)
            plt.close()
        
            if "f1 bin" in include_metrics:
                dic_res["F1 bin"] = dic_des["F1 NONE"][1]
            if "p bin" in include_metrics:
                dic_res["P bin"] = dic_des["P NONE"][1]
        
            if len(d_me.n_samples) == 2:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3), squeeze=False)
                plot_prob_dist_bin(ax[0,0], y_pred_cont, y_true, f"(run-{r})")
                save_results(f"{dir_folder}/plots/{ind_save}/preds_proba_r{r:02d}", plt)
                plt.close()

        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3), squeeze=False)
            plot_dist_bin(ax[0,0], y_pred_cont, y_true, f"(run-{r})")
            save_results(f"{dir_folder}/plots/{ind_save}/preds_r{r:02d}", plt)
            plt.close()
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5), squeeze=False)
            plot_true_vs_pred(ax[0,0], y_pred_cont, y_true, f"(run-{r})")
            save_results(f"{dir_folder}/plots/{ind_save}/preds_vs_ground_r{r:02d}", plt)
            plt.close()

        df_res = pd.DataFrame(dic_res, index=["test"])

        if plot_runs:
            print(f"Run {r} being shown")
            print(df_res.round(4).to_markdown())
        df_runs.append(df_res)
   
    df_concat = pd.concat(df_runs).groupby(level=0)
    df_mean = df_concat.mean()
    df_std = df_concat.std()

    save_results(f"{dir_folder}/plots/{ind_save}/preds_mean", df_mean)
    save_results(f"{dir_folder}/plots/{ind_save}/preds_std", df_std)
    print(f"################ Showing the {ind_save} ################")
    print(df_mean.round(4).to_markdown())
    print(df_std.round(4).to_markdown())

    if task_type == "classification":
        df_concat_diss = pd.concat(df_runs_diss).groupby(level=0)
        df_mean_diss = df_concat_diss.mean()
        df_std_diss = df_concat_diss.std()

        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_mean", df_mean_diss)
        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_std", df_std_diss)
        print(df_mean_diss.round(4).to_markdown())

    return df_mean,df_std

def calculate_metrics(df_summary, df_std, data_te,data_name, method, task_type="classification", **args):
    preds_p_run_te, indexs_p_run_te = load_data_sup(data_name+"/test", method, **args)

    df_aux, df_aux2= classification_metric(
                        preds_p_run_te,
                        indexs_p_run_te,
                        data_te,
                        ind_save=f"{data_name}/{method}/",
                        task_type = task_type,
                        **args
                        )
    df_summary[method] = df_aux.loc["test"]
    df_std[method] = df_aux2.loc["test"]

def ensemble_avg(method_names, df_summary, df_std, data_te,data_name, method="EnsembleAVG", view_experiments=[],task_type="classification", **args):
    if len(view_experiments) == 0:
        preds_p_run_te,indexs_p_run_te= average_ensemble(data_name, method_names, **args)
        df_aux, df_aux2= classification_metric(
                            preds_p_run_te,
                            indexs_p_run_te,
                            data_te,
                            f"{data_name}/{method}/",
                            task_type=task_type,
                            **args
                            )
        df_summary[method] = df_aux.loc["test"]
        df_std[method] = df_aux2.loc["test"]
    else:
        for view_experiment in view_experiments:
            preds_p_run_te, indexs_p_run_te = average_ensemble(data_name, method_names, pivot=view_experiment, **args)
            print("inference with",view_experiment)
            
            df_aux, df_aux2= classification_metric(
                                preds_p_run_te,
                                indexs_p_run_te,
                                data_te,
                                f"{data_name}/{method}-Forw_{view_experiment}",
                                task_type=task_type,
                                **args
                                )
            df_summary[method+f"-Forw_{view_experiment}"] = df_aux.loc["test"]
            df_std[method+f"-Forw_{view_experiment}"] = df_aux2.loc["test"]

def main_evaluation(config_file):
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    include_metrics = ["f1 bin", "p bin"]

    data_tr = load_structure(f"{input_dir_folder}/{data_name}.nc")

    if config_file.get("methods_to_plot"):
        methods_to_plot = config_file["methods_to_plot"]
    else:
        methods_to_plot = sorted(os.listdir(f"{output_dir_folder}/pred/{data_name}/test"))

    df_summary_sup, df_summary_sup_s = pd.DataFrame(), pd.DataFrame()
    pool_names = {}
    missview_methods = {}
    for method in methods_to_plot:
        print(f"Evaluating method {method}")
        calculate_metrics(df_summary_sup, df_summary_sup_s,
                        data_tr, 
                        data_name,
                        method,
                        include_metrics=include_metrics,
                        plot_runs=config_file.get("plot_runs"),
                        dir_folder=output_dir_folder,
                        task_type = config_file.get("task_type", "classification"),
                        )
        if "-Forw" in method:
            m_fullview = "-".join([i for i in method.split("-") if "Forw" not in i])
            if m_fullview not in missview_methods:
                missview_methods[m_fullview] = []
            missview_methods[m_fullview].append(method)
        if method.lower().startswith("pool"):
            key_pool = method.lower().split("_")[0].split("pool")[-1]
            if key_pool =="":
                key_pool = "_"
            if key_pool in pool_names:
                pool_names[key_pool].append(method)
            else:
                pool_names[key_pool] = [method]
    
    view_experiments = [v.split("-Forw_")[-1].split("-")[0] for v in list(missview_methods.values())[0]]
    if len(pool_names) != 0:
        for key_pool in pool_names:
            ensemble_avg(pool_names[key_pool], df_summary_sup, df_summary_sup_s, data_tr,data_name,
                    include_metrics=include_metrics,
                    plot_runs=config_file.get("plot_runs"),
                    dir_folder=output_dir_folder,
                    method="EnsembleAVG"+key_pool,
                    task_type = config_file.get("task_type", "classification"),
                    )
            ensemble_avg(pool_names[key_pool], df_summary_sup, df_summary_sup_s, data_tr,data_name,
                    include_metrics=include_metrics,
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
    df_summary_sup.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_mean.csv")
    df_summary_sup_s.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_std.csv")

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
