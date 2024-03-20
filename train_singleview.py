import yaml
import argparse
import os
import sys
import time
import gc
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

from src.training.utils import preprocess_views
from src.training.learn_pipeline import InputFusion_train
from src.training.utils import get_loss_by_name
from src.datasets.views_structure import DataViews, load_structure
from src.datasets.utils import _to_loader

def main_run(config_file):
    start_time = time.time()
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    view_names = config_file["view_names"]
    runs = config_file["experiment"].get("runs", 1)
    kfolds = config_file["experiment"].get("kfolds", 2)
    preprocess_args = config_file["experiment"]["preprocess"]
    mlflow_runs_exp = config_file["experiment"]["mlflow_runs_exp"]
    BS = config_file["training"]["batch_size"]
    method_name = f"Input"
    if config_file.get("additional_method_name"):
        method_name += config_file.get("additional_method_name")
    
    try:
        data_views_all = load_structure(f"{input_dir_folder}/{data_name}_input")
    except:
        data_views_all = load_structure(f"{input_dir_folder}/{data_name}")
    raw_dims = {}
    for view_name in data_views_all.get_view_names():
        aux_data = data_views_all.get_view_data(view_name)["views"]
        raw_dims[view_name] = {"raw": list(aux_data.shape[1:]), "flatten": int(np.prod(aux_data.shape[1:]))}
    if "input_views" not in preprocess_args:
        preprocess_args["input_views"] = view_names
    preprocess_views(data_views_all, **preprocess_args)

    indexs_ = data_views_all.get_all_identifiers() 
    
    if "loss_args" not in config_file["training"]: 
        config_file["training"]["loss_args"] = {}
    if config_file.get("task_type", "").lower() == "classification":
        loss_name = "ce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"].pop("name")
    elif config_file.get("task_type", "").lower() == "regression":
        loss_name = "mse" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"].pop("name")
    run_id_mlflow = None 
    metadata_r = {"epoch_runs":[], "full_prediction_time":[], "training_time":[], "best_score":[] }
    for r in range(runs):
        if config_file["experiment"].get("group"): #stratified cross-validation
            name_group = config_file["experiment"].get("group")
            values_to_random_group = data_views_all.get_view_data(name_group)["views"]
            uniques_values_to_random_group = np.unique(values_to_random_group)
            np.random.shuffle(uniques_values_to_random_group)
            stratified_values_runs = np.array_split(uniques_values_to_random_group , kfolds)
            indexs_runs = []
            for stratified_values_r in stratified_values_runs:
                indxs_r = []
                for ind_i, value_i in zip(indexs_, values_to_random_group):
                    if value_i in stratified_values_r:
                        indxs_r.append(ind_i)
                indexs_runs.append(indxs_r)
        else: #regular random cross-validation
            np.random.shuffle(indexs_)
            indexs_runs = np.array_split(indexs_, kfolds)
        for k in range(kfolds):
            if mlflow_runs_exp:
                run_id_mlflow = "ind"
            print(f"******************************** Executing model on run {r+1} and kfold {k+1}")
            
            data_views_all.set_test_mask(indexs_runs[k], reset=True)

            train_data = data_views_all.generate_full_view_data(train = True, views_first=True, view_names=view_names)
            val_data = data_views_all.generate_full_view_data(train = False, views_first=True, view_names=view_names)
            print(f"Training with {len(train_data['identifiers'])} samples and validating on {len(val_data['identifiers'])}")

            if config_file.get("task_type", "").lower() == "classification":
                train_data_target = train_data["target"].astype(int).flatten()
                weight_=class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(train_data_target), y=train_data_target)
                config_file["training"]["loss_args"]["weight"] = torch.tensor(weight_,dtype=torch.float)
                N_LABELS = len(weight_)
            else:
                N_LABELS = 1
            config_file["training"]["loss_function"] = get_loss_by_name(loss_name, **config_file["training"]["loss_args"])
            config_file["architecture"]["n_labels"] = N_LABELS
            
            start_aux = time.time()
            method, trainer = InputFusion_train(train_data, val_data=val_data, run_id=r,fold_id=k,method_name=method_name ,
                                                                        run_id_mlflow = run_id_mlflow, **config_file)
            mlf_logger, run_id_mlflow = trainer.loggers[0], trainer.loggers[0].run_id 
            mlf_logger.experiment.log_dict(run_id_mlflow, raw_dims, "original_data_dim.yaml")
            mlf_logger.experiment.log_dict(run_id_mlflow, config_file, "config_file.yaml")
            metadata_r["training_time"].append(time.time()-start_aux)
            metadata_r["epoch_runs"].append(trainer.callbacks[0].stopped_epoch)
            metadata_r["best_score"].append(trainer.callbacks[0].best_score.cpu().numpy())
            print("Training done")

            ### STORE ORIGINAL predictions
            pred_time_Start = time.time()
            outputs_tr = method.transform(_to_loader(train_data, batch_size=BS, train=False), out_norm=(config_file.get("task_type", "").lower() != "regression"))    
            outputs_te = method.transform(_to_loader(val_data, batch_size=BS, train=False), out_norm=(config_file.get("task_type", "").lower() != "regression"))
            metadata_r["full_prediction_time"].append(time.time()-pred_time_Start)
            
            data_save_tr = DataViews([outputs_tr["prediction"]], identifiers=train_data["identifiers"], view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
            data_save_tr.save(f"{output_dir_folder}/pred/{data_name}/train/{method_name}", ind_views=True,xarray=False)
            mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/pred/{data_name}/train/{method_name}/out_run-{r:02d}_fold-{k:02d}.csv",
                                            artifact_path="preds/train")
            
            data_save_te = DataViews([outputs_te["prediction"]], identifiers=val_data["identifiers"], view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
            data_save_te.save(f"{output_dir_folder}/pred/{data_name}/test/{method_name}", ind_views=True,xarray=False)
            mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/pred/{data_name}/test/{method_name}/out_run-{r:02d}_fold-{k:02d}.csv",
                                            artifact_path=f"preds/test")
            
            if config_file.get("args_forward") and config_file["args_forward"].get("list_testing_views"): 
                for test_views in config_file["args_forward"].get("list_testing_views"):
                    print("Inference with the following views ",test_views)
                    args_forward = {"inference_views":test_views}
                    if "missing_method" in config_file["args_forward"]:
                        args_forward = {**args_forward, **{k:v for k,v in config_file["args_forward"].items() if k!= "list_testing_views"}}
                    
                    pred_time_Start = time.time()
                    outputs_te = method.transform(_to_loader(val_data, batch_size=config_file['args_forward'].get("batch_size", BS), train=False), out_norm=(config_file.get("task_type", "").lower() != "regression"), args_forward=args_forward)          
                    if f"{'_'.join(test_views)}_prediction_time" not in metadata_r:
                        metadata_r[f"{'_'.join(test_views)}_prediction_time"] = []
                    metadata_r[f"{'_'.join(test_views)}_prediction_time"].append(time.time()-pred_time_Start)

                    aux_name = f"Input"
                    aux_name += "-Forw_" + "_".join(test_views) #common names should be in a util file
                    if config_file.get("additional_method_name"):
                        aux_name += config_file.get("additional_method_name")
                    data_save_te = DataViews([outputs_te["prediction"]], identifiers=val_data["identifiers"], view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                    data_save_te.save(f"{output_dir_folder}/pred/{data_name}/test/{aux_name}", ind_views=True, xarray=False)
                    mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/pred/{data_name}/test/{aux_name}/out_run-{r:02d}_fold-{k:02d}.csv",
                                                artifact_path="preds/test")
                    print(f"Fold {k+1}/{kfolds} of Run {r+1}/{runs} in {aux_name} finished...")
            print(f"Fold {k+1}/{kfolds} of Run {r+1}/{runs} in {method_name} finished...")
    if type(run_id_mlflow) != type(None):
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_tr_time", np.mean(metadata_r["training_time"]))
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_pred_time", np.mean(metadata_r["full_prediction_time"]))
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_epoch_runs", np.mean(metadata_r["epoch_runs"]))
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_best_score", np.mean(metadata_r["best_score"]))
        Path(f"{output_dir_folder}/metadata/{data_name}/{method_name}").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metadata_r).to_csv(f"{output_dir_folder}/metadata/{data_name}/{method_name}/metadata_runs.csv")
        mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/metadata/{data_name}/{method_name}/metadata_runs.csv",)
        #os.remove(f"{output_dir_folder}/metadata_runs.csv")
    print("Epochs for %s runs on average for %.2f epochs +- %.3f"%(method_name,np.mean(metadata_r["epoch_runs"]),np.std(metadata_r["epoch_runs"])))
    print(f"Finished whole execution of {runs} runs in {time.time()-start_time:.2f} secs")    

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
    
    main_run(config_file)
