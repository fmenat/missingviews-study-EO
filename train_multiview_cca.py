import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from cca_zoo.linear import CCA, MCCA

from src.training.utils import preprocess_views
from src.training.learn_pipeline import MultiFusion_train, assign_multifusion_name
from src.training.utils import get_loss_by_name
from src.datasets.views_structure import load_structure, DataViews
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
        loss_name = "ce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    elif config_file.get("task_type", "").lower() == "regression":
        loss_name = "mse" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    method_name = "CCA-"+assign_multifusion_name(config_file["training"],config_file["method"], more_info_str=config_file.get("additional_method_name", ""))
    run_id_mlflow = None 
    metadata_r = { "full_prediction_time": []}
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
            method, trainer = MultiFusion_train(train_data, val_data=val_data,run_id=r,fold_id=k,method_name=method_name, 
                                                            run_id_mlflow = run_id_mlflow, **config_file)
            mlf_logger, run_id_mlflow = trainer.loggers[0], trainer.loggers[0].run_id 
            mlf_logger.experiment.log_dict(run_id_mlflow, raw_dims, "original_data_dim.yaml")
            mlf_logger.experiment.log_dict(run_id_mlflow, config_file, "config_file.yaml")
            print("Training encoder done")

            #store original predictions to calculate error-difference
            pred_time_Start = time.time()
            outputs_te = method.transform(_to_loader(val_data, batch_size=BS, train=False), out_norm=(config_file.get("task_type", "").lower() != "regression"))            
            metadata_r["full_prediction_time"].append(time.time()-pred_time_Start)
            data_save_te = DataViews([outputs_te["prediction"]], identifiers=val_data["identifiers"], view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
            data_save_te.save(f"{output_dir_folder}/pred/{data_name}/test/{method_name}", ind_views=True, xarray=False)
            mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/pred/{data_name}/test/{method_name}/out_run-{r:02d}_fold-{k:02d}.csv",
                                            artifact_path="preds/test")

            ## TRAIN CCA
            print("Training CCA...", end="")
            outputs_tr = method.transform(_to_loader(train_data, batch_size=BS, train=False), args_forward={"forward_only_representation":True}) 

            database_4_lookup = list(outputs_tr["views:rep"].values()) #list of database for each view
            dims_embed = list(method.get_embeddings_size().values())
            if type(config_file["training"]["cca_config"].get("latent_dimensions"))== float:
                config_file["training"]["cca_config"]["latent_dimensions"] = int(np.sum(dims_embed)*config_file["training"]["cca_config"].get("latent_dimensions"))
            config_file["training"]["cca_config"]["latent_dimensions"] = np.minimum(config_file["training"]["cca_config"]["latent_dimensions"], np.min(dims_embed))
            model_search = MCCA(**config_file["training"]["cca_config"])
            std_model = {v: StandardScaler() for v in outputs_tr["views:rep"].keys()}
            embedding_4_lookup = model_search.fit_transform([ f.fit_transform(d) for f, d in zip(list(std_model.values()), database_4_lookup)])
            database_names = np.asarray(list(outputs_tr["views:rep"].keys()))
            print("Done!")

            for test_views in config_file["args_forward"].get("list_testing_views"):
                print("Inference with the following views ",test_views)

                pred_time_Start = time.time()
                #BUILD PREDICTION DATA
                val_data_missing = {"identifiers": val_data["identifiers"], "view_names": [], "views": []}
                missing_view = []
                for view_data_i, view_name_i in zip(val_data["views"], val_data["view_names"]):
                    if view_name_i in test_views:
                        val_data_missing["views"].append(view_data_i)
                        val_data_missing["view_names"].append(view_name_i)
                    else:
                        missing_view.append(view_name_i)
                #FORWARD REPRESENTATION FROM AVAILABLE VIEWS
                data_forward = method.transform(_to_loader(val_data_missing, batch_size=config_file['args_forward'].get("batch_size", BS), train=False),
                                                              args_forward={"forward_only_representation":True})
                val_data_missing["views"] = list(data_forward["views:rep"].values())
                val_data_missing["view_names"] = list(data_forward["views:rep"].keys())

                #EMBED REPRESENTATION OF AVAILABLE VIEW TO CCA
                view_data_to_search = val_data_missing["views"][-1]
                query_view_cca = model_search.transform([std_model[val_data_missing["view_names"][-1]].transform(view_data_to_search)])[-1] #embed to CCA the view used to search

                for m_view in missing_view: #SEARCH SIMILAR WITH CCA embedding
                    print(f"View {m_view} is missing, searching a similar with CCA over view {val_data_missing['view_names'][-1]} in database")
                                        
                    idx_view_missing = np.where(database_names == m_view)[0][0] #find which view is the one to search for in database

                    indx_most_similar = cdist(query_view_cca, embedding_4_lookup[idx_view_missing], **config_file["training"]["search_config"]).argmax(axis=1) #search index for most similar

                    val_data_missing["views"].append(database_4_lookup[idx_view_missing][indx_most_similar])
                    val_data_missing["view_names"].append(m_view)
            
                outputs_te = method.transform(_to_loader(val_data_missing, batch_size=config_file['args_forward'].get("batch_size", BS), train=False), 
                                              out_norm=(config_file.get("task_type", "").lower() != "regression"), args_forward={"forward_from_representation":True})
                if f"{'_'.join(test_views)}_prediction_time" not in metadata_r:
                    metadata_r[f"{'_'.join(test_views)}_prediction_time"] = []
                metadata_r[f"{'_'.join(test_views)}_prediction_time"].append(time.time()-pred_time_Start)

                aux_name = "CCA-"+assign_multifusion_name(config_file["training"],config_file["method"], forward_views=test_views, 
                                                more_info_str=config_file.get("additional_method_name", ""))
                ## STORE PREDICTIONS ##
                data_save_te = DataViews([outputs_te["prediction"]], identifiers=val_data["identifiers"], view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                data_save_te.save(f"{output_dir_folder}/pred/{data_name}/test/{aux_name}", ind_views=True, xarray=False)
                mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/pred/{data_name}/test/{aux_name}/out_run-{r:02d}_fold-{k:02d}.csv",
                                                artifact_path="preds/test")

                print(f"Fold {k+1}/{kfolds} of Run {r+1}/{runs} in {aux_name} finished...")
            print(f"Fold {k+1}/{kfolds} of Run {r+1}/{runs} in {method_name} finished...")
    if type(run_id_mlflow) != type(None):
        Path(f"{output_dir_folder}/metadata/{data_name}/{method_name}").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metadata_r).to_csv(f"{output_dir_folder}/metadata/{data_name}/{method_name}/metadata_runs.csv")
        mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/metadata/{data_name}/{method_name}/metadata_runs.csv",)
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
