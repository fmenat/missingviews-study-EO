import shutil, os, sys, gc, time
from typing import List, Union, Dict
from pathlib import Path
import copy
import numpy as np

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
import pytorch_lightning as pl

from mvlearning.fusion import InputFusion, SingleViewPool, FeatureFusion, FeatureFusionMultiLoss, DecisionFusion, DecisionFusionMultiLoss, HybridFusion_FD
from mvlearning.single.models import create_model
from mvlearning.merge_module import MergeModule
from mvlearning.utils import get_dic_emb_dims

from src.datasets.utils import _to_loader


def prepare_loggers(data_name, method_name, run_id, fold_id, folder_c, tags_ml, run_id_mlflow,monitor_name, **early_stop_args):
    save_dir_tsboard = f'{folder_c}/tensorboard_logs/'
    save_dir_chkpt = f'{folder_c}/checkpoint_logs/'
    save_dir_mlflow = f'{folder_c}/mlruns/'
    exp_folder_name = f'{data_name}/{method_name}'

    for v in Path(f'{save_dir_chkpt}/{exp_folder_name}/').glob(f'r={run_id:02d}_{fold_id:02d}*'):
        v.unlink()
    if os.path.exists(f'{save_dir_tsboard}/{exp_folder_name}/version_{run_id:02d}_{fold_id:02d}'):
        shutil.rmtree(f'{save_dir_tsboard}/{exp_folder_name}/version_{run_id:02d}_{fold_id:02d}')
    early_stop_callback = EarlyStopping(monitor=monitor_name, **early_stop_args)
    tensorlogg = TensorBoardLogger(name="", save_dir=f'{save_dir_tsboard}/{exp_folder_name}/')
    checkpoint_callback = ModelCheckpoint(monitor=monitor_name, mode=early_stop_args["mode"], every_n_epochs=1, save_top_k=1,
        dirpath=f'{save_dir_chkpt}/{exp_folder_name}/', filename=f'r={run_id:02d}_{fold_id:02d}-'+'{epoch}-{step}-{val_objective:.2f}')
    tags_ml = dict(tags_ml,**{"data_name":data_name,"method_name":method_name})
    if run_id_mlflow == "ind":
        mlf_logger = MLFlowLogger(experiment_name=exp_folder_name, run_name = f"version-{run_id:02d}_{fold_id:02d}",
                              tags = tags_ml, tracking_uri=f"file:{save_dir_mlflow}")
    else:
        mlf_logger = MLFlowLogger(experiment_name=data_name, run_name = method_name,
                              run_id= run_id_mlflow,
                              tags = tags_ml, tracking_uri=f"file:{save_dir_mlflow}")
    return {"callbacks": [early_stop_callback,checkpoint_callback], "loggers":[mlf_logger,tensorlogg]}

def log_additional_mlflow(mlflow_model, trainer, model, architecture):
    mlflow_model.experiment.log_artifact(mlflow_model.run_id, trainer.checkpoint_callback.best_model_path, artifact_path="models")
    mlflow_model.experiment.log_text(mlflow_model.run_id, str(model), "models/model_summary.txt")
    mlflow_model.experiment.log_param(mlflow_model.run_id, "type_fusion", model.where_fusion)
    mlflow_model.experiment.log_param(mlflow_model.run_id, "feature_pool", model.feature_pool)
    if model.where_fusion =="feature":
        mlflow_model.experiment.log_param(mlflow_model.run_id, "joint_dim", model.merge_module.get_info_dims()["joint_dim"] )
        
def InputFusion_train(train_data: dict, val_data = None,
                data_name="", method_name="", run_id=0, fold_id=0, output_dir_folder="", run_id_mlflow=None,
                training={}, architecture= {}, **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]

    folder_c = output_dir_folder+"/run-saves"

    #MODEL DEFINITION
    feats_dims = [v.shape[-1] for v in train_data["views"]]

    encoder_model = create_model(np.sum(feats_dims), emb_dim, **architecture["encoders"])
    predictive_model = create_model(emb_dim, architecture["n_labels"], **architecture["predictive_model"], encoder=False)  #default is mlp
    full_model = torch.nn.Sequential(encoder_model, predictive_model)
    #FUSION DEFINITION
    model = InputFusion(predictive_model=full_model, view_names=train_data["view_names"], loss_function=training["loss_function"])
   
    if "missing_method" in training:
        model.set_missing_info(**training.get("missing_method"))

    #DATA DEFITNION
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    extra_objects = prepare_loggers(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, run_id_mlflow, monitor_name, **early_stop_args)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1,
                         callbacks=extra_objects["callbacks"],logger=extra_objects["loggers"])
    trainer.fit(model, train_dataloader, val_dataloaders=(val_dataloader if type(val_data) != type(None) else None))

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log_additional_mlflow(trainer.loggers[0], trainer, model, architecture)
    return model, trainer

def assign_multifusion_name(training = {}, method = {}, forward_views= [], more_info_str = ""):
    method_name = ""
    if method.get("hybrid"):
        method_name += f"Hyb_{method['agg_args']['mode']}"
    elif method["feature"]:
        method_name += f"Feat_{method['agg_args']['mode']}"
    else:
        method_name += f"Deci_{method['agg_args']['mode']}"

    if "adaptive" in method["agg_args"]:
        if method["agg_args"]["adaptive"]:
            method_name += "_GF"
    if "features" in method["agg_args"]:
        if method["agg_args"]["features"]:
            method_name += "f"

    if training.get("missing_method"):
        method_name += f"-{training['missing_method']['name']}" + (f"_{training['missing_method']['where']}" if training['missing_method'].get("where") else "")
    
    if len(forward_views) != 0: ## define correclty
        method_name += "-Forw_" + "_".join(forward_views)
    
    return method_name + more_info_str


def MultiFusion_train(train_data: dict, val_data = None, 
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="", run_id_mlflow=None,
                     training = {}, method = {}, architecture={}, **kwargs):
    if method_name == "":
        method_name = assign_multifusion_name(training, method)
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]

    folder_c = output_dir_folder+"/run-saves"
    
    #MODEL DEFINITION -- ENCODER
    views_encoder  = {}
    for i, view_n in enumerate(train_data["view_names"]):
        views_encoder[view_n] = create_model(train_data["views"][i].shape[-1], emb_dim, **architecture["encoders"][view_n])
    #MODEL DEFINITION -- Fusion-Part
    
    if method.get("hybrid"):
        method["agg_args"]["emb_dims"] = get_dic_emb_dims(views_encoder)
        merge_module_feat = MergeModule(**method["agg_args"])
        input_dim_task_mapp = merge_module_feat.get_info_dims()["joint_dim"]
        if method["agg_args"].get("adaptive"):
            method["agg_args"]["emb_dims"] = [architecture["n_labels"] for _ in range(len(views_encoder))]
            merge_module_deci = MergeModule(**method["agg_args"])
        else:
            merge_module_deci = None

        predictive_model = create_model(input_dim_task_mapp, architecture["n_labels"], **architecture["predictive_model"], encoder=False) #default is mlp
        model = HybridFusion_FD(views_encoder, merge_module_feat, predictive_model,merge_module_deci=merge_module_deci, loss_function=training["loss_function"])

    elif method["feature"]:
        method["agg_args"]["emb_dims"] = get_dic_emb_dims(views_encoder)
        merge_module = MergeModule(**method["agg_args"])
        input_dim_task_mapp = merge_module.get_info_dims()["joint_dim"]

        predictive_model = create_model(input_dim_task_mapp, architecture["n_labels"], **architecture["predictive_model"], encoder=False)  #default is mlp
        if "multiloss_weights" in training:
            model = FeatureFusionMultiLoss(views_encoder, merge_module, predictive_model, loss_function=training["loss_function"], multiloss_weights=training["multiloss_weights"])
        else:
            model = FeatureFusion(views_encoder, merge_module, predictive_model, loss_function=training["loss_function"])

    else:
        method["agg_args"]["emb_dims"] = [architecture["n_labels"] for _ in range(len(views_encoder))]
        merge_module = MergeModule(**method["agg_args"])

        pred_base = create_model(emb_dim, architecture["n_labels"], **architecture["predictive_model"], encoder=False )  #default is mlp
        prediction_models = {}
        for view_n in views_encoder:
            if architecture["predictive_model"].get("sharing"):
                pred_ = pred_base
            else:
                pred_ = copy.deepcopy(pred_base)
                pred_.load_state_dict(pred_base.state_dict())  
            prediction_models[view_n] = torch.nn.Sequential(views_encoder[view_n], pred_)
            prediction_models[view_n].get_output_size = pred_.get_output_size
        if "multiloss_weights" in training:
            model = DecisionFusionMultiLoss(view_encoders=prediction_models, merge_module=merge_module, loss_function=training["loss_function"], multiloss_weights=training["multiloss_weights"])
        else:
            model = DecisionFusion(view_encoders=prediction_models, merge_module=merge_module, loss_function=training["loss_function"])

    if "missing_method" in training:
        model.set_missing_info(**training.get("missing_method"))
            
    #DATA DEFINITION --
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    #FIT
    extra_objects = prepare_loggers(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, run_id_mlflow, monitor_name, **early_stop_args)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1,
                         callbacks=extra_objects["callbacks"],logger=extra_objects["loggers"])
    trainer.fit(model, train_dataloader, val_dataloaders=(val_dataloader if type(val_data) != type(None) else None))

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log_additional_mlflow(trainer.loggers[0], trainer, model, architecture)
    return model, trainer


def PoolEnsemble_train(train_data: dict, val_data = None,
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="MuFu", run_id_mlflow=None,
                     training = {}, architecture={}, **kwargs):
    start_time_pre = time.time()
    folder_c = output_dir_folder+"/run-saves"
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    
    #MODEL DEFINITION -- ENCODER
    pred_base = create_model(emb_dim, architecture["n_labels"], **architecture["predictive_model"], encoder=False )  #default is mlp
    prediction_models  = {}
    for i, view_n in enumerate(train_data["view_names"]):
        encoder_view_n = create_model(train_data["views"][i].shape[-1], emb_dim, **architecture["encoders"][view_n])
        if architecture["predictive_model"].get("sharing"):
            prediction_head_view_n = pred_base
        else:
            prediction_head_view_n = copy.deepcopy(pred_base)
            prediction_head_view_n.load_state_dict(pred_base.state_dict())    
        prediction_models[view_n] = torch.nn.Sequential(encoder_view_n, prediction_head_view_n)
    model = SingleViewPool(prediction_models, loss_function=training["loss_function"])

    #DATA DEFINITION --
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    #FIT
    extra_objects = prepare_loggers(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, run_id_mlflow, monitor_name, **early_stop_args)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1,
                         callbacks=extra_objects["callbacks"],logger=extra_objects["loggers"])
    trainer.fit(model, train_dataloader, val_dataloaders=(val_dataloader if type(val_data) != type(None) else None))

    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    log_additional_mlflow(trainer.loggers[0], trainer, model, architecture)
    return model, trainer
