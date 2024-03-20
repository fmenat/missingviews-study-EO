from pathlib import Path
import pandas as pd


def load_data_sup(data_name, method_name, dir_folder="", **args):
    files_load = [str(v) for v in Path(f"{dir_folder}/pred/{data_name}/{method_name}").glob(f"*.csv")]
    files_load.sort()

    preds_p_run = []
    indxs_p_run = []
    for file_n in files_load:
        data_views = pd.read_csv(file_n, index_col=0) #load_structure(file_n)
        preds_p_run.append(data_views.values)
        indxs_p_run.append(list(data_views.index))
    return preds_p_run,indxs_p_run


def average_ensemble(data_name, method_names, pivot="", **args):
    n_methods_avg = 0
    for method_n in (method_names):
        print(method_n, method_n.split("_")[-1], pivot)
        if pivot == "":
            if n_methods_avg == 0:
                preds_p_run_te, indexs_p_run_te = load_data_sup(data_name+"/test", method_n, **args )
            else:
                preds_p_run_te_a, indexs_p_run_te = load_data_sup(data_name+"/test", method_n, **args )
                preds_p_run_te = [a+b for a,b in zip(preds_p_run_te, preds_p_run_te_a)]
            n_methods_avg+=1
        else:
            if method_n.split("_")[-1] in pivot:
                if n_methods_avg == 0:
                    preds_p_run_te, indexs_p_run_te = load_data_sup(data_name+"/test", method_n, **args )
                else:
                    preds_p_run_te_a, indexs_p_run_te = load_data_sup(data_name+"/test", method_n, **args )
                    preds_p_run_te = [a+b for a,b in zip(preds_p_run_te, preds_p_run_te_a)]
                n_methods_avg+=1
    return [v/n_methods_avg for v in preds_p_run_te], indexs_p_run_te