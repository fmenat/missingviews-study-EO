import itertools
import numpy as np
import torch

def get_missviews_mask_i(n_views, i):
    lst = list(itertools.product([0, 1], repeat=n_views))[1:]
    return np.asarray(lst[i]).astype(bool)

def possible_missing_mask(n_views):
    return (2**n_views)-1

def augment_based_missing(view_names_forward, views, value_fill = 0.0):
    #for a single=example
    n_views = len(view_names_forward)
    n_missing_mask = possible_missing_mask(n_views)
    
    augmented_views = {v_name: [] for v_name in views}
    for i in range(n_missing_mask):
        views_i = get_missviews_mask_i(n_views, i) #mask as [0,1,1]

        views_i_names = np.asarray(view_names_forward)[views_i] #converted into name of views ["s2","s3"]
        for v_name in views_i_names:
            augmented_views[v_name].append(views[v_name])
        
        missing_i_views = np.asarray(view_names_forward)[~views_i] #converted into name of views ["s2","s3"]
        for v_name in missing_i_views: #fill missing - for now
            if type(views[v_name]) == np.ndarray:
                ones_ = np.ones_like(views[v_name])
                numpy_=True
            elif type(views[v_name]) == torch.Tensor:
                ones_ = torch.ones_like(views[v_name])
                numpy_ = False
            augmented_views[v_name].append(ones_*value_fill)
    if numpy_:
        return {v_name: np.stack(augmented_views[v_name]) for v_name in augmented_views}
    else:
        return {v_name: torch.concat(augmented_views[v_name], dim=0) for v_name in augmented_views}

def augment_array(data, n_views):
    n_missing_mask = possible_missing_mask(n_views)
    return np.repeat(data, n_missing_mask)
    #views_target.repeat( (n_missing_mask,) + tuple([1 for _ in range(len(views_target.shape)-1)]))            
