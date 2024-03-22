# Missing Views impact on Multi-view Learning (MVL) models with EO applications
 Public repository of our work in missing views for EO applications.

### Data
Preprocessed data can be accessed at: [Link](https://cloud.dfki.de/owncloud/index.php/s/yxAfArTXkMF7nM2)

### Training
* To train a single-view learning model (e.g. Input-level fusion):  
```
python train_singleview.py -s config/input.yaml
```

* To train all the views individually with single-view learning (e.g. for single-view predictions or Ensemble-based fusion):  
```
python train_singleview_pool.py -s config/pool.yaml
```

* To train a multi-view learning model (e.g. Feature-level fusion, Decision-level fusion, Gated Fusion, Feature-level fusion with MultiLoss):  
```
python train_multiview.py -s config/mv_feat.yaml
```

* To train a multi-view learning model with CCA searching in case of missing views:  
```
python train_multiview_cca.py -s config/mv_cca.yaml
```

### Evaluation
* To evaluate the model by its predictive quality:
```
python evaluate_predictions.py -s config/evaluation.yaml
```

* To evaluate the model by its predictive robustness:
```
python evaluate_rob_pred.py -s config/evaluation.yaml
```

## Source
* [Preprint](https://arxiv.org/abs/2403.14297v1)
* [Final published version]()

## Citation
Mena, F., et al. "Impact Assessment of Missing Data in Model Predictions for Earth Observation Applications." *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)* (2024).

> Bibtex
```
@inproceedings{mena2023comparativeassessmentmultiview,
  title = {Impact Assessment of Missing Data in Model Predictions for {Earth} Observation Applications},
  booktitle = {{IEEE International Geoscience} and {Remote Sensing Symposium} ({IGARSS})},
  author = {Mena, Francisco and Arenas, Diego and Charfuelan, Marcela and Nuske, Marlon and Dengel, Andreas},
  date = {2024},
  publisher = {{IEEE}}
```
