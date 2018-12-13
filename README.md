# T-CONV:A Convolutional Neural Network For Multi-scale Taxi Trajectory Prediction
This repository contains data and code for our Bigcomp 2018 paper"[T-CONV: A Convolutional Neural Network for Multi-scale Taxi Trajectory Prediction](https://arxiv.org/abs/1611.07635)"
Please cite this paper if you use our code or data.
```
@INPROCEEDINGS{8367101,
author = {J. Lv and Q. Li and Q. Sun and X. Wang},
booktitle = {2018 IEEE International Conference on Big Data and Smart Computing (BigComp)},
title = {T-CONV: A Convolutional Neural Network for Multi-scale Taxi Trajectory Prediction},
year = {2018},
volume = {00},
number = {},
pages = {82-89},
keywords={Trajectory;Public transportation;Predictive models;Prediction algorithms;Clustering algorithms;Neural networks;Companies},
doi = {10.1109/BigComp.2018.00021},
url = {doi.ieeecomputersociety.org/10.1109/BigComp.2018.00021},
ISSN = {2375-9356},
month={Jan}
}
```
This program is base on keras, it has two part:
- the global CNN part(T-CONV-Basic)
- local enhancement CNN part(T-CONV-LE)
- [The link to download data](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data)
</br>
Run the T-CONV/src/Models/conn-global-real.py for global part and the T-CONV/src/Models/conn-local.py for local enhancement part. </br>
Some details for code:</br>
T-CONV/src/Data/hdf5_dataset_cut.py is used to convert the csv file to hdf5 file</br>
T-CONV/src/Data/cluster_arrival.py is used for clustering the destinations</br>