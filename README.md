# T-CONV:A Convolutional Neural Network For Multi-scale Taxi Trajectory Prediction
This program is base on theano, it has two part:
- the global CNN part(T-CONV-Basic)
- local enhancement CNN part(T-CONV-LE)
- [The link to download paper](https://arxiv.org/abs/1611.07635)
- [The link to download data](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data)
</br>
Run the T-CONV/src/Models/conn-global-real.py for global part and the T-CONV/src/Models/conn-2-local.py for local enhancement part. 
Some details for code:
- T-CONV/src/Data/hdf5_dataset_cut.py is used to convert the csv file to hdf5 file
- T-CONV/src/Data/cluster_arrival.py is used for clustering the destinations
- T-CONV/src/Models/save_load_model is used to save and load the paramters for the network.
 
