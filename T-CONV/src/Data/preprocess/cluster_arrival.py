# -*- coding:utf-8 -*-
#!/usr/bin/env python2
import numpy
import cPickle
import os
import sys
import h5py
from sklearn.cluster import MeanShift, estimate_bandwidth
sys.path.append ("..")

import Data

print "Generating arrival point list"
dests = []
hdf5_file=h5py.File(os.path.join(Data.path,'mydata.hdf5'), 'r')
print("source data dir is %s" %hdf5_file)
index=hdf5_file['train_latitude'].shape
for v in xrange(int(index[0])):
    if len(hdf5_file['train_latitude'][v]) == 0: continue
    dests.append([hdf5_file['train_latitude'][v][-1], hdf5_file['train_longitude'][v][-1]])
pts = numpy.array(dests)

with open(os.path.join(Data.path, "arrival.pkl"), "w") as f:
    cPickle.dump(pts, f, protocol=cPickle.HIGHEST_PROTOCOL)

#print "Doing clustering"

#bw = estimate_bandwidth(pts, quantile=.1, n_samples=10000)
#print bw

bw = 0.001 #这里直接手工指定的

print "finish find bandwidth"
ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(pts)
cluster_centers = ms.cluster_centers_

print "Clusters shape: ", cluster_centers.shape

with open(os.path.join(Data.path, "arrival-clusters_bw=0001.pkl"), "w") as f:
    cPickle.dump(cluster_centers, f, protocol=cPickle.HIGHEST_PROTOCOL)

