import h5py
import numpy
path = '/home/sqh/workspace/T-CONV/data/'
Polyline = h5py.special_dtype(vlen=numpy.float32)
stands_size = 1 # include 0 ("no origin_stands")
train_gps_mean = numpy.array([41.1573, -8.61612], dtype=numpy.float32) #the center point of map
train_gps_std = numpy.sqrt(numpy.array([0.00549598, 0.00333233], dtype=numpy.float32)) #the 1/4 square of the map
train_size=1720000
