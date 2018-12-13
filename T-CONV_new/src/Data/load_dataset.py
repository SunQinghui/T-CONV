import ast
import csv
import os
import sys
import Config as config
import h5py
import numpy
import Data as data
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


class Dataset(object):
    def __init__(self, file_path, log_path):
        print (file_path)
        self.h5file = h5py.File(os.path.join(file_path, 'mydata.hdf5'), 'r')
        self.size = 1600000  # len(self.h5file['train_trip_id'])
        self.start = 0
        self.piece = 1600000
        self.end = self.start + self.piece
        self.logf = open(log_path, 'w')
        self.list = []
        self.rng = numpy.random.RandomState(12345)

    def load_taxi_data_train(self):
        print ('loading new data ...')
        train_trip_id = self.h5file['train_trip_id'][self.start:self.end]
        train_call_type = self.h5file['train_call_type'][self.start:self.end]
        train_origin_call = self.h5file['train_origin_call'][self.start:self.end]
        train_origin_stand = self.h5file['train_origin_stand'][self.start:self.end]
        train_taxi_id = self.h5file['train_taxi_id'][self.start:self.end]
        train_timestamp = self.h5file['train_timestamp'][self.start:self.end]
        train_day_type = self.h5file['train_day_type'][self.start:self.end]
        train_missing_data = self.h5file['train_missing_data'][self.start:self.end]
        train_latitude = self.h5file['train_latitude'][self.start:self.end]
        train_longitude = self.h5file['train_longitude'][self.start:self.end]

        print('finish loading ...%09d - %09d' % (self.start, self.end))
        self.logf.write('finish loading ...%09d - %09d \r\n' % (self.start, self.end))
        self.logf.flush()
        self.start = (self.start + self.piece) % self.size
        self.end = (self.end + self.piece) % self.size
        if self.end == 0:
            self.end = self.size

        return train_trip_id, train_call_type, train_origin_call, train_origin_stand, train_taxi_id, \
               train_timestamp, train_day_type, train_missing_data, train_latitude, train_longitude

    def get_weight_matrix(self):
        train_latitude = self.h5file['train_latitude'][0:100000]
        train_longitude = self.h5file['train_longitude'][0:100000]
        max_num = 0
        row = (80)
        col = (50)
        weight_matrix = numpy.zeros(shape=(row, col))
        # half_dim = 10
        dev_latitude = (data.train_gps_std[0]) / (row / 2)  # 8000/40=200m each cell.
        dev_longitude = (data.train_gps_std[1]) / (col / 2)  # 5000/25=200m
        lefttop_x = data.train_gps_mean[0] - data.train_gps_std[0]
        lefttop_y = data.train_gps_mean[1] - data.train_gps_std[1]

        for i in range(len(train_latitude)):
            if (i % 10000 == 0):
                print(i)
            lats = train_latitude[i]
            lons = train_longitude[i]
            for j in range(len(lats)):
                x = int((lats[j] - lefttop_x) // dev_latitude)
                y = int((lons[j] - lefttop_y) // dev_longitude)

                # print(x,y)
                if (x >= 0 and x < row and y >= 0 and y < col):
                    weight_matrix[x][y] += 1
        for k in range(row):
            for l in range(col):
                # print(weight_matrix[k][l])
                max_num = max(int(weight_matrix[k][l]), int(max_num))
        for p in range(row):
            for q in range(col):
                weight_matrix[p][q] = float(weight_matrix[p][q]) / max_num
                # print(weight_matrix[p][q])
                # print(weight_matrix)
        # df=pd.DataFrame(weight_matrix)
        # weight_matrix.to_csv(weight_matrix.csv)
        # print("finish output")
        return weight_matrix

    def load_taxi_data_valid(self):
        print("test load valid dataset")
        '''
            start = 1350000
            end = 1690000
    
            valid_trip_id =  self.h5file['train_trip_id'][start:end]
            valid_call_type = self.h5file['train_call_type'][start:end]
            valid_origin_call = self.h5file['train_origin_call'][start:end]
            valid_origin_stand = self.h5file['train_origin_stand'][start:end]
            valid_taxi_id = self.h5file['train_taxi_id'][start:end]
            valid_timestamp = self.h5file['train_timestamp'][start:end]
            valid_day_type =self.h5file['train_day_type'][start:end]
            valid_missing_data = self.h5file['train_missing_data'][start:end]
            valid_latitude = self.h5file['train_latitude'][start:end]
            valid_longitude = self.h5file['train_longitude'][start:end]
            valid_dest_latitude = self.h5file['train_latitude'][start:end][-1]
            valid_dest_longitude = self.h5file['train_longitude'][start:end][-1]
            '''
        valid_trip_id = self.h5file['valid_trip_id'][:]
        valid_call_type = self.h5file['valid_call_type'][:]
        valid_origin_call = self.h5file['valid_origin_call'][:]
        valid_origin_stand = self.h5file['valid_origin_stand'][:]
        valid_taxi_id = self.h5file['valid_taxi_id'][:]
        valid_timestamp = self.h5file['valid_timestamp'][:]
        valid_day_type = self.h5file['valid_day_type'][:]
        valid_missing_data = self.h5file['valid_missing_data'][:]
        valid_latitude = self.h5file['valid_latitude'][:]
        valid_longitude = self.h5file['valid_longitude'][:]
        valid_dest_latitude = self.h5file['valid_dest_latitude'][:]
        valid_dest_longitude = self.h5file['valid_dest_longitude'][:]

        print("valid_dest size is %s" % valid_trip_id.shape)
        print("finish load vaild dataset")
        return valid_trip_id, valid_call_type, valid_origin_call, valid_origin_stand, valid_taxi_id, \
               valid_timestamp, valid_day_type, valid_missing_data, valid_latitude, valid_longitude, \
               valid_dest_latitude, valid_dest_longitude
