import ast
import csv
import os
import sys

import h5py
import numpy 


class Dataset (object):
    def __init__(self, file_path,log_path):
        print (file_path)
        self.h5file = h5py.File(os.path.join(file_path,'mydata_backup.hdf5'), 'r') 
        self.size =1600000 #len(self.h5file['train_trip_id'])
        self.start= 0
        self.piece = 1600000
        self.end = self.start+ self.piece
        self.logf = open(log_path, 'w')
        
    def load_taxi_data_train (self):    
        print ('loading new data ...')
        train_trip_id =  self.h5file['train_trip_id'][self.start:self.end]
        train_call_type = self.h5file['train_call_type'][self.start:self.end]
        train_origin_call = self.h5file['train_origin_call'][self.start:self.end] 
        train_origin_stand = self.h5file['train_origin_stand'][self.start:self.end]
        train_taxi_id = self.h5file['train_taxi_id'][self.start:self.end] 
        train_timestamp = self.h5file['train_timestamp'][self.start:self.end] 
        train_day_type =self.h5file['train_day_type'][self.start:self.end] 
        train_missing_data = self.h5file['train_missing_data'][self.start:self.end]
        train_latitude = self.h5file['train_latitude'][self.start:self.end]
        train_longitude = self.h5file['train_longitude'][self.start:self.end]
        print('finish loading ...%09d - %09d' % (self.start, self.end))
        self.logf.write ('finish loading ...%09d - %09d \r\n' % (self.start, self.end))
        self.logf.flush()
        self.start = (self.start + self.piece) % self.size
        self.end = (self.end + self.piece) % self.size
        if self.end == 0:
            self.end = self.size

        return train_trip_id, train_call_type, train_origin_call, train_origin_stand, train_taxi_id, \
                    train_timestamp, train_day_type, train_missing_data, train_latitude, train_longitude 

    def load_taxi_data_valid(self):
	print("test load valid dataset")
		  
        valid_trip_id =  self.h5file['valid_trip_id'][:]
        valid_call_type = self.h5file['valid_call_type'][:]
        valid_origin_call = self.h5file['valid_origin_call'][:] 
        valid_origin_stand = self.h5file['valid_origin_stand'][:]
        valid_taxi_id = self.h5file['valid_taxi_id'][:] 
        valid_timestamp = self.h5file['valid_timestamp'][:] 
        valid_day_type =self.h5file['valid_day_type'][:] 
        valid_missing_data = self.h5file['valid_missing_data'][:]
        valid_latitude = self.h5file['valid_latitude'][:]
        valid_longitude = self.h5file['valid_longitude'][:]
        valid_dest_latitude = self.h5file['valid_dest_latitude'][:]
        valid_dest_longitude = self.h5file['valid_dest_longitude'][:]
	'''
        valid_trip_id =  self.h5file['train_trip_id'][1600000:1690000]
        valid_call_type = self.h5file['train_call_type'][1600000:1690000]
        valid_origin_call = self.h5file['train_origin_call'][1600000:1690000]
        valid_origin_stand = self.h5file['train_origin_stand'][1600000:1690000]
        valid_taxi_id = self.h5file['train_taxi_id'][1600000:1690000]
        valid_timestamp = self.h5file['train_timestamp'][1600000:1690000]
        valid_day_type =self.h5file['train_day_type'][1600000:1690000]
        valid_missing_data = self.h5file['train_missing_data'][1600000:1690000]
        valid_latitude = self.h5file['train_latitude'][1600000:1690000]
        valid_longitude = self.h5file['train_longitude'][1600000:1690000]
        valid_dest_latitude = self.h5file['train_latitude'][1600000:1690000][-1]
        valid_dest_longitude = self.h5file['train_longitude'][1600000:1690000][-1]

	
        valid_trip_id =  self.h5file['train_trip_id'][-10000,:]
        valid_call_type = self.h5file['train_call_type'][-10000,:]
        valid_origin_call = self.h5file['train_origin_call'][-10000,:]
        valid_origin_stand = self.h5file['train_origin_stand'][-10000,:]
        valid_taxi_id = self.h5file['train_taxi_id'][[-10000,:]]
        valid_timestamp = self.h5file['train_timestamp'][[-10000,:]]
        valid_day_type =self.h5file['train_day_type'][self.start:self.end]
        valid_missing_data = self.h5file['train_missing_data'][self.start:self.end]
        valid_latitude = self.h5file['train_latitude'][self.start:self.end]
        valid_longitude = self.h5file['train_longitude'][self.start:self.end]
	'''
	print("valid_dest size is %s" %self.h5file['valid_latitude'].shape)
	print("finish load vaild dataset")     
        return valid_trip_id, valid_call_type, valid_origin_call, valid_origin_stand, valid_taxi_id, \
            valid_timestamp, valid_day_type, valid_missing_data, valid_latitude, valid_longitude, \
            valid_dest_latitude, valid_dest_longitude
     
    def load_taxi_data_test( self):  
        test_trip_id =  self.h5file['test_trip_id'][:]
        test_call_type = self.h5file['test_call_type'][:]
        test_origin_call = self.h5file['test_origin_call'][:] 
        test_origin_stand = self.h5file['test_origin_stand'][:]
        test_taxi_id = self.h5file['test_taxi_id'][:] 
        test_timestamp = self.h5file['test_timestamp'][:] 
        test_day_type =self.h5file['test_day_type'][:] 
        test_missing_data = self.h5file['test_missing_data'][:]
        test_latitude = self.h5file['test_latitude'][:]
        test_longitude = self.h5file['test_longitude'][:]
     
        return test_trip_id, test_call_type, test_origin_call, test_origin_stand, test_taxi_id, \
            test_timestamp, test_day_type, test_missing_data, test_latitude, test_longitude 

    def load_stands(self ):
        stands_name = self.h5file['test_trip_id'][:]
        stands_latitude = self.h5file['test_trip_id'][:]
        stands_longitude = self.h5file['test_trip_id'][:]
    
        return stands_name,stands_latitude,stands_longitude   

