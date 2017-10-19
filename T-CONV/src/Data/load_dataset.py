import ast
import csv
import os
import sys

import h5py
import numpy 



class Dataset (object):
    def __init__(self, file_path,log_path):
        print (file_path)
        self.h5file = h5py.File(os.path.join(file_path,'mydata.hdf5'), 'r') 
        self.size = 1600000 #len(self.h5file['train_trip_id'])
        self.start= 0
        self.piece = 400000
        self.end = self.start+ self.piece
        self.logf = open(log_path, 'w')
        
    def load_taxi_data_train (self):    
        print ('loading data ...')
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
        print ('finish loading ...%09d - %09d' % (self.start, self.end))
        self.logf.write ('finish loading ...%09d - %09d \r\n' % (self.start, self.end))
        self.logf.flush()
        self.start = (self.start + self.piece) % self.size
        self.end = (self.end + self.piece) % self.size
        if self.end == 0:
            self.end = self.size

        return train_trip_id, train_call_type, train_origin_call, train_origin_stand, train_taxi_id, \
                    train_timestamp, train_day_type, train_missing_data, train_latitude, train_longitude 

    def load_taxi_data_valid( self):  
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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s file_path' % sys.argv[0]
        sys.exit(1) 

    file_path = sys.argv[1] 
    dataset = Dataset(file_path)    
    [train_trip_id, train_call_type, train_origin_call, train_origin_stand, train_taxi_id, \
            train_timestamp, train_day_type, train_missing_data, train_latitude,  \
            train_longitude] = dataset.load_taxi_data_train()
    [valid_trip_id, valid_call_type, valid_origin_call, valid_origin_stand, valid_taxi_id, \
            valid_timestamp, valid_day_type, valid_missing_data, valid_latitude, \
            valid_longitude] = dataset.load_taxi_data_valid()

    #[stands_name,stands_latitude,stands_longitude] = dataset.load_stands()

    print 'training set ...'
    for i in range(1):
        x = ()
        x += (train_trip_id[i],)
        for j in range(len(train_latitude[i])):
            x+= (train_latitude[i][j], train_longitude[i][j])

        if i == 0:
            print x

        
    print 'validation set ...'
    for i in range(1):
        x = ()
        x += (valid_trip_id[i],) 
        for j in range(len(valid_latitude[i])):
            x += (valid_latitude[i][j], valid_latitude[i][j])
        print x
