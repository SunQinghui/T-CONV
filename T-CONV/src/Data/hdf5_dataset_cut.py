import ast
import csv
import os
import sys
import h5py
import numpy
import random
import Data as data
import random

taxi_id_dict = {}     #a map from taxiid->0,1,2,3,....
origin_call_dict = {0: 0}  #a map caller -> 0,1,2,3...
# Cuts of the test set minus 1 year

begin = 1372636853
end = 1404172787

random.seed(42)
cuts = []
for i in range(500):
    cuts.append(random.randrange(begin, end))



def get_unique_taxi_id(val):
    if val in taxi_id_dict:
        return taxi_id_dict[val]
    else:
        taxi_id_dict[val] = len(taxi_id_dict)
        return len(taxi_id_dict) - 1

def get_unique_origin_call(val):
    if val in origin_call_dict:
        return origin_call_dict[val]
    else:
        origin_call_dict[val] = len(origin_call_dict)
        return len(origin_call_dict) - 1
 
def convert_test_taxis(input_directory, h5file):
 
    size=getattr(data, 'test_size')

    trip_id = numpy.empty(shape=(size,), dtype='S19')
    call_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    origin_call = numpy.empty(shape=(size,), dtype=numpy.int32)
    origin_stand = numpy.empty(shape=(size,), dtype=numpy.int8)
    taxi_id = numpy.empty(shape=(size,), dtype=numpy.int16)
    timestamp = numpy.empty(shape=(size,), dtype=numpy.int32)
    day_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    missing_data = numpy.empty(shape=(size,), dtype=numpy.bool)
    latitude = numpy.empty(shape=(size,), dtype=h5py.special_dtype(vlen=numpy.float32))
    longitude = numpy.empty(shape=(size,), dtype=h5py.special_dtype(vlen=numpy.float32))
    with open(input_directory, 'r') as f:
        reader = csv.reader(f)
        reader.next() # header
        id=0
        for line in reader:
            if id%10000==0 and id!=0:
                print >> sys.stderr, 'read : %d done' % id
            trip_id[id] = line[0]
            call_type[id] = ord(line[1][0]) - ord('A')
            origin_call[id] = 0 if line[2]=='NA' or line[2]=='' else get_unique_origin_call(int(line[2]))
            origin_stand[id] = 0 if line[3]=='NA' or line[3]=='' else int(line[3])
            taxi_id[id] = get_unique_taxi_id(int(line[4]))
            timestamp[id] = int(line[5])
            day_type[id] = ord(line[6][0]) - ord('A')
            missing_data[id] = line[7][0] == 'T'
            polyline = ast.literal_eval(line[8])
            latitude[id] = numpy.array([point[1] for point in polyline], dtype=numpy.float32)
            longitude[id] = numpy.array([point[0] for point in polyline], dtype=numpy.float32)
            id+=1
    

    for name in ['trip_id', 'call_type', 'origin_call', 'origin_stand', 'taxi_id', 'timestamp', 'day_type', 'missing_data', 'latitude', 'longitude']:
        h5file.create_dataset('test_%s' % name, data = locals()[name])


def convert_train_taxis(input_directory, h5file):
    #print >> sys.stderr, 'read %s: begin' % dataset
    size=getattr(data, 'train_size')

    trip_id = numpy.empty(shape=(size,), dtype='S19')
    call_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    origin_call = numpy.empty(shape=(size,), dtype=numpy.int32)
    origin_stand = numpy.empty(shape=(size,), dtype=numpy.int8)
    taxi_id = numpy.empty(shape=(size,), dtype=numpy.int16)
    timestamp = numpy.empty(shape=(size,), dtype=numpy.int32)
    day_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    missing_data = numpy.empty(shape=(size,), dtype=numpy.bool)
    latitude = numpy.empty(shape=(size,), dtype=h5py.special_dtype(vlen=numpy.float32))
    longitude = numpy.empty(shape=(size,), dtype=h5py.special_dtype(vlen=numpy.float32))

    valid_trip_id = numpy.empty(shape=(size,), dtype='S19')
    valid_call_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    valid_origin_call = numpy.empty(shape=(size,), dtype=numpy.int32)
    valid_origin_stand = numpy.empty(shape=(size,), dtype=numpy.int8)
    valid_taxi_id = numpy.empty(shape=(size,), dtype=numpy.int16)
    valid_timestamp = numpy.empty(shape=(size,), dtype=numpy.int32)
    valid_day_type = numpy.empty(shape=(size,), dtype=numpy.int8)
    valid_missing_data = numpy.empty(shape=(size,), dtype=numpy.bool)
    valid_latitude = numpy.empty(shape=(size,), dtype=h5py.special_dtype(vlen=numpy.float32))
    valid_longitude = numpy.empty(shape=(size,), dtype=h5py.special_dtype(vlen=numpy.float32))
    valid_dest_latitude = numpy.empty(shape=(size,),dtype=numpy.float32)
    valid_dest_longitude = numpy.empty(shape=(size,),dtype=numpy.float32)

    with open(input_directory, 'r') as f:
        reader = csv.reader(f)
        reader.next() # header
        t_id=0
        v_id=0
        for line in reader:
            if t_id%10000==0 and t_id!=0:
                print >> sys.stderr, 'read : %d done' % t_id
            
            time = int(line[5])
            polyline = ast.literal_eval(line[8])
            sel = False
            for ts in cuts:            
                if time <= ts and time + 15 * (len(polyline) - 1) >= ts:
                    # keep it
                    sel = True
                    n = (ts - time) / 15 + 1
                
                    valid_trip_id[v_id] = line[0]
                    valid_call_type[v_id] = ord(line[1][0]) - ord('A')
                    valid_origin_call[v_id] = 0 if line[2]=='NA' or line[2]=='' else get_unique_origin_call(int(line[2]))
                    valid_origin_stand[v_id] = 0 if line[3]=='NA' or line[3]=='' else int(line[3])
                    valid_taxi_id[v_id] = get_unique_taxi_id(int(line[4]))
                    valid_timestamp[v_id] = int(line[5])
                    valid_day_type[v_id] = ord(line[6][0]) - ord('A')
                    valid_missing_data[v_id] = line[7][0] == 'T'
                    
                    valid_latitude[v_id] = numpy.array([point[1] for point in polyline[:n]], dtype=numpy.float32)
                    valid_longitude[v_id] = numpy.array([point[0] for point in polyline[:n]], dtype=numpy.float32)
                    valid_dest_latitude[v_id] = polyline[-1][1]
                    valid_dest_longitude[v_id] = polyline[-1][0]
                    v_id+=1

                    break
            if sel == False:
                trip_id[t_id] = line[0]
                call_type[t_id] = ord(line[1][0]) - ord('A')
                origin_call[t_id] = 0 if line[2]=='NA' or line[2]=='' else get_unique_origin_call(int(line[2]))
                origin_stand[t_id] = 0 if line[3]=='NA' or line[3]=='' else int(line[3])
                taxi_id[t_id] = get_unique_taxi_id(int(line[4]))
                timestamp[t_id] = int(line[5])
                day_type[t_id] = ord(line[6][0]) - ord('A')
                missing_data[t_id] = line[7][0] == 'T'
                polyline = ast.literal_eval(line[8])
                latitude[t_id] = numpy.array([point[1] for point in polyline], dtype=numpy.float32)
                longitude[t_id] = numpy.array([point[0] for point in polyline], dtype=numpy.float32)
                t_id += 1
 
    print t_id, v_id
    for name in ['trip_id', 'call_type', 'origin_call', 'origin_stand', 'taxi_id', 'timestamp', 'day_type', 'missing_data', 'latitude', 'longitude']:
        h5file.create_dataset('train_%s' % name, data = locals()[name][:t_id] )
    for name in ['valid_trip_id', 'valid_call_type', 'valid_origin_call', 'valid_origin_stand', 'valid_taxi_id', 'valid_timestamp', 'valid_day_type', 'valid_missing_data', 'valid_latitude', 'valid_longitude', 'valid_dest_latitude', 'valid_dest_longitude']:
        h5file.create_dataset(name, data = locals()[name][:v_id] )


def convert_stands(input_directory, h5file):
    stands_name = numpy.empty(shape=(data.stands_size,), dtype=('a', 24))
    stands_latitude = numpy.empty(shape=(data.stands_size,), dtype=numpy.float32)
    stands_longitude = numpy.empty(shape=(data.stands_size,), dtype=numpy.float32)
    stands_name[0] = 'None'
    stands_latitude[0] = stands_longitude[0] = 0
    with open(os.path.join(input_directory, 'metaData_taxistandsID_name_GPSlocation.csv'), 'r') as f:
        reader = csv.reader(f)
        reader.next() # header
        for line in reader:
            id = int(line[0])
            stands_name[id] = line[1]
            stands_latitude[id] = float(line[2])
            stands_longitude[id] = float(line[3])
    for name in ['stands_name','stands_latitude','stands_longitude']:
        h5file.create_dataset('%s' % name, data = locals()[name])

 

def convert(input_file, test_file, save_file):
    h5file = h5py.File(save_file, 'w')
    convert_train_taxis(input_file, h5file) 
     
    convert_test_taxis(test_file, h5file) 
    #convert_taxis(input_directory, h5file, 'test')
    #convert_stands(input_directory, h5file)

    h5file.flush()
    h5file.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print >> sys.stderr, 'Usage: %s input_file, test_file, save_file' % sys.argv[0]
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
