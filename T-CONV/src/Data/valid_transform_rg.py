import datetime
import numpy
from load_dataset import Dataset
import sys	
import fuel
import Data as data

def at_least_k(k, v, pad_at_begin, is_longitude):
    if len(v) == 0:
        v = numpy.array([data.valid_gps_mean[1 if is_longitude else 0]], dtype=theano.config.floatX)
    if len(v) < k:
        if pad_at_begin:
            v = numpy.concatenate((numpy.full((k - len(v),), v[0],numpy.float32), v))
        else:
            v = numpy.concatenate((v, numpy.full((k - len(v),), v[-1],numpy.float32)))
    return v

def transfer_pos( latitude,longitude,dim):	        
	min_latitude = data.train_gps_mean[0] - data.train_gps_std[0]
	max_latitude = data.train_gps_mean[0] + data.train_gps_std[0]
	min_longtitude = data.train_gps_mean[1] - data.train_gps_std[1]
	max_longtitude = data.train_gps_mean[1] + data.train_gps_std[1]

	dev_latitude = (max_latitude - min_latitude)/dim
	dev_longtitude = (max_longtitude - min_longtitude)/dim

	x = int((latitude - min_latitude)/dev_latitude)
	y = int((longitude - min_longtitude) /dev_longtitude)

	#print (latitude, longitude, min_latitude,max_latitude, dev_latitude, min_longtitude,max_longtitude,dev_longtitude, x,y)    
	if(x >= dim):
		#print ('x', x,dim)
		x = dim - 1		
	if(x < 0):
		x = 0
	if(y >= dim):
		#print ('y', y,dim)
		y = dim - 1		
	if(y < 0):
		y = 0
	return (x,y)
  
class VStream (object):

	def __init__(self, trip_id, call_type, origin_call, origin_stand, taxi_id, \
		timestamp, day_type, missing_data, latitude, longitude,\
		dest_latitude, dest_longitude, conv_dim = 5):
		self.trip_id = trip_id
		self.call_type = call_type
		self.origin_call = origin_call
		self.origin_stand = origin_stand
		self.taxi_id = taxi_id
		self.timestamp = timestamp
		self.day_type = day_type
		self.missing_data = missing_data
		self.latitude = latitude
		self.longitude = longitude
		self.dest_latitude = dest_latitude
		self.dest_longitude = dest_longitude

		self.id = -1
		self.size =  len(self.latitude)

		self.max_splits = 100 
		self.splits = []
		self.isplit = 0
		self.rng = numpy.random.RandomState(fuel.config.default_seed)

		self.k = 20
		self.conv_dim = conv_dim
		self.epoch = 0

	def get_size(self):
		return self.size

	def normalize(self, traj, islatitude):
		if islatitude == True:
			return (traj - data.train_gps_mean[0]) / data.train_gps_std[0]  	
		else:
			return (traj - data.train_gps_mean[1]) / data.train_gps_std[1] 

	 
	def get_sample_data(self, k, bnormalize = True): # get a row of the data, and the sub-traj with randomized length is selected

		if self.id < self.size - 1:
			self.id += 1
		else:
			self.id = 0
			print ('finish epoch %s' % self.epoch)
			self.epoch += 1
 
 		trip_id_ = self.trip_id[self.id]
		call_type_ = self.call_type[self.id]
		origin_call_ = self.origin_call[self.id]
		origin_stand_ = self.origin_stand[self.id]
		taxi_id_ = self.taxi_id[self.id]
		timestamp_ = self.timestamp[self.id]
		day_type_ = self.day_type[self.id]
		missing_data_ = self.missing_data[self.id]
		snapshot_ = self.get_snapshot(self.latitude[self.id], self.longitude[self.id],self.conv_dim)


	
		dest_latitude = self.dest_latitude[self.id]
		dest_longitude = self.dest_longitude [self.id]


		return trip_id_, call_type_, origin_call_,origin_stand_,taxi_id_,timestamp_,day_type_,missing_data_, \
			snapshot_,dest_latitude,dest_longitude
 

	#return the filtered result
	def get_data_batch(self, batchsize,bnormalize = True):
		trip_id_ = numpy.empty(shape=(batchsize,), dtype='S19')
		call_type_ = numpy.empty(shape=(batchsize,), dtype=numpy.int8)
		origin_call_ = numpy.empty(shape=(batchsize,), dtype=numpy.int32)
		origin_stand_ = numpy.empty(shape=(batchsize,), dtype=numpy.int8)
		taxi_id_ = numpy.empty(shape=(batchsize,), dtype=numpy.int16)
		timestamp_ = numpy.empty(shape=(batchsize,), dtype=numpy.int32)
		day_type_ = numpy.empty(shape=(batchsize,), dtype=numpy.int8)
		missing_data_ = numpy.empty(shape=(batchsize,), dtype=numpy.bool)
		snapshot_ =  numpy.empty(shape=(batchsize,1,self.conv_dim, self.conv_dim), dtype=numpy.float32)
		dest_latitude_ = numpy.empty(shape=(batchsize,1), dtype=numpy.float32)
		dest_longitude_ = numpy.empty(shape=(batchsize,1), dtype=numpy.float32)
 
		for i in range(batchsize):
			[trip_id_[i], call_type_[i], origin_call_[i],origin_stand_[i],taxi_id_[i],timestamp_[i], \
	    		day_type_[i],missing_data_[i],snapshot_[i][0],dest_latitude_[i],dest_longitude_[i]] = self.get_sample_data(self.k,bnormalize)

		return trip_id_, call_type_, origin_call_,origin_stand_,taxi_id_,timestamp_,day_type_,missing_data_, \
			snapshot_,dest_latitude_,dest_longitude_
 
 


	def get_snapshot(self, latitudes, longtitudes, output_dim):
		array_x = numpy.zeros((output_dim, output_dim),dtype=numpy.float32) 
		#array_last = numpy.zeros((1,dim, dim),dtype='float32')     
		for i in range(len(latitudes)):
			x = latitudes[i]
			y = longtitudes [i]
			(x,y) = transfer_pos(x,y,output_dim)
    	 
			array_x[x][y] = 0.5 #-self.k + i

		x = latitudes[-1]
		y = longtitudes [-1]
		(x,y) = transfer_pos(x,y,output_dim)
		array_x[x][y] = 1 #i + 1
         
		return array_x

	def get_data_batch_conv(self, batchsize):
		[trip_id_, call_type_, origin_call_,origin_stand_,taxi_id_,timestamp_,day_type_,missing_data_,
			snapshot_,dest_latitude_,dest_longitude_] = self.get_data_batch( batchsize, False)
        
		week_of_year_ = numpy.empty(shape=(batchsize,), dtype=numpy.int16)
		day_of_week_ = numpy.empty(shape=(batchsize,), dtype=numpy.int16)
		qhour_of_day_ = numpy.empty(shape=(batchsize,), dtype=numpy.int16)
		
		for i in range(batchsize):
			date = datetime.datetime.utcfromtimestamp(timestamp_[i])
			yearweek = date.isocalendar()[1] - 1
			week_of_year_[i] = numpy.int8(51 if yearweek == 52 else yearweek)
			day_of_week_[i] =  numpy.int8(date.weekday())
			qhour_of_day_[i] = numpy.int8(date.hour * 4 + date.minute / 15)
		
		#print len(first_k_latitude_),len(first_k_latitude_[0]),len(dest_latitude_),len(dest_longitude_)
		return self.epoch,origin_call_, origin_stand_, day_type_,taxi_id_, week_of_year_,day_of_week_,qhour_of_day_,\
			snapshot_, \
			numpy.concatenate((dest_latitude_,dest_longitude_), axis = 1)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print >> sys.stderr, 'Usage: %s file_path' % sys.argv[0]
		sys.exit(1) 
	file_path = sys.argv[1] 
	dataset = Dataset(file_path)
	[trip_id, call_type, origin_call, origin_stand, taxi_id, \
		timestamp, day_type, missing_data, latitude, longitude] = dataset.load_taxi_data_train()
 
	train_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
		timestamp, day_type, missing_data, latitude, longitude)
	
	[trip_id, call_type, origin_call, origin_stand, taxi_id, \
		timestamp, day_type, missing_data, latitude, longitude] = dataset.load_taxi_data_valid()
	valid_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
		timestamp, day_type, missing_data, latitude, longitude)
    
	[trip_id_, call_type_, origin_call_,origin_stand_,taxi_id_,timestamp_,day_type_, \
    		missing_data_, first_k_latitude_,first_k_longitude_,last_k_latitude_, \
    		last_k_longitude_,dest_latitude_, dest_longitude_] = train_stream.get_data_batch(50)
    
	(x,y) = train_stream.get_data_batch_mlp(50)
	print len(x[0]),len(y[0])

	print 'training set ...'
	for i in range(len(trip_id_)):
		x = ()
		x += (trip_id_[i],)
		for j in range(len(first_k_latitude_[i])):
			x+= (first_k_latitude_[i][j], first_k_longitude_[i][j])
		for j in range(len(last_k_latitude_[i])):
			x+= (last_k_latitude_[i][j], last_k_longitude_[i][j])

		x += (dest_latitude_[i], dest_longitude_[i])
		#print x
    
	[trip_id_, call_type_, origin_call_,origin_stand_,taxi_id_,timestamp_,day_type_, \
    		missing_data_, first_k_latitude_,first_k_longitude_,last_k_latitude_, \
    		last_k_longitude_,dest_latitude_, dest_longitude_] = valid_stream.get_data_batch(50)
	
	print 'valid set ...'
	for i in range(len(trip_id_)):
		x = ()
		x += (trip_id_[i],)
		for j in range(len(first_k_latitude_[i])):
			x+= (first_k_latitude_[i][j], first_k_longitude_[i][j])
		for j in range(len(last_k_latitude_[i])):
			x+= (last_k_latitude_[i][j], last_k_longitude_[i][j])
 
 		x += (dest_latitude_[i], dest_longitude_[i])
		#print x