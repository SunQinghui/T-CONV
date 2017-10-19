from __future__ import print_function

import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer,MLP,Embedding
from run_test import runOnTest
from save_load_model import SaveLoadParams
import Data as data
from Data import load_dataset
from Data.transform2 import Stream  
from Data.valid_transform2 import VStream

class LeNetConvPoolLayer(object):


    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class CNN (object):

    def __init__(self, input, image_shape):
        #parameters
        rng = numpy.random.RandomState(23455)
        nfilters = [20, 50]
        filter_size = [5,5]
        poolsize_ = (2,2)
        batch_size = image_shape[0]
        image_size = [image_shape[2],image_shape[3]]
        layer0_input = input
        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape= image_shape,
            filter_shape=(nfilters[0], 4, filter_size[0], filter_size[1]),
            poolsize= poolsize_
        )
        image_size[0] = (image_size[0] - filter_size[0] + 1)//2
        image_size[1] = (image_size[1] - filter_size[1] + 1)//2
        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nfilters[0], image_size[0], image_size[1]),
            filter_shape=(nfilters[1], nfilters[0], filter_size[0], filter_size[1]),
            poolsize=poolsize_
        )

        image_size[0] = (image_size[0] - filter_size[0] + 1)//2
        image_size[1] = (image_size[1] - filter_size[1] + 1)//2
        self.output = layer1.output
        self.outputdim = nfilters[1]*image_size[0]*image_size[1]
        self.params = layer0.params + layer1.params

class Embedding_Conv_Mlp(object):
    def __init__(self, image_shape,snapshot,  origin_call, origin_stand, week_of_year, \
            day_of_week, qhour_of_day, day_type, taxi_id):

        rng = numpy.random.RandomState(1234)

        e_origin_call = Embedding(rng,data.origin_call_train_size, 10, origin_call)
        e_origin_stand = Embedding(rng,data.stands_size, 10, origin_stand)
        e_week_of_year = Embedding(rng,52, 10, week_of_year)
        e_day_of_week = Embedding(rng,7, 10, day_of_week)
        e_qhour_of_day = Embedding(rng,24*4, 10, qhour_of_day)
        e_day_type = Embedding(rng,3, 10, day_type)
        e_taxi_id = Embedding(rng,448, 10, taxi_id)

 

        self.cnn = CNN(snapshot,image_shape)
        conv_output = self.cnn.output.flatten(2)  #??? 
        mlp_input = T.concatenate((conv_output, e_origin_call.output, e_origin_stand.output, e_week_of_year.output, \
                e_day_of_week.output, e_qhour_of_day.output, e_day_type.output, e_taxi_id.output), axis = 1) 
                 

        self.mlp = MLP(
            rng=rng,
            input= mlp_input,
            n_in=self.cnn.outputdim + 7*10, #embedding + first_last_traj + snapshot
            n_hidden= 500,
            n_out=2
        )

        #output
        self.output = self.mlp.output

        #parameter
        self.params = self.cnn.params + self.mlp.params + e_origin_call.params + e_origin_stand.params + \
                e_week_of_year.params + e_day_of_week.params + e_qhour_of_day.params + \
                e_day_type.params + e_taxi_id.params
        #cost
        self.cost = self.mlp.cost



def test_conv_mlp(learning_rate=0.01, n_epochs=10000000, batch_size=200):
    learning_rate = learning_rate/5
    conv_dim = 30
    
    image_shape = [batch_size,4,conv_dim,conv_dim]

    #dataset
    dataset = load_dataset.Dataset('../../../data/','../../../logs/log_load_file_c-2-l')

    #log file
    logf = open('../../../logs/log-conn-2-local','w')
    
    #for i in range(2 % 4):
    #    dataset.load_taxi_data_train()

    [trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude] = dataset.load_taxi_data_train()
 
    train_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude,conv_dim)
    
    [trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude, dest_latitude, dest_longitude] = dataset.load_taxi_data_valid()
    valid_stream = VStream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude,dest_latitude, dest_longitude, conv_dim)


    [trip_id, call_type, origin_call, origin_stand, taxi_id, \
            timestamp, day_type, missing_data, latitude, longitude]= dataset.load_taxi_data_test()
        
    test_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
            timestamp, day_type, missing_data, latitude, longitude,conv_dim)

    #test_stream = transform.Stream(dataset.load_taxi_data_test())
     
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_stream.get_size( ) // batch_size
    n_valid_batches = valid_stream.get_size( ) // batch_size
    if n_valid_batches < 1:
        n_valid_batches = 1
    #n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    #first_last_traj = T.matrix('traj')  # the data is presented as rasterized images
    dest = T.matrix('dest')  # the labels are presented as 1D vector of
    origin_call = T.ivector('origin_call_x')
    origin_stand = T.bvector('origin_stand_x')
    week_of_year = T.ivector('week_of_year_x')
    day_of_week = T.ivector('day_of_week_x')
    qhour_of_day = T.ivector('qhour_of_day_x')
    day_type = T.bvector('day_type_x')
    taxi_id = T.wvector('taxi_id_x')
    snapshot= T.tensor4 ('snapeshot') 

    #classifier
    classifier = Embedding_Conv_Mlp(image_shape, snapshot, origin_call, origin_stand, week_of_year, \
            day_of_week, qhour_of_day, day_type, taxi_id) 
    #cost function
    cost = classifier.cost(dest)
    output = classifier.output
    save_loader = SaveLoadParams('conv-2-local') 
    base_iter = 1# 2439999 #1869999
    '''
    length = len(classifier.params)
    loaded_params = save_loader.load(base_iter,length)    
    for i in range(length):
        classifier.params[i].set_value(loaded_params[i])
    '''
    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

 


    train_model = theano.function(
        inputs=[origin_call,origin_stand,day_type,taxi_id,week_of_year,day_of_week,qhour_of_day,snapshot,dest],
        outputs=cost,
        updates=updates,
 
    )

    validate_model = theano.function(
        inputs=[origin_call,origin_stand,day_type,taxi_id,week_of_year,day_of_week,qhour_of_day,snapshot,dest],
        outputs=cost,
         
    )
   
    '''
    test_data = test_stream.get_all_data_conv(batch_size)
 
    test_model = theano.function(
        inputs=[origin_call,origin_stand,day_type,taxi_id,week_of_year,day_of_week,qhour_of_day,snapshot],
        outputs= output,         
    )
    '''
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 1000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = 100 #min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    #test = runOnTest('conn-2-local')
    real_epoch = 0

    temp0 = [param.get_value() for param in classifier.params]
    temp1 = [param.get_value() for param in classifier.params] 
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        print("begin the %s epoch" %epoch)
        if epoch > 10:
            learning_rate = learning_rate / 5
        for minibatch_index in range(n_train_batches):
            [real_epoch,origin_call_train_, origin_stand_train_, day_type_train_,taxi_id_train_, week_of_year_train_, \
                day_of_week_train_,qhour_of_day_train_, snapshot_train_, first_last_traj_train_, dest_train_] \
                = train_stream.get_data_batch_conv(batch_size)
            
            temp0 = temp1
            temp1 = [param.get_value() for param in classifier.params]
            #print (traj_train[0][0],traj_train[0][5],traj_train[0][14],traj_train[0][19], dest_train[0][0],dest_train[0][1])
            minibatch_avg_cost = train_model(origin_call_train_, origin_stand_train_, day_type_train_,taxi_id_train_, week_of_year_train_, \
                day_of_week_train_,qhour_of_day_train_,snapshot_train_,  dest_train_) #minibatch_index)
            

            if numpy.isnan(minibatch_avg_cost)  == True:
                for i in range(len(classifier.params)):
                    classifier.params[i].set_value (temp0[i])
                print('-----------------------------------')    

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index + base_iter
            if(iter%100==0):
                print("the iter is %s" %iter)
            #save parameter:
            if (iter) % 1000 == 0:
                print ('saving ...')
                save_loader.save(iter,classifier.params) 

            # change the dataset
            if (iter)% 2000 == 0:
                print("change the dataset!")
                [trip_id, call_type, origin_call, origin_stand, taxi_id, \
                    timestamp, day_type, missing_data, latitude, longitude] = dataset.load_taxi_data_train()
 
                train_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
                    timestamp, day_type, missing_data, latitude, longitude,conv_dim)
                logf.write('change the dataset......')
                logf.flush()
		

            #print (minibatch_index,validation_frequency) 
            if (iter) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = []
                for i in range(n_valid_batches):
                    [valid_epoch, origin_call_valid_, origin_stand_valid_, day_type_valid_,taxi_id_valid_, \
                        week_of_year_valid_,day_of_week_valid_,qhour_of_day_valid_,\
                        snapshot_valid_, first_last_traj_valid_, dest_valid_] = valid_stream.get_data_batch_conv(batch_size)
   
                    validation_losses += [validate_model(origin_call_valid_, origin_stand_valid_, day_type_valid_,taxi_id_valid_, \
                        week_of_year_valid_,day_of_week_valid_,qhour_of_day_valid_,\
                        snapshot_valid_, dest_valid_),]  
                
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'real_epoch %i, iter %i, train error %f, validation error %f' %
                    (
                        real_epoch,
                        iter,
                        minibatch_avg_cost,
                        this_validation_loss))
                logf.write('%i,%f,%f\r\n' % (iter, minibatch_avg_cost,this_validation_loss))
                logf.flush()

            	'''
                #test
                result = []
                ids = []
                total_size = test_data[0]
                #print ('len = %09d' % len(test_data[1]))
                for i in range(len(test_data[1])):
                    [trip_id_test, origin_call_test,origin_stand_test,\
                        day_type_test,taxi_id_test, week_of_year_test,\
                        day_of_week_test,qhour_of_day_test, snapshot_test, first_last_traj_test] \
                        = test_data[1][i]   
              
                    test_result = test_model(origin_call_test,origin_stand_test,\
                        day_type_test,taxi_id_test, week_of_year_test,\
                        day_of_week_test,qhour_of_day_test, snapshot_test) 

                    if i > 0:
                        result = numpy.concatenate((result, test_result),axis = 0)
                    else:
                        result = test_result
                    if i > 0:
                        ids = numpy.concatenate((ids, trip_id_test),axis = 0)
                    else:
                        ids = trip_id_test
                 
                results = (ids[0:total_size], result[0:total_size])
                test.perform(iter, results,this_validation_loss)

           	  
                '''
                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    #test_losses = [test_model(i) for i
                    #               in range(n_test_batches)]
                    #test_score = numpy.mean(test_losses)

                    #print(('     epoch %i, minibatch %i/%i, test error of '
                    #       'best model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))

             

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss, best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

  

if __name__ == '__main__':
    test_conv_mlp()
