"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from __future__ import print_function

__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
 
import Data as data
from Data import load_dataset
from Data.valid_transform import VStream 
from Data.transform import Stream   
from linear import Linear   
from embedding import Embedding
import Config.mlp as config 
import error
from save_load_model import SaveLoadParams
#config = importlib.import_module('.mlp', '../Config')

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """

        #------parameters
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            bb = numpy.zeros((n_out,), dtype=theano.config.floatX)
            bb.fill(0.01)
            b_values = bb
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.params = [self.W, self.b]

       

        #------intput and output
        self.input = input
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )



# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        #members - cluster
        self.classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX), name='classes')

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.nnet.relu
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.outputLayer = Linear(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=config.tgtcls.shape[0]
        )
     
        # the parameters of the model are the parameters of the two layer it is
        self.params = self.hiddenLayer.params + self.outputLayer.params

        
        # keep track of model input and output
        self.input = input
        self.output = T.dot(self.outputLayer.output, self.classes)


    def cost(self, y):
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.outputLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (

            (self.hiddenLayer.W ** 2).sum()
            + (self.outputLayer.W ** 2).sum()
        )

        L1_reg = 0.0
        L2_reg = 0.001
        #cost = (self.output + y*0).mean()
        cost = error.erdist(self.output, y).mean() 
        return cost # + L1_reg * self.L1 + L2_reg * self.L2_sqr
            

class Embedding_Mlp(object):
    def __init__(self, traj, origin_call, origin_stand, week_of_year, \
            day_of_week, qhour_of_day, day_type, taxi_id):
        rng = numpy.random.RandomState(1234)

        e_origin_call = Embedding(rng,data.origin_call_train_size, 10, origin_call)
        e_origin_stand = Embedding(rng,data.stands_size, 10, origin_stand)
        e_week_of_year = Embedding(rng,52, 10, week_of_year)
        e_day_of_week = Embedding(rng,7, 10, day_of_week)
        e_qhour_of_day = Embedding(rng,24*4, 10, qhour_of_day)
        e_day_type = Embedding(rng,3, 10, day_type)
        e_taxi_id = Embedding(rng,448, 10, taxi_id)


        #input  
        self.input = T.concatenate((traj, e_origin_call.output, e_origin_stand.output, e_week_of_year.output, \
                e_day_of_week.output, e_qhour_of_day.output, e_day_type.output, e_taxi_id.output), axis = 1) 
        
        #model
        self.mlp = MLP(
            rng=rng,
            input= self.input,
            n_in=4*5 + 7*10, #embedding + trajectory
            n_hidden= 500,
            n_out=2
        )

        #output
        self.output = self.mlp.output

        #parameter
        self.params = self.mlp.params + e_origin_call.params + e_origin_stand.params + \
                e_week_of_year.params + e_day_of_week.params + e_qhour_of_day.params + \
                e_day_type.params + e_taxi_id.params
        #cost
        self.cost = self.mlp.cost

def test_mlp(learning_rate=0.01, n_epochs=10000000, batch_size=200):
    #dataset
    dataset = load_dataset.Dataset('/public/jianmlyu/data/taxi/')
    [trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude] = dataset.load_taxi_data_train()
 
    train_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude)
    
    [trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude, dest_latitude, dest_longitude] = dataset.load_taxi_data_valid()
    valid_stream = VStream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
        timestamp, day_type, missing_data, latitude, longitude, dest_latitude, dest_longitude)


    [trip_id, call_type, origin_call, origin_stand, taxi_id, \
            timestamp, day_type, missing_data, latitude, longitude]= dataset.load_taxi_data_test()
        
    test_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
            timestamp, day_type, missing_data, latitude, longitude)

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
    traj = T.matrix('traj')  # the data is presented as rasterized images
    dest = T.matrix('dest')  # the labels are presented as 1D vector of
    origin_call = T.ivector('origin_call_x')
    origin_stand = T.bvector('origin_stand_x')
    week_of_year = T.ivector('week_of_year_x')
    day_of_week = T.ivector('day_of_week_x')
    qhour_of_day = T.ivector('qhour_of_day_x')
    day_type = T.bvector('day_type_x')
    taxi_id = T.wvector('taxi_id_x')
 
    #classifier
    classifier = Embedding_Mlp( traj, origin_call, origin_stand, week_of_year, \
            day_of_week, qhour_of_day, day_type, taxi_id) 

    #cost function
    cost = classifier.cost(dest)
    output = classifier.output
       
    #test output
    '''
    test_model = theano.function(
        inputs=[bsize],
        outputs=classifier.output,
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    '''

    # specify how to update the parameters of the model as a list of
    save_loader = SaveLoadParams('mlp') 
    #save_loader.save(0, classifier.params)
    #x = save_loader.load(4679)
    #print(len(x))
    #for i in range(len(x)):
    #print (x[0].eval())

     

    #print (len(classifier.params), param.eval() for param in classifier.params)
        
    #    classifier.params = t
    gparams = [T.grad(cost, param) for param in classifier.params]

     
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ] 
 
   
    '''
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            traj: traj_train,
            dest: dest_train,
            origin_call : origin_call_train,
            origin_stand : origin_stand_train,
            day_type : day_type_train,
            taxi_id : taxi_id_train,
            week_of_year : week_of_year_train,
            day_of_week : day_of_week_train,
            qhour_of_day : qhour_of_day_train
        }
    )
    '''
     
    train_model = theano.function(
        inputs=[origin_call,origin_stand,day_type,taxi_id,week_of_year,day_of_week,qhour_of_day,traj,dest,],
        outputs=cost,
        updates=updates 
    )

    validate_model = theano.function(
        inputs=[origin_call,origin_stand,day_type,taxi_id,week_of_year,day_of_week,qhour_of_day,traj,dest,],
        outputs=cost,
    )
   
      
    [trip_id_test, origin_call_test,origin_stand_test,day_type_test,taxi_id_test, \
            week_of_year_test,day_of_week_test,qhour_of_day_test, traj_test] \
            =  test_stream.get_all_data()
        
    test_model = theano.function(
        inputs=[],
        outputs= output,
        givens={
            traj: traj_test,         
            origin_call : origin_call_test,
            origin_stand : origin_stand_test,
            day_type : day_type_test,
            taxi_id : taxi_id_test,
            week_of_year : week_of_year_test,
            day_of_week : day_of_week_test,
            qhour_of_day : qhour_of_day_test
        }
    )
   
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
    validation_frequency = 100#min(n_train_batches, patience // 2)
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
    #test = runOnTest('mlp')
 
    temp0 = [param.get_value() for param in classifier.params]
    temp1 = [param.get_value() for param in classifier.params]   
    while (epoch < n_epochs) :
        epoch = epoch + 1
        if epoch > 10:
            learning_rate = learning_rate / 5


        for minibatch_index in range(n_train_batches):

            [real_epoch, origin_call_train, origin_stand_train, day_type_train,taxi_id_train, week_of_year_train, \
                day_of_week_train,qhour_of_day_train,traj_train, dest_train] \
                = train_stream.get_data_batch_mlp(batch_size)
            
            temp0 = temp1
            temp1 = [param.get_value() for param in classifier.params]
            minibatch_avg_cost = train_model(origin_call_train, origin_stand_train, day_type_train,taxi_id_train, week_of_year_train, \
                day_of_week_train,qhour_of_day_train,traj_train, dest_train) #minibatch_index)
            
  
            if numpy.isnan(minibatch_avg_cost)  == True:
                for i in range(len(classifier.params)):
                    classifier.params[i].set_value (temp0[i])
                print('-----------------------------------')                
            
                  
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            #print (minibatch_index,validation_frequency) 
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = []
                for i in range(n_valid_batches):
                    [valid_epoch, origin_call_valid, origin_stand_valid, day_type_valid,taxi_id_valid, \
                        week_of_year_valid,day_of_week_valid,qhour_of_day_valid,\
                        traj_valid, dest_valid] = valid_stream.get_data_batch_mlp(batch_size)
 
                    validation_losses += [validate_model(origin_call_valid, origin_stand_valid, day_type_valid,taxi_id_valid, \
                        week_of_year_valid,day_of_week_valid,qhour_of_day_valid,\
                        traj_valid, dest_valid),]  
                
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, iter %i, train error %f, validation error %f' %
                    (
                        real_epoch,
                        iter,
                        minibatch_avg_cost,
                        this_validation_loss))

                #save parameter:
                save_loader.save(iter,classifier.params)
                #test               
                results = (trip_id_test, test_model())
                test.perform(iter, results, this_validation_loss)

                # change the dataset
                if (iter + 1)% 150000 == 0:
                    [trip_id, call_type, origin_call, origin_stand, taxi_id, \
                        timestamp, day_type, missing_data, latitude, longitude] = dataset.load_taxi_data_train()
     
                    train_stream = Stream(trip_id, call_type, origin_call, origin_stand, taxi_id, \
                        timestamp, day_type, missing_data, latitude, longitude,conv_dim)

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

            #if patience <= iter:
            #    done_looping = True
            #    break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss, best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)


if __name__ == '__main__':
    test_mlp()
