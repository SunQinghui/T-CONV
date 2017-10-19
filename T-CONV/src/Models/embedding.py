"""Introduces Lookup brick."""
import theano
import numpy

class Embedding (object):
    """Encapsulates representations of a range of integers.

    Parameters
    ----------
    length : int
        The size of the lookup table, or in other words, one plus the
        maximum index for which a representation is contained.
    dim : int
        The dimensionality of representations.

    Notes
    -----
 
    """ 
 
    def __init__(self, rng, length, dim, input):
        self.length = length
        self.dim = dim
        
        #parameter
        W_values = numpy.asarray( \
                rng.uniform( \
                    low=-numpy.sqrt(6. / (length + dim)), \
                    high=numpy.sqrt(6. / (length + dim)), \
                    size=(length, dim) \
                ), \
                dtype=theano.config.floatX \
            ) 

        self.W = theano.shared(
            value= W_values,
            name='W',
            borrow=True
        )

        # parameters of the model
        self.params = [self.W]

        # keep track of model input
        self.input = input

        # keep track of model output
        output_shape = [input.shape[0], self.dim]
        self.output = self.W[input.flatten()].reshape(output_shape) 

  
