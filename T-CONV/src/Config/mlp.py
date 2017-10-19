import os
import cPickle

#from blocks.initialization import IsotropicGaussian, Constant
#from blocks.algorithms import Momentum

import Data as data
 


n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory

with open(os.path.join(data.path, 'arrival-clusters.pkl')) as f: tgtcls = cPickle.load(f)

dim_embeddings = [
    ('origin_call', data.origin_call_train_size, 10),
    ('origin_stand', data.stands_size, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 10),
    ('taxi_id', 448, 10)
]


#parameters of CNN
conv_step = (1,1)
num_channels = 1
conv_dim = 100
image_shape = (conv_dim ,conv_dim )
conv_sizes = [5, 5]
filter_sizes = zip(conv_sizes, conv_sizes)
feature_maps = [20,50]
border_mode = 'full'
pool_sizes =[2, 2]

#parameters of the MLP
csize = conv_dim
for i in range(len(feature_maps)):
	csize = (csize + conv_sizes[0] - 1)/pool_sizes[0]
dim_input = feature_maps[-1] * csize * csize + sum(x for (_, _, x) in dim_embeddings)
print dim_input
dim_hidden = [500]
dim_output = [tgtcls.shape[0]]
  
'''
embed_weights_init = IsotropicGaussian(0.01)
mlp_weights_init = IsotropicGaussian(0.1) 
mlp_biases_init = Constant(0.01)

step_rule = Momentum(learning_rate=0.01, momentum=0.9)
'''

batch_size = 200

max_splits = 100
  
 
 
