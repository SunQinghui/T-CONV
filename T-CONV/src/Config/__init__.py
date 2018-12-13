import os
import cPickle

# from blocks.initialization import IsotropicGaussian, Constant
# from blocks.algorithms import Momentum

import Data as data

n_begin_end_pts = 5  # how many points we consider at the beginning and end of the known trajectory

with open(os.path.join(data.path, 'arrival-clusters.pkl')) as f: tgtcls = cPickle.load(f)
