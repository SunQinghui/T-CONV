import cPickle
import logging
import os


logger = logging.getLogger(__name__)

class SaveLoadParams(object):
    def __init__(self,model_name):
        self.model_name = model_name


    def save(self,index, parameters):
        self.path = os.path.join('../../../model_data/', 'model_%s_it%09d.pkl' % (self.model_name,index))
        with open(self.path, 'w') as f:
            for param in parameters:
                cPickle.dump(param.get_value() , f, protocol=cPickle.HIGHEST_PROTOCOL)
           
    def load(self, index, len):
        try:
            self.path = os.path.join('../../../model_data/', 'model_%s_it%09d.pkl' % (self.model_name,index))
            result = []
            with open(self.path, 'r') as f:
                for i in range(len):
                    result += [cPickle.load(f),]
                return result

        except IOError:
            print ('load parameter error - %s!' % self.path )
            return None

    

 
