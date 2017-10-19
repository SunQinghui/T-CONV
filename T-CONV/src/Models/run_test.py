#!/usr/bin/env python
import os
import csv
from Data import load_dataset


class runOnTest(object):
 
    def __init__(self, model_name ):
        self.model_name = model_name

    #results : the shape is (size, 2)   
    def perform(self, index, results, error_rate, delta = 0.0):
        trip_id = results[0]
        dest = results[1]
        dest_outname = 'test-%s-it%09d-er%f-d%f' %(self.model_name, index, error_rate,delta)
        dest_outfile =  open(os.path.join('output', dest_outname), 'w')
        dest_outcsv = csv.writer(dest_outfile)
        dest_outcsv.writerow(["TRIP_ID", "LATITUDE", "LONGITUDE"])     
                 
        for i in range(len(trip_id)):
            dest_outcsv.writerow((trip_id[i][0], dest[i][0], dest[i][1]))
        
        dest_outfile.close()
