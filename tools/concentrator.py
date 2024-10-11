''' Script to make concentrator studies '''
import pandas as pd
import os
from optparse import OptionParser
import numpy as np
import glob

# Geometry stuff for plotting 
import geometry.CMSDT as CMSDT
from geometry.MBstation import MBstation
from geometry.Layer import Layer
from geometry.DriftCell import DriftCell

from particle_objects.Primitive import Primitive
from particle_objects.Pattern import Pattern

from utils.DTTrainer import DTTrainer
from utils.DTPlotter import DTPlotter
from utils.rfile_gen import *

pr = OptionParser(usage="%prog [options]")

def addConcentratorOptions(pr):
  pr.add_option('--inpath', '-i', type="string", dest = "inpath", default = "./results/")
  return



def combine_dataframes(dfs, axis = 0):
  '''
  This function combines multiple dataframes:
    axis=0 -- concatenate rows (i.e. add more events)
    axis=1 -- concatenate columns (i.e. add more features)
  '''
  ignore_index = True if axis == 0 else False
  super_df = pd.concat(dfs, axis, ignore_index = ignore_index) 
  return super_df

def csv2df(path):
  df = pd.DataFrame()
  for root, dirs, files in os.walk(inpath):
    for file_ in files:
      print(" >> Reading file: %s in %s "%(file_, root))
      df = combine_dataframes( [pd.read_csv(os.path.join(root, file_)), df], axis = 0)

  df = df.replace(r'\[|\]', '', regex = True)
  return df

if __name__ == "__main__":
  addConcentratorOptions(pr)
  (options, args) = pr.parse_args()
  inpath = options.inpath 
 
  # Load all the data into one dataframe 
  df = csv2df(inpath)
 
  # Each row is an event
  event = df.head(100) # Get the first event  
  print(event)
