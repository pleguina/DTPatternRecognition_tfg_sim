''' Implementation of the baseDumper class '''
# Import libraries 
import os, re, time, sys, pickle
import ROOT as r
from optparse import OptionParser
import numpy as np
import pandas as pd
import multiprocessing
from root_numpy import tree2array
from concentrator import combine_dataframes

# -- To remove a very noisy warning that appears when creating arrays with dataframes 
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.object` is a deprecated alias')


class baseDumper(object):
  def __init__(self, options, filename):
    print("   - Reading %s"%filename)
    self.options = options
    self.file_ = filename
    self.stop = options.nevents 
    self.wheels = np.arange(-1, 5)
    self.sectors = np.arange(1, 13)
    self.stations = np.arange(1, 5)
    outpath = os.path.join(options.outpath, filename.split("/")[-1].replace(".root", ""))
    self.outpath = outpath
    if not(os.path.exists(self.outpath)): 
      os.system("mkdir -p %s"%self.outpath)
    return
    

  def loadVariables(self):
    ''' Variables to be fetch from the ntuples '''
    self.vars_ = []
    for var in ["wheel", "sector", "station", "layer", "wire", "time"]:
      self.vars_.append("digi_%s"%var)      
    return 

  def load_selections(self):
    ''' Produce output splitted by wheel and sector'''
    # I don't think this is usefull... Branches are stored as vectors
    # and a selection like wheel = 0 will not work as one would expect...
    self.sels_ = {}
    for wh in self.wheels:
      for sc in self.sectors:
        for mb in self.stations:
          self.sels_["Wh%d_S%d_MB%d"%(wh, sc, mb)] = "digi_sector==%d && digi_station==%d && digi_wheel==%d"%(sc, mb, wh)
    return 


  def load_data(self, file_, vars_ = [], sel = "", treename = "dtNtupleProducer/DTTREE"): 
    ''' Function to load data from a Ntuple and convert it into numpy arrays'''

    tfile = r.TFile.Open(self.options.inputFolder + "/%s"%file_)    
    
    ttree = tfile.Get(treename)
    arr = tree2array(ttree, branches = vars_, selection = sel, start = 0, stop = self.stop)
    print(sel)
    print(pd.DataFrame(arr, columns=vars_).head(10))
    return pd.DataFrame(arr, columns=vars_) 
  
  def loadDataframes(self):
    ''' Load information into a dataframe '''
    self.df = pd.DataFrame() 
    for key, sel in self.sels_.items():
      self.df = combine_dataframes([self.df, self.load_data(self.file_, self.vars_, sel)], axis = 0)
    return
  
  def saveDataFrames(self):
    self.df.to_csv(self.outpath +"/out.csv")
    return

  def run(self):
    ''' Wrapper function for multiprocessing '''
    self.loadVariables()

    self.load_selections()

    self.loadDataframes() 
  
    self.saveDataFrames()
    return 

   
   
