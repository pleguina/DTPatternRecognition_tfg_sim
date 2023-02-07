''' Script to dump contents from Ntuples into CSV format '''

# -- Import libraries -- #
import ROOT as r
import os, re, time, sys, pickle
from optparse import OptionParser
from root_numpy import tree2array
import numpy as np
import pandas as pd
import multiprocessing as mp
import utils.baseDumper as bd
from concentrator import combine_dataframes
import time

def addbaseDumperOptions(pr):
  pr.add_option("-v","--verbose"  , dest="verbose"    , action="store_true", default=False, help="If activated, print verbose output")
  pr.add_option('--outpath', '-o',  type="string", dest = "outpath", default = "./results/")
  pr.add_option('--inputFolder', '-i', type="string", dest = "inputFolder", default = ".")
  pr.add_option('--nevents', '-n', type="int", metavar="nevents", dest="nevents", default = 100)
  pr.add_option("--njobs", dest="njobs", type="int", help="Number of cores to use for multiprocess mode.", default = 1)
  return 

def submit(args):
  ''' Function to submit jobs '''
  DTdumper = bd.baseDumper(args[0], args[1])
  DTdumper.run()
  return

def main_run(opts, classtype):   
  # Function for job submission 

  files_  = [f for f in os.listdir(options.inputFolder) if ".root" in f]  
  t0 = time.time()
  with mp.Pool(processes = opts.njobs) as pool:
    pool.map( submit, ((opts, f) for f in files_))
  tf = time.time()
  print("I has taken %3.2f minutes (%3.2f seconds) to write these files."%(abs(t0-tf)/60, abs(t0-tf)))
  return

if __name__ == "__main__":
  # parser inputs
  pr = OptionParser(usage="%prog [options]")

  addbaseDumperOptions(pr)
  (options,args) = pr.parse_args()

  main_run(options, bd.baseDumper)
