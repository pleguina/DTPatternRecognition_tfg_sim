# ----------------------------------- #
#           IMPORT LIBRARIES          #
# ----------------------------------- #

# General Imports
import os, re, time, sys, pickle
import ROOT as r
from optparse import OptionParser
import numpy as np
import pandas as pd
import multiprocessing
from root_numpy import tree2array

r.gStyle.SetOptStat(0)
r.gROOT.SetBatch(True)

# parser inputs
pr = OptionParser(usage="%prog [options]")

def addbaseDumperOptions(pr):
    pr.add_option("-v","--verbose"  , dest="verbose"    , action="store_true", default=False         , help="If activated, print verbose output")
    pr.add_option('--outpath', '-o',  type="string", dest = "outpath", default = "./results/")
    pr.add_option('--inputfile', '-i', type="string", dest = "inputfile", default = "./input.root")
    pr.add_option('--nevents', '-n', type="int", metavar="nevents", dest="nevents", default = -1)
    pr.add_option("--splitBySector", dest="splitBySector", action="store_true", default=False, help="If activated, split the job in one per sector")
    pr.add_option("--splitByWheel", dest="splitByWheel", action="store_true", default=False, help="If activated, split the job in one per wheel")
    pr.add_option("--dumpDigis", dest="dumpDigis", action="store_true", default=False, help="If activated, dump DT digis")
    pr.add_option("--dumpGeom", dest="dumpGeom", action="store_true", default=False, help="If activated, save position of each DT digi (local reference frame)")   
    pr.add_option('--wh', type=int, metavar="wheel", dest="wheel", default=99)   
    pr.add_option('--sc', type=int, metavar="sector", dest="sector", default=99)   
    pr.add_option('--st', type=int, metavar="station", dest="station", default=99)   


addbaseDumperOptions(pr)
(options,args) = pr.parse_args()

def doRun(obj):
    obj.run()
    return 0

def load_data(file_, vars_ = [], sel="", treename = "DTTree", start = None, stop = 100): 
    '''
    Esta funcion sirve para leer rootfiles y convertirlas en
    pandas.DataFrames --> 
    Ref1: https://www.geeksforgeeks.org/python-pandas-dataframe/
    Ref2: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    Esta funcion devuelve un pandas dataframe en el que cada fila
    es un suceso (un event) y cada columna, una variable 
    (e.g. lep1_pt seria una columna)
        '''
    # -- Abrimos la rootfile, y 
    #    convertimos el tree EVENTS en un array
    tfile = r.TFile.Open(file_)    
    
    ttree = tfile.Get(treename)
    arr = tree2array(ttree, branches = vars_, selection = sel, start = start, stop = stop)
    
    # -- Convertimos a dataframe
    df_out = pd.DataFrame(arr, columns=vars_)       
    
    return df_out


class baseDumper(object):
    def __init__(self, options = None, wheel=[], sector=[], nJobs = -1, iJob = -1):
        self.options = options
        self.iJob    = iJob
        self.nJobs   = nJobs
        self.wheels = []
        self.wheels.append(wheel)
        self.sectors = []
        self.sectors.append(sector)
        self.verbose = options.verbose
        self.stations = [1,2,3,4]

        if not(os.path.isdir(options.outpath)):
            try: 
                os.mkdir(options.outpath)
            except:
                pass

    def loadFile(self):
        self.file_ = options.inputfile

    def loadSelections(self):
        self.sels_ = {}
        for wh in self.wheels:
            for sc in self.sectors:
                for mb in self.stations:
                    cutstring = "digi_sector==%d && digi_station==%d && digi_wheel==%d" %(sc,mb,wh)
                    keyname = "Wh%d_S%d_MB%d" %(wh,sc,mb)
                    self.sels_[keyname] = cutstring
        
    def loadVariables(self): 
        self.vars_ = []
        
        if self.verbose: print("  - general variables")
        self.vars_.append("runnumber")
        self.vars_.append("lumiblock")
        self.vars_.append("eventNumber")
       
        if (options.dumpDigis): 
            if self.verbose: print("  - DT digis")
            self.vars_.append("digi_wheel")
            self.vars_.append("digi_sector")
            self.vars_.append("digi_station")
            self.vars_.append("digi_sl")
            self.vars_.append("digi_layer")
            self.vars_.append("digi_wire")
            self.vars_.append("digi_time")

        
    def loadDataFrames(self):        
        self.dfs_ = {}
        for key,sel in self.sels_.items():
            self.dfs_[key] = load_data(self.file_,self.vars_,sel)
        
    def saveDataFrames(self):
        for key, df in self.dfs_.items():
            outfilename = options.outpath + key
            df.to_pickle(outfilename+".pkl")
            df.to_csv(outfilename+".csv")
    
    def run(self):
        if self.verbose: print("Loading file...")
        self.loadFile()

        if self.verbose: print("Loading variables to dump...")
        self.loadVariables()

        if self.verbose: print("Loading selections...")
        self.loadSelections()

        if self.verbose: print("Loading dataFrames...")
        self.loadDataFrames()

        if self.verbose: print("Saving dataFrames...")
        self.saveDataFrames()
              

def main_run(opts,classtype):   
    
    wheels = np.arange(-2,3)
    sectors = np.arange(1,13)
    if opts.splitBySector:
        print("Will run a multi core local job, one per sector")    
        nJobs = sectors.size  
     
        DTDArray = [ classtype(opts, wheels.tolist(), sectors[iJ], nJobs, iJ) for iJ in range(nJobs) ]
        processes = []
        for i in range(nJobs):
            processes.append(multiprocessing.Process(target=doRun, args=[DTDArray[i]]))
            processes[-1].start()
        
        for i in range(nJobs):
            processes[i].join()
        print("We finished")

    elif opts.splitByWheel:
        print("Will run a multi core local job, one per wheel")    
        nJobs = wheels.size  
        
        DTDArray = [ classtype(opts, wheels[iJ], sectors.tolist(), nJobs, iJ) for iJ in range(nJobs) ]
        processes = []
        for i in range(nJobs):
            processes.append(multiprocessing.Process(target=doRun, args=[DTDArray[i]]))
            processes[-1].start()
        
        for i in range(nJobs):
            processes[i].join()
        print("We finished")

    else:
        print("Will run a single core local job")    
        
        for wh in wheels:
            if wh != opts.wheel and opts.wheel<3: continue
            for sc in sectors:
                if sc != opts.sector and opts.sector<13: continue
                print("Dump digis for Wh%d Sc%d" %(wh,sc))
                DTdumper = baseDumper(opts,wh,sc)
                DTdumper.run()
            

if __name__ == "__main__":    
    main_run(options, baseDumper)
                

