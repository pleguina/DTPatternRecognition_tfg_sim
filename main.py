'''
Code for plotting pseudo-bayes/Standard patterns
in DTs
'''


from plotter.plotter import *
from objects.Primitive import *
from geometry import *
import argparse 
import re
import pandas as pd

def recta(x, x0, slope):
    y = [(xi-x0)*slope for xi in x]
    return y


def generate_csv(logfile):
    f = open(logfile, "r")
    processed_output = {}
    chamber_match = "(.*): Wh(.*) Se(.*) St(..)?"
    
    LUT = {}
    event = -1
    lineCounter = 0
    muonCounter = 0
    for line in f.readlines():
        if line[0] in [" ", "\n", "-"]: continue
        if "Inspecting" in line:
            muonCounter = 0
            evt = int(line.split(" ")[-2]) # Forget the \n
            event = evt
            processed_output[evt] = {}
            continue

        lineCounter += 1 
               
        muonCounter += 1     
        chamber_info = re.match(chamber_match, line).groups()
        
        MuonType = chamber_info[0]
        wheel = chamber_info[1]
        sector = chamber_info[2]
        station = chamber_info[3]
        hits = [int(hit) for hit in line.split("|")[1].split(" ") if hit != '']
        tdcs = [float(tdc) for tdc in line.split("|")[2].split(" ") if tdc != '']
        lateralities = [int(lat) for lat in line.split("|")[3].split(" ") if lat != '']
        fits = [fit for fit in line.split("|")[4].split(" ") if fit != '']
        attrs = ["phi", "phib", "bX", "Chi2", "x", "tanPsi", "t0"]

        LUT[lineCounter] = {"MuonType"  : MuonType,
                            "wheel"   : wheel,
                            "sector"  : sector,
                            "station" : station,
                            "hits"    : hits,
                            "tdcs"    : tdcs,
                            "lateralities" : lateralities,
                            }
        LUT[lineCounter]["Q"] = fits[0]
        for fitpar, attr in enumerate(attrs):
            LUT[lineCounter][attr] = float(fits[fitpar+1])                
        id = "%d-%d"%(event, muonCounter)
        LUT[lineCounter]["id"] = id           
    f.close()
    df = pd.DataFrame(LUT).transpose()
    df.to_csv("%s.csv"%logfile.split(".")[0], sep = "\t")
    return 


def add_parsing_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfile','-l', default="EventDumpList_StdToBayes.log", metavar = "logfile",  dest = "logfile")  
    parser.add_argument('--station','-s', metavar = "station",  dest = "station", required = True)  
    return parser.parse_args()

# Main script 
if __name__ == "__main__":
    
    # Parse options
    opts = add_parsing_options()	
    
    station = opts.station
    logfile = opts.logfile

    stations = {"MB1" : MB1, "MB2" : MB2, "MB3" : MB3, "MB4" : MB4}
    if station not in stations.keys():
        raise RuntimeError("There is no station %s in CMS :)"%station)

    st = stations[station]
    # Create a plotter object
    p = plotter(st)
    
    #generate_csv(logfile)
    # Open data stored in a csv    
    f = open(logfile.replace(".log", ".csv"), "r")
    data = f.readlines()

    header = data[0] # To parse the arguments for the primitive in order
    # Let's calculate the centers

   
    ans = ""
    print("What patterns do you want to plot?\n")
    print("Note that you can either give a single index, or a comma-separated list of patterns")
    print("-----------------------------")
    
    while ans != "exit":
        # Ask for muons
        ans = input("\n > ")
        ans = str(ans)
        if ans != "exit":
#            p.clear()
            # Remove any remnant pattern from a previous execution
            p.clear_patterns()
            muons = [int(muon) for muon in ans.split(",")]
            
            # Create the need primitives with default configuration
            prims = []
            
            for muonid, muon in enumerate(muons):
                pars = data[muon].split("\t")
                attrs = header.split("\t")
                prim = Primitive(pars, attrs, st)
                prims.append(prim)
            p.plot_pattern(prims)

            p.show() 
    f.close()












