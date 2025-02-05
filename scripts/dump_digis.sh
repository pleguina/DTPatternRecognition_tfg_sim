# Script to dump digis from rootfiles

eospath=/eos/user/c/cvicovil/ZprimeToMuMu_M-6000_TuneCP5_14TeV-pythia8/ZprimeToMuMu_M-6000_TuneCP5_14TeV-pythia8/230204_175448/0000/

#python3 dumpDTDigisFromNtuples.py --inputFolder $eospath --nevents 10000 --outpath  ZprimeToMuMu_M-6000_TuneCP5_14TeV-pythia8_multi

nohup python3 dumpDTDigisFromNtuples.py --inputFolder $eospath --nevents 1000 --outpath  ZprimeToMuMu_M-6000_TuneCP5_14TeV-pythia8 --njobs 6 &
