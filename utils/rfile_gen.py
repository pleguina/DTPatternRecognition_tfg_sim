import pickle
import os 

def save_pickle(outpath, pck_name, what):
    pick = open("%s/%s.pck"%(outpath, pck_name), "wb")
    pickle.dump(what, pick)
    return

def pickle_toc(outpath, pck):
    print("Loading pickle")
    patterns = pickle.load(open("%s/%s.pck"%(outpath, pck),"rb"))
    print("Creating patterns")

    f = open("%s/%s.cc"%(outpath, pck), "w")
    i = 0

    # Writing header
    f.write('#include "TFile.h"\n')
    f.write('#include <vector>\n')
    f.write("\n")
    f.write('int ' + pck + '() {\n')
    f.write("\n")
    f.write('gInterpreter->GenerateDictionary("vector<vector<vector<int>>>", "vector");\n')
    f.write("\n")
    f.write('TFile* f = new TFile("{}.root", "RECREATE");\n'.format(pck))
    f.write("\n")
    f.write("std::vector<std::vector<std::vector<int>>> allPatterns;\n")
    f.write("\n")

    # Writing patterns
    for p in patterns:
        i += 1
        # print(i, len(patterns))
        f.write("std::vector<std::vector<int>> pattern_"+str(i) +" = {std::vector<int> {" + str(p.seeds[0])+", "+str(p.seeds[1])+ ", " + str(p.seeds[2]) + "}, std::vector<int> {" + "}, std::vector<int>{ ".join([", ".join([str(int(i)) for i in p.hits[j][:]]) for j in range(len(p.hits)) ])+ "}};\n")
        f.write("allPatterns.push_back(pattern_{});\n".format(i))
        f.write("\n")
        # print(p.seeds, p.hits)

    # Closing lines
    f.write('f->cd();\n')
    f.write('f->WriteObject(&allPatterns, "allPatterns");\n')
    f.write('f->Close();\n')
    f.write('return 0;\n')
    f.write('}\n')
        
    f.close()
    return
    