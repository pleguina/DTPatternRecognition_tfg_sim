import pickle
import os 

def save_pickle(path, pck_name, what):
    if not os.path.exists(path+"/pickles"): 
        os.system("mkdir -p %s/pickles/"%path)
    pick = open("%s/pickles/%s.pck"%(path, pck_name), "wb")
    pickle.dump(what, pick)
    return

def pickle_toc(path, output_file_name, pck):
    if not os.path.exists(path+"/cfiles"): 
        os.system("mkdir -p %s/cfiles/"%path)
    print("Loading pickle")
    patterns = pickle.load(open("%s/pickles/%s"%(path, pck),"rb"))
    print("Creating patterns")

    f = open("%s/cfiles/%s"%(path, output_file_name) + ".cc", "w")
    i = 0

    # Writing header
    f.write('#include "TFile.h"\n')
    f.write('#include <vector>\n')
    f.write("\n")
    f.write('int ' + output_file_name + '() {\n')
    f.write("\n")
    f.write('gInterpreter->GenerateDictionary("vector<vector<vector<int>>>", "vector");\n')
    f.write("\n")
    f.write('TFile* f = new TFile("{}.root", "RECREATE");\n'.format(output_file_name))
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
    