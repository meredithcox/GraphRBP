import RNA
import forgi
import numpy as np
import pandas as pd
import pickle

count = 0
to_pickle = []
pos_dotbracket = open('pos_dotbracket_FXR1.txt','r')
for dotbracket in pos_dotbracket:
        source = []
        target = []
        bg, = forgi.load_rna(dotbracket.rstrip())
        pt = bg.to_pair_table()[1:]
        #am = np.zeros((100,100))
        for i in range(len(pt)):
                if i+1 < len(pt):
                        source.append(i)
                        target.append(i+1)
                if i-1 > -1:
                        source.append(i)
                        target.append(i-1)
                if pt[i] != 0:
                        source.append(i)
                        target.append(pt[i]-1)
        source = np.array(source)
        target = np.array(target)
        to_pickle.append(np.stack((source,target)))
        if count % 1000 == 0:
                print(count)
        count += 1

with open('pos_ei_FXR1.pkl','wb') as f:
        pickle.dump(to_pickle,f,protocol=4)
                                               
