import sys
import pickle
path=sys.argv[1]
d = {}
with open(path,"r") as f:
    for i,l in enumerate(f):
        docid = l.strip()
        d[docid] = i
with open("docid_dict","wb") as f:
    pickle.dump(d,f,protocol=-1)

