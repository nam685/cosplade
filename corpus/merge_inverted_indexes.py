import h5py
from tqdm import tqdm
import numpy as np
import sys
import json
if len(sys.argv)!=4:
    print("python merge_inverted_indexes <name> <nb_fragments> <nb_docs>")
    exit(1)
name = sys.argv[1]
nb_fragments = int(sys.argv[2])
nb_docs = int(sys.argv[3])
size=2048000
frags = [h5py.File(f"{name}/{name}_splade_index/{size*no_fragment}/index/array_index.h5py", "r") for no_fragment in range(nb_fragments)]
f = h5py.File(f"{name}/{name}_splade_index/array_index.h5py", "w")
count_keys = 0
count_nonzeros = 0
index_dist = {}
for key in tqdm(range(30522)):
    ids = []
    values = []
    for no_fragment in range(nb_fragments):
        frag = frags[no_fragment]
        try:
            ids += (size*no_fragment + np.array(frag["index_doc_id_{}".format(key)], dtype=np.int32)).tolist()
            values += np.array(frag["index_doc_value_{}".format(key)], dtype=np.float32).tolist()
        except:
            continue
    if len(ids) > 0:
        count_keys += 1
        count_nonzeros += len(ids)
        f.create_dataset("index_doc_id_{}".format(key), data=np.array(ids, dtype=np.int32))
        f.create_dataset("index_doc_value_{}".format(key), data=np.array(values, dtype=np.float32))
        index_dist[int(key)] = len(ids)
f.create_dataset("dim", data=count_keys)
for frag in frags:
    frag.close()
f.close()

print("saving index distribution...")  # => size of each posting list in a dict
json.dump(index_dist, open(f"{name}/{name}_splade_index/index_dist.json", "w"))

print("L0_d")
L0_d = count_nonzeros / nb_docs
json.dump({'L0_d':L0_d}, open(f"{name}/{name}_splade_index/index_stats.json","w"))

print("index contains {} posting lists".format(count_keys))
