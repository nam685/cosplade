import pickle
import sys
import numpy as np
import numba
import json
from tqdm import tqdm
import h5py

if len(sys.argv) != 4:
	print("python convers_retrieve.py <emb_file> <corpus> <out>")
	exit(1)
embf = sys.argv[1]
corpus = sys.argv[2]
out = sys.argv[3]

def select_topk(filtered_indexes, scores, k):
    if len(filtered_indexes) > k:
        sorted_ = np.argpartition(scores, k)[:k]
        filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
    else:
        scores = -scores
    return filtered_indexes, scores

@numba.njit(nogil=True, parallel=True, cache=True)
def numba_score_float(inverted_index_ids: numba.typed.Dict,
                      inverted_index_floats: numba.typed.Dict,
                      indexes_to_retrieve: np.ndarray,
                      query_values: np.ndarray,
                      threshold: float,
                      size_collection: int):
    scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
    n = len(indexes_to_retrieve)
    for _idx in range(n):
        local_idx = indexes_to_retrieve[_idx]  # which posting list to search
        query_float = query_values[_idx]  # what is the value of the query for this posting list
        retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
        retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
        for j in numba.prange(len(retrieved_indexes)):
            scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
    filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
    # unused documents => this should be tuned, currently it is set to 0
    return filtered_indexes, -scores[filtered_indexes]

sizes = {'car':29794697, 'kilt':17108378, 'msp':8841823, 'msdoc1':22526377, 'msdoc2':97111960, 'wapo':2963130, 'orqa':11377951}

with open(f"/data/lenam/cvst/emb/{embf}.pkl","rb") as f:
	emb = pickle.load(f)
doc_ids = []
with open(f"/data/lenam/corpus/{corpus}/{corpus}_ids","r") as f:
	for line in tqdm(f, total=sizes[corpus]):
		doc_ids.append(line.strip())

index = h5py.File(f"/data/lenam/corpus/{corpus}/{corpus}_splade_index/array_index.h5py","r")
index_doc_ids = numba.typed.Dict()
index_doc_values = numba.typed.Dict()
for key in tqdm(range(30522)):
	try:
		index_doc_ids[key] = np.array(index[f"index_doc_id_{key}"], dtype=np.int32)
		index_doc_values[key] = np.array(index[f"index_doc_value_{key}"], dtype=np.float32)
	except:
		index_doc_ids[key] = np.array([], dtype=np.int32)
		index_doc_values[key] = np.array([], dtype=np.float32)

res = dict()
for topic in tqdm(emb):
	for turn in emb[topic]:
		query = emb[topic][turn]
		col = np.nonzero(query)[0]
		values = query[col.tolist()]
		filtered_indexes, scores = numba_score_float(\
			index_doc_ids,\
			index_doc_values,\
			col,\
			values,\
			threshold=0,\
			size_collection=sizes[corpus])
		filtered_indexes, scores = select_topk(filtered_indexes, scores, k=1000)
		
		qid = str(topic)+"_"+str(turn)
		res[qid] = dict()
		for id_, score in zip(filtered_indexes, scores):
			res[qid][doc_ids[id_]] = float(score)

with open(f"/data/lenam/eval/runs/{out}.json","w") as f:
	json.dump(res, f)
