import pickle
import sys
import json
import numpy as np
from tqdm import tqdm

CNDV = "cn_dev_orqa.json"
CNTS = "cn_test_orqa.json"
CNTR = "cn_train_orqa.json"

def sort_run(run):
	return {qid:sort_ranking(ranking) for qid,ranking in run.items()}
def sort_ranking(ranking):
	return {k:v for k,v in sorted(ranking.items(),key=lambda item:item[1],reverse=True)[:1000]}

with open(f"/data/lenam/eval/runs/{CNTR}",'r') as f:
	train_run = sort_run(json.load(f))
with open(f"/data/lenam/eval/runs/{CNDV}",'r') as f:
	dev_run = sort_run(json.load(f))
with open(f"/data/lenam/eval/runs/{CNTS}",'r') as f:
	test_run = sort_run(json.load(f))
with open(f"/data/lenam/corpus/orqa/orqa.offset","rb") as f:
	offset_orqa = pickle.load(f)

def get_line(corpus_file,offset_dict,key):
	offset = offset_dict[key]
	corpus_file.seek(offset)
	line = corpus_file.readline()
	return line.rstrip().split('\t')

def lookup(run, out):
	with open(f"/data/lenam/cvst/rerank/fs_{out}.txt","w") as outfile:
		with open(f"/data/lenam/corpus/orqa/orqa.tsv","r") as corpus_file:
			for qid in tqdm(run):
				for docid in list(run[qid].keys())[:100]:
					_, passage = get_line(corpus_file,offset_orqa,docid)
					outfile.write(f"{qid}\t{docid}\t{passage}\n")
	return run

dev_run = lookup(dev_run,'dv')
test_run = lookup(test_run,'ts')
train_run = lookup(train_run,'tr')