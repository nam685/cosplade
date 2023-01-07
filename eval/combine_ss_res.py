import pickle
import sys
import json
import numpy as np
from tqdm import tqdm
import sys

if len(sys.argv)!=4:
	print("python combine_ss_res.py <name1> <name2> <name>")
	exit(1)
ss_1 = f"{sys.argv[1]}.json"
ss_2 = f"{sys.argv[2]}.json"
name = sys.argv[3]

with open(f"/data/lenam/eval/runs/{ss_1}",'r') as f:
	run1 = json.load(f)
with open(f"/data/lenam/eval/runs/{ss_2}",'r') as f:
	run2 = json.load(f)

for qid in tqdm(run1):
	for docid in run1[qid]:
		run1[qid][docid] += run2[qid][docid]


with open(f"/data/lenam/eval/runs/{name}.json",'w') as f:
	run1 = json.dump(run1,f)