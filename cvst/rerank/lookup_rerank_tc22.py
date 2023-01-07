import pickle
import sys
import json
import numpy as np
from tqdm import tqdm

TC22KILT = "6529kilt22.json"
TC22MSDOC2 = "6529msdoc2.json"
TC22WAPO = "6529wapo22.json"

def sort_run(run):
	return {qid:sort_ranking(ranking) for qid,ranking in run.items()}
def sort_ranking(ranking):
	return {k:v for k,v in sorted(ranking.items(),key=lambda item:item[1],reverse=True)[:1000]}

with open(f"/data/lenam/eval/runs/{TC22KILT}",'r') as f:
	tc22_run_kilt = json.load(f)
with open(f"/data/lenam/eval/runs/{TC22MSDOC2}",'r') as f:
	tc22_run_msdoc2 = json.load(f)
with open(f"/data/lenam/eval/runs/{TC22WAPO}",'r') as f:
	tc22_run_wapo = json.load(f)

tc22_run = dict()
for qid in tc22_run_kilt:
	tc22_run[qid] = {**tc22_run_kilt[qid],**tc22_run_msdoc2[qid],**tc22_run_wapo[qid]}
tc22_run = sort_run(tc22_run)

def get_line(corpus_file,offset_dict,key):
	offset = offset_dict[key]
	corpus_file.seek(offset)
	line = corpus_file.readline()
	return line.rstrip().split('\t',1)

with open(f"/data/lenam/corpus/kilt/kilt.offset","rb") as f:
	offset_kilt = pickle.load(f)
with open(f"/data/lenam/corpus/msdoc2/msdoc2.offset","rb") as f:
	offset_msdoc2 = pickle.load(f)
with open(f"/data/lenam/corpus/wapo/wapo.offset","rb") as f:
	offset_wapo = pickle.load(f)

with open(f"/data/lenam/cvst/rerank/fs_t22.txt",'w') as outfile:
	with open(f"/data/lenam/corpus/kilt/kilt.tsv","r") as kilt_file:
		with open(f"/data/lenam/corpus/msdoc2/msdoc2.tsv","r") as msdoc2_file:
			with open(f"/data/lenam/corpus/wapo/wapo.tsv","r") as wapo_file:
				for qid in tqdm(sorted(tc22_run,key=lambda x:tuple(x.split('_')))):
					for docid in tc22_run[qid]:
						if docid.startswith('KILT_'):
							_, passage = get_line(kilt_file,offset_kilt,docid)
							outfile.write(f"{qid}\t{docid}\t{passage}\n")
						elif docid.startswith('MARCO_'):
							_, passage = get_line(msdoc2_file,offset_msdoc2,docid)
							outfile.write(f"{qid}\t{docid}\t{passage}\n")
						elif docid.startswith('WAPO_'):
							_, passage = get_line(wapo_file,offset_wapo,docid)
							outfile.write(f"{qid}\t{docid}\t{passage}\n")
						else:
							print(f"Invalid docid {docid}")
							exit(1)
