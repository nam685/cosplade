import pickle
import sys
import json
import numpy as np
from tqdm import tqdm

TC21KILT = "6529kilt.json"
TC21MSDOC1 = "6529msdoc1.json"
TC21WAPO = "6529wapo.json"

def sort_run(run):
	return {qid:sort_ranking(ranking) for qid,ranking in run.items()}
def sort_ranking(ranking):
	return {k:v for k,v in sorted(ranking.items(),key=lambda item:item[1],reverse=True)[:1000]}

with open(f"/data/lenam/eval/runs/{TC21KILT}",'r') as f:
	tc21_run_kilt = json.load(f)
with open(f"/data/lenam/eval/runs/{TC21MSDOC1}",'r') as f:
	tc21_run_msdoc1 = json.load(f)
with open(f"/data/lenam/eval/runs/{TC21WAPO}",'r') as f:
	tc21_run_wapo = json.load(f)

tc21_run = dict()
for qid in tc21_run_kilt:
	tc21_run[qid] = {**tc21_run_kilt[qid],**tc21_run_msdoc1[qid],**tc21_run_wapo[qid]}
tc21_run = sort_run(tc21_run)

def get_line(corpus_file,offset_dict,key):
	offset = offset_dict[key]
	corpus_file.seek(offset)
	line = corpus_file.readline()
	return line.rstrip().split('\t',1)

with open(f"/data/lenam/corpus/kilt/kilt.offset","rb") as f:
	offset_kilt = pickle.load(f)
with open(f"/data/lenam/corpus/msdoc1/msdoc1.offset","rb") as f:
	offset_msdoc1 = pickle.load(f)
with open(f"/data/lenam/corpus/wapo/wapo.offset","rb") as f:
	offset_wapo = pickle.load(f)

with open(f"/data/lenam/cvst/rerank/fs_t21.txt",'w') as outfile:
	with open(f"/data/lenam/corpus/kilt/kilt.tsv","r") as kilt_file:
		with open(f"/data/lenam/corpus/msdoc1/msdoc1.tsv","r") as msdoc1_file:
			with open(f"/data/lenam/corpus/wapo/wapo.tsv","r") as wapo_file:
				for qid in tqdm(sorted(tc21_run,key=lambda x:tuple(map(int,x.split('_'))))):
					for docid in tc21_run[qid]:
						if docid.startswith('KILT_'):
							_, passage = get_line(kilt_file,offset_kilt,docid)
							outfile.write(f"{qid}\t{docid}\t{passage}\n")
						elif docid.startswith('MARCO_D'):
							_, passage = get_line(msdoc1_file,offset_msdoc1,docid)
							outfile.write(f"{qid}\t{docid}\t{passage}\n")
						elif docid.startswith('WAPO_'):
							_, passage = get_line(wapo_file,offset_wapo,docid)
							outfile.write(f"{qid}\t{docid}\t{passage}\n")
						else:
							print(f"Invalid docid {docid}")
							exit(1)
