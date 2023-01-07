import pickle
import sys
import json
import numpy as np
from tqdm import tqdm

TC20 = "6529_20.json"

def sort_run(run):
	return {qid:sort_ranking(ranking) for qid,ranking in run.items()}
def sort_ranking(ranking):
	return {k:v for k,v in sorted(ranking.items(),key=lambda item:item[1],reverse=True)[:1000]}

with open(f"/data/lenam/eval/runs/{TC20}",'r') as f:
	tc20_run = json.load(f)

def get_line(corpus_file,offset_dict,key):
	offset = offset_dict[key]
	corpus_file.seek(offset)
	line = corpus_file.readline()
	return line.rstrip().split('\t',1)

with open(f"/data/lenam/corpus/msp/msp.offset","rb") as f:
	offset_msp = pickle.load(f)
with open(f"/data/lenam/corpus/car/car.offset","rb") as f:
	offset_car = pickle.load(f)

with open(f"/data/lenam/cvst/rerank/fs_t20.txt",'w') as outfile:
	with open(f"/data/lenam/corpus/msp/msp.tsv","r") as msp_file:
		with open(f"/data/lenam/corpus/car/car.tsv","r") as car_file:
			for qid in tqdm(sorted(tc20_run,key=lambda x:tuple(map(int,x.split('_'))))):
				for docid in tc20_run[qid]:
					if docid.startswith('CAR_'):
						_, passage = get_line(car_file,offset_car,docid)
						outfile.write(f"{qid}\t{docid}\t{passage}\n")
					elif docid.startswith('MARCO_'):
						_, passage = get_line(msp_file,offset_msp,docid[6:])
						outfile.write(f"{qid}\t{docid}\t{passage}\n")
					else:
						print(f"Invalid docid {docid}")
						exit(1)
