import pickle
import sys
import json
import numpy as np
from tqdm import tqdm

import spacy
nlp = spacy.load('en_core_web_sm')

def sentence_segmentation(passage):
    return [s.text for s in nlp(passage).sents]

if len(sys.argv) != 2:
	print("python lookup.py <corpus>")
	exit()
corpus = sys.argv[1]
texttsv = f"/data/lenam/corpus/{corpus}/{corpus}.tsv"

def get_line(path,offset_dict,key):
	offset = offset_dict[key]
	with open(path, 'r', encoding='utf-8') as f:
		f.seek(offset)
		line = f.readline()
	return line.rstrip().split('\t')

with open("/data/lenam/cvst/cn/canard.pkl","rb") as f:
        cn = pickle.load(f)

cn_train, cn_dev, cn_test = cn
_, _, train_rp = cn_train
_, _, dev_rp = cn_dev
_, _, test_rp = cn_test

with open(f"/data/lenam/corpus/{corpus}/{corpus}.offset","rb") as f:
	offset_dict = pickle.load(f)

def sort_run(run):
	return {qid:sort_ranking(ranking) for qid,ranking in run.items()}

def sort_ranking(ranking):
	return {k:v for k,v in sorted(ranking.items(),key=lambda item:item[1],reverse=True)[:1000]}

with open(f"/data/lenam/eval/runs/cn_train_{corpus}.json",'r') as f:
	train_run = sort_run(json.load(f))

with open(f"/data/lenam/eval/runs/cn_dev_{corpus}.json",'r') as f:
	dev_run = sort_run(json.load(f))

with open(f"/data/lenam/eval/runs/cn_test_{corpus}.json",'r') as f:
	test_run = sort_run(json.load(f))

def lcs(S,T):
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    longest = c
    return longest

def overlap(short_answer, passage):
	return lcs(short_answer,passage) > 0.8*len(short_answer)

def trim_answer(short_answer, passage, trimmed_len=4):
	sentences = sentence_segmentation(passage)
	for i in range(len(sentences)):
		if short_answer in sentences[i] or (len(short_answer) > 30 and overlap(short_answer,sentences[i])):
			sentences = sentences[max(0,i-3):min(len(sentences),i+4)]
			if len(sentences) <= trimmed_len:
				start = 0
			else:
				start = np.random.choice(len(sentences)-trimmed_len+1)
			sentences = sentences[max(0,start):min(len(sentences),start+trimmed_len)]
			return ' '.join(sentences)
	return None

print("dev")
with open(f"/data/lenam/cvst/weak_sup/cn_dev_{corpus}.tsv",'w') as f:
	for qid in tqdm(dev_run):
		topic, turn = qid.rsplit('_',1)
		if int(turn) not in dev_rp[topic]:
			continue
		short_answer = dev_rp[topic][int(turn)]
		answer = short_answer
		for rank in range(20):
			docid, score = list(dev_run[qid].items())[rank]
			_, passage = get_line(texttsv, offset_dict, docid)
			trimmed_answer = trim_answer(short_answer,passage)
			if not trimmed_answer is None:
				answer = trimmed_answer
				break
		f.write(f"{qid}\t{score}\t{short_answer}\t{answer}\n")

print("test")
with open(f"/data/lenam/cvst/weak_sup/cn_test_{corpus}.tsv",'w') as f:
	for qid in tqdm(test_run):
		topic, turn = qid.rsplit('_',1)
		if int(turn) not in test_rp[topic]:
			continue
		short_answer = test_rp[topic][int(turn)]
		answer = short_answer
		for rank in range(20):
			docid, score = list(test_run[qid].items())[rank]
			_, passage = get_line(texttsv, offset_dict, docid)
			trimmed_answer = trim_answer(short_answer,passage)
			if not trimmed_answer is None:
				answer = trimmed_answer
				break
		f.write(f"{qid}\t{score}\t{short_answer}\t{answer}\n")

print("train")
with open(f"/data/lenam/cvst/weak_sup/cn_train_{corpus}.tsv",'w') as f:
	for qid in tqdm(train_run):
		topic, turn = qid.rsplit('_',1)
		if int(turn) not in train_rp[topic]:
			continue
		short_answer = train_rp[topic][int(turn)]
		answer = short_answer
		for rank in range(20):
			docid, score = list(train_run[qid].items())[rank]
			_, passage = get_line(texttsv, offset_dict, docid)
			trimmed_answer = trim_answer(short_answer,passage)
			if not trimmed_answer is None:
				answer = trimmed_answer
				break
		f.write(f"{qid}\t{score}\t{short_answer}\t{answer}\n")