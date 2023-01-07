from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import pickle

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")

cn_dev = dict()
with open("/data/lenam/topics/cat/cn_dev.tsv","r") as f:
	for line in tqdm(f):
		qid, context, query = line.strip().split("\t",2)
		topic, turn = qid.rsplit('_',1)
		if topic not in cn_dev:
			cn_dev[topic] = dict()
		cn_dev[topic][int(turn)] = tokenizer(context,query)
cn_test = dict()
with open("/data/lenam/topics/cat/cn_test.tsv","r") as f:
	for line in tqdm(f):
		qid, context, query = line.strip().split("\t",2)
		topic, turn = qid.rsplit('_',1)
		if topic not in cn_test:
			cn_test[topic] = dict()
		cn_test[topic][int(turn)] = tokenizer(context,query)
cn_train = dict()
with open("/data/lenam/topics/cat/cn_train.tsv","r") as f:
	for line in tqdm(f):
		qid, context, query = line.strip().split("\t",2)
		topic, turn = qid.rsplit('_',1)
		if topic not in cn_train:
			cn_train[topic] = dict()
		cn_train[topic][int(turn)] = tokenizer(context,query)
tc20 = dict()
with open("/data/lenam/topics/cat/tc20.tsv","r") as f:
	for line in tqdm(f):
		qid, context, query = line.strip().split("\t",2)
		topic, turn = qid.rsplit('_',1)
		if int(topic) not in tc20:
			tc20[int(topic)] = dict()
		tc20[int(topic)][int(turn)] = tokenizer(context,query)
tc21 = dict()
with open("/data/lenam/topics/cat/tc21.tsv","r") as f:
	for line in tqdm(f):
		qid, context, query = line.strip().split("\t",2)
		topic, turn = qid.rsplit('_',1)
		if int(topic) not in tc21:
			tc21[int(topic)] = dict()
		tc21[int(topic)][int(turn)] = tokenizer(context,query)
with open("/data/lenam/cvst/cate2e/cat_tokens.pkl","wb") as f:
	pickle.dump((cn_train,cn_dev,cn_test,tc20,tc21), f, protocol=-1)