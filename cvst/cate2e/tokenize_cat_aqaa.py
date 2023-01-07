from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import pickle
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import spacy
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
import numpy as np
nlp = spacy.load('en_core_web_sm')

def sentence_segmentation(passage):
    return [s.text for s in nlp(passage).sents]

stemmer = SnowballStemmer("english")
key_tags = set(["CD","FW","JJ","JJR","JJS","NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ"])

def extract(s,stem=False,lower=True):
	# (to lower), tokenize, then filter by tag and remove stopword, (then stem)
	if lower:
		s = s.lower()
	if stem:
		return [stemmer.stem(w) for (w,tag) in nltk.pos_tag(nltk.word_tokenize(s)) if tag in key_tags and w not in stopwords.words('english')]
	else:
		return [w for (w,tag) in nltk.pos_tag(nltk.word_tokenize(s)) if tag in key_tags and w not in stopwords.words('english')]

def sim_overlap(u,ss):
	# number of keywords in common
	return [len(set(extract(s,True,False)) & set(extract(u,True,False))) for s in ss]

def key_sentence(p,u_next,u):
	ss = sentence_segmentation(p)
	if len(ss) == 0:
		return ""
	scores = sim_overlap(u_next,ss)
	if np.max(scores,initial=-1) > 0:
		return ss[np.argmax(scores)]
	else:
		scores = sim_overlap(u,ss)
		if np.max(scores,initial=-1) > 0:
			return ss[np.argmax(scores)]
	return ""

with open("/data/lenam/cvst/tc/treccast_2020.pkl","rb") as f:
	tc20 = pickle.load(f)
q20, rw20, rp20 = tc20
with open("/data/lenam/cvst/tc/treccast_2021.pkl","rb") as f:
	tc21 = pickle.load(f)
q21, rw21, rp21 = tc21

with open("/data/lenam/cvst/cn/canard.pkl","rb") as f:
	cn = pickle.load(f)
cn_train, cn_dev, cn_test = cn
train_q, train_rw, train_rp = cn_train
dev_q, dev_rw, dev_rp = cn_dev
test_q, test_rw, test_rp = cn_test

def trim_past_responses(past_responses,query,past_queries):
	trimmed_past_responses = []
	for i in range(len(past_responses)):
		if len(past_responses[i]) > 200:
			trimmed_past_responses.append(key_sentence(past_responses[i],query,past_queries[i]))
	return trimmed_past_responses

cn_dev_aqfla = dict()
for topic in tqdm(dev_q):
	cn_dev_aqfla[topic] = dict()
	title = dev_rp[topic][0]
	for turn in dev_q[topic]:
		past_queries = [dev_q[topic][t] for t in range(1,turn)]
		past_responses = [dev_rp[topic][t] for t in range(1, turn)]
		query = dev_q[topic][turn]
		trimmed_past_responses = trim_past_responses(past_responses,query,past_queries)
		history = re.sub(" +"," ","; ".join([val for pair in list(zip(past_queries,trimmed_past_responses)) for val in pair]))
		context = title+". "+history
		context = re.sub(" +"," ",context)
		cn_dev_aqfla[topic][turn] = tokenizer(context,query,truncation=True)
cn_test_aqfla = dict()
for topic in tqdm(test_q):
	cn_test_aqfla[topic] = dict()
	title = test_rp[topic][0]
	for turn in test_q[topic]:
		past_queries = [test_q[topic][t] for t in range(1,turn)]
		past_responses = [test_rp[topic][t] for t in range(1, turn)]
		query = test_q[topic][turn]
		trimmed_past_responses = trim_past_responses(past_responses,query,past_queries)
		history = re.sub(" +"," ","; ".join([val for pair in list(zip(past_queries,trimmed_past_responses)) for val in pair]))
		context = title+". "+history
		context = re.sub(" +"," ",context)
		cn_test_aqfla[topic][turn] = tokenizer(context,query,truncation=True)
cn_train_aqfla = dict()
for topic in tqdm(train_q):
	cn_train_aqfla[topic] = dict()
	title = train_rp[topic][0]
	for turn in train_q[topic]:
		past_queries = [train_q[topic][t] for t in range(1,turn)]
		past_responses = [train_rp[topic][t] for t in range(1, turn)]
		query = train_q[topic][turn]
		trimmed_past_responses = trim_past_responses(past_responses,query,past_queries)
		history = re.sub(" +"," ","; ".join([val for pair in list(zip(past_queries,trimmed_past_responses)) for val in pair]))
		context = title+". "+history
		context = re.sub(" +"," ",context)
		cn_train_aqfla[topic][turn] = tokenizer(context,query,truncation=True)

tc20_aqfla = dict()
for topic in tqdm(q20):
	tc20_aqfla[topic] = dict()
	for turn in q20[topic]:
		query = q20[topic][turn]
		past_queries = [q20[topic][t] for t in range(1,turn)]
		past_responses = [rp20[topic][t] for t in range(1, turn)]
		query = q20[topic][turn]
		trimmed_past_responses = trim_past_responses(past_responses,query,past_queries)
		history = re.sub(" +"," ","; ".join([val for pair in list(zip(past_queries,trimmed_past_responses)) for val in pair]))
		context = title+". "+history
		context = re.sub(" +"," ",context)
		tc20_aqfla[topic][turn] = tokenizer(context,query,truncation=True)

tc21_aqfla = dict()
for topic in tqdm(q21):
	tc21_aqfla[topic] = dict()
	for turn in q21[topic]:
		query = q21[topic][turn]
		past_queries = [q21[topic][t] for t in range(1,turn)]
		past_responses = [rp21[topic][t] for t in range(1, turn)]
		query = q21[topic][turn]
		trimmed_past_responses = trim_past_responses(past_responses,query,past_queries)
		history = re.sub(" +"," ","; ".join([val for pair in list(zip(past_queries,trimmed_past_responses)) for val in pair]))
		context = title+". "+history
		context = re.sub(" +"," ",context)
		tc21_aqfla[topic][turn] = tokenizer(context,query,truncation=True)


with open("/data/lenam/cvst/cate2e/cat_tokens_aqaa.pkl","wb") as f:
	pickle.dump((cn_train_aqfla,cn_dev_aqfla,cn_test_aqfla,tc20_aqfla,tc21_aqfla), f, protocol=-1)