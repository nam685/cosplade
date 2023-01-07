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

cn_dev_aqfla = dict()
for topic in tqdm(dev_q):
	cn_dev_aqfla[topic] = dict()
	title = dev_rp[topic][0]
	for turn in dev_q[topic]:
		past_queries = [dev_q[topic][t] for t in range(1,turn)]
		first_answer = dev_rp[topic][1] if turn>1 else ""
		last_answer = dev_rp[topic][turn-1] if turn>2 else ""
		query = dev_q[topic][turn]
		if len(past_queries)==0:
			context = title
		elif len(past_queries)==1:
			context = title+" "+past_queries[0]+" "+first_answer
		else:
			context = title+" "+past_queries[0]+" "+first_answer+" ".join(past_queries[1:])+" "+last_answer
		context = re.sub(" +"," ",context)
		cn_dev_aqfla[topic][turn] = tokenizer(context,query)
cn_test_aqfla = dict()
for topic in tqdm(test_q):
	cn_test_aqfla[topic] = dict()
	title = test_rp[topic][0]
	for turn in test_q[topic]:
		past_queries = [test_q[topic][t] for t in range(1,turn)]
		first_answer = test_rp[topic][1] if turn>1 else ""
		last_answer = test_rp[topic][turn-1] if turn>2 else ""
		query = test_q[topic][turn]
		if len(past_queries)==0:
			context = title
		elif len(past_queries)==1:
			context = title+" "+past_queries[0]+" "+first_answer
		else:
			context = title+" "+past_queries[0]+" "+first_answer+" ".join(past_queries[1:])+" "+last_answer
		context = re.sub(" +"," ",context)
		cn_test_aqfla[topic][turn] = tokenizer(context,query)
cn_train_aqfla = dict()
for topic in tqdm(train_q):
	cn_train_aqfla[topic] = dict()
	title = train_rp[topic][0]
	for turn in train_q[topic]:
		past_queries = [train_q[topic][t] for t in range(1,turn)]
		first_answer = train_rp[topic][1] if turn>1 else ""
		last_answer = train_rp[topic][turn-1] if turn>2 else ""
		query = train_q[topic][turn]
		if len(past_queries)==0:
			context = title
		elif len(past_queries)==1:
			context = title+" "+past_queries[0]+" "+first_answer
		else:
			context = title+" "+past_queries[0]+" "+first_answer+" ".join(past_queries[1:])+" "+last_answer
		context = re.sub(" +"," ",context)
		cn_train_aqfla[topic][turn] = tokenizer(context,query)

tc20_aqfla = dict()
for topic in tqdm(q20):
	tc20_aqfla[topic] = dict()
	for turn in q20[topic]:
		query = q20[topic][turn]
		past_queries = [q20[topic][t] for t in range(1,turn)]
		if turn > 1:
			first_answer = rp20[topic][1]
			first_answer = key_sentence(first_answer,query,q20[topic][1])
		else:
			first_answer = ""
		if turn > 2:
			last_answer = rp20[topic][turn-1]
			last_answer = key_sentence(last_answer,query,q20[topic][turn-1])
		else:
			last_answer = ""
		if len(past_queries)==0:
			context = ""
		elif len(past_queries)==1:
			context = past_queries[0]+" "+first_answer
		else:
			context = past_queries[0]+" "+first_answer+" ".join(past_queries[1:])+" "+last_answer
		tc20_aqfla[topic][turn] = tokenizer(context,query)

tc21_aqfla = dict()
for topic in tqdm(q21):
	tc21_aqfla[topic] = dict()
	for turn in q21[topic]:
		past_queries = [q21[topic][t] for t in range(1,turn)]
		query = q21[topic][turn]
		if turn > 1:
			first_answer = rp21[topic][1]
			first_answer = key_sentence(first_answer,query,q21[topic][1])
		else:
			first_answer = ""
		if turn > 2:
			last_answer = rp21[topic][turn-1]
			last_answer = key_sentence(last_answer,query,q21[topic][turn-1])
		else:
			last_answer = ""
		if len(past_queries)==0:
			context = ""
		elif len(past_queries)==1:
			context = past_queries[0]+" "+first_answer
		else:
			context = past_queries[0]+" "+first_answer+" ".join(past_queries[1:])+" "+last_answer
		tc21_aqfla[topic][turn] = tokenizer(context,query)


with open("/data/lenam/cvst/cate2e/cat_tokens_aqfla.pkl","wb") as f:
	pickle.dump((cn_train_aqfla,cn_dev_aqfla,cn_test_aqfla,tc20_aqfla,tc21_aqfla), f, protocol=-1)