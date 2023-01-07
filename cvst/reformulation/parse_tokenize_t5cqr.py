from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import numpy as np
import pickle
import re
from torch import cuda
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import spacy
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)

with open("/data/lenam/cvst/cn/canard.pkl","rb") as f:
	cn = pickle.load(f)
cn_train, cn_dev, cn_test = cn
train_q, train_rw, train_rp = cn_train
dev_q, dev_rw, dev_rp = cn_dev
test_q, test_rw, test_rp = cn_test

with open("/data/lenam/cvst/tc/treccast_2020.pkl","rb") as f:
	tc20 = pickle.load(f)
q20, rw20, rp20 = tc20
with open("/data/lenam/cvst/tc/treccast_2021.pkl","rb") as f:
	tc21 = pickle.load(f)
q21, rw21, rp21 = tc21

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

def trim_past_responses(past_responses, turn, type, query, query_pred):
	if type == 1:
		past_responses[1:-1] = " "*(len(past_responses)-2)
		if len(past_responses) > 1:
			length = max(120, 1800-len(past_responses[-1])-70*int(turn))
			if len(past_responses[0]) > length:
				past_responses[0] = past_responses[0][:length].rsplit(" ",1)[0] + "..."
	elif type == 2:
		past_responses = " "*(len(past_responses))
	else:
		past_responses[1:-1] = " "*(len(past_responses)-2)
		if len(past_responses) >= 1:
			if len(past_responses[0]) > 200:
				past_responses[0] = key_sentence(past_responses[0],query,query_pred)
		if len(past_responses) >= 2:
			if len(past_responses[-1]) > 200:
				past_responses[-1] = key_sentence(past_responses[-1],query,query_pred)
	return past_responses

def parse_tokenize(q,rw,rp,t,canard=False):
	txt = dict()
	tkns = dict()
	for topic in tqdm(q):
		txt[topic] = dict()
		tkns[topic] = dict()
		turns = list(q[topic].keys())
		queries = list(q[topic].values())
		rewrites = list(rw[topic].values())
		responses = list(rp[topic].values())
		for i in range(len(turns)):
			turn = turns[i]
			query = queries[i]
			query_pred = queries[i-1] if i-1>0 else " "
			rewrite = rewrites[i]
			past_queries = queries[:i]
			if canard:
				title = responses[0]
				past_responses = responses[1:i+1]
			else:
				past_responses = responses[:i]
			trimmed_past_responses = trim_past_responses(past_responses, turn, t, query, query_pred)
			history = re.sub(" +"," ","; ".join([val for pair in list(zip(past_queries,trimmed_past_responses)) for val in pair]))
			if canard:
				history = title + ". " + history
			txt[topic][turn] = (query,history,rewrite)
			input = f"Query: {query}. Context: {history}. Reformulation: "
			tkns[topic][turn] = (tokenizer(input,truncation=True),tokenizer(rewrite))
	return txt, tkns

for t in [1,2,3]:
	tc20_txt, tc20_tk = parse_tokenize(q20, rw20, rp20,t)
	tc21_txt, tc21_tk = parse_tokenize(q21, rw21, rp21,t)
	train_txt, train_tk = parse_tokenize({**train_q,**dev_q},{**train_rw,**dev_rw},{**train_rp,**dev_rp},t,canard=True)
	val_txt, val_tk = parse_tokenize(test_q,test_rw,test_rp,t,canard=True)

	with open(f"/data/lenam/cvst/reformulation/train_txt_{t}.pkl","wb") as f:
		pickle.dump(train_txt,f,protocol=-1)
	with open(f"/data/lenam/cvst/reformulation/train_tk_{t}.pkl","wb") as f:
		pickle.dump(train_tk,f,protocol=-1)

	with open(f"/data/lenam/cvst/reformulation/val_txt_{t}.pkl","wb") as f:
		pickle.dump(val_txt,f,protocol=-1)
	with open(f"/data/lenam/cvst/reformulation/val_tk_{t}.pkl","wb") as f:
		pickle.dump(val_tk,f,protocol=-1)

	with open(f"/data/lenam/cvst/reformulation/tc20_txt_{t}.pkl","wb") as f:
		pickle.dump(tc20_txt,f,protocol=-1)
	with open(f"/data/lenam/cvst/reformulation/tc20_tk_{t}.pkl","wb") as f:
		pickle.dump(tc20_tk,f,protocol=-1)

	with open(f"/data/lenam/cvst/reformulation/tc21_txt_{t}.pkl","wb") as f:
		pickle.dump(tc21_txt,f,protocol=-1)
	with open(f"/data/lenam/cvst/reformulation/tc21_tk_{t}.pkl","wb") as f:
		pickle.dump(tc21_tk,f,protocol=-1)
