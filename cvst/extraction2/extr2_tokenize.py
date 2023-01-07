import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import spacy
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

def process(q,rw,rp):
	qcat = dict()
	qcat_tokens = dict()
	qa_pairs = dict()
	qa_pairs_tokens = dict()
	for topic in tqdm(q):
		tmp_qcat = []
		qcat[topic] = dict()
		qcat_tokens[topic] = dict()
		qa_pairs[topic] = dict()
		qa_pairs_tokens[topic] = dict()
		for q_turn in q[topic]:
			if q_turn == 1:
				query = rw[topic][q_turn]
			else:
				query = q[topic][q_turn]
			qcat[topic][q_turn] = query + '\t' + " [SEP] ".join(tmp_qcat)
			qcat_tokens[topic][q_turn] = tokenizer(query," [SEP] ".join(tmp_qcat))
			tmp_qcat.append(query)
			context_turns = sorted(set(rp[topic].keys()).intersection(range(1,q_turn)).intersection([q_turn-1]))
			qa_pairs[topic][q_turn] = dict()
			qa_pairs_tokens[topic][q_turn] = dict()
			for a_turn in context_turns:
				answer = rp[topic][a_turn]
				#if len(answer) > 200:
				#	answer = key_sentence(answer,query,q[topic][a_turn])
				qa_pairs[topic][q_turn][a_turn] = (query, answer)
				qa_pairs_tokens[topic][q_turn][a_turn] = tokenizer(query, answer)
	return qcat, qcat_tokens, qa_pairs, qa_pairs_tokens

trqc,trqct,trqap,trqapt = process(train_q,train_rw,train_rp)
dvqc,dvqct,dvqap,dvqapt = process(dev_q,dev_rw,dev_rp)
tsqc,tsqct,tsqap,tsqapt = process(test_q,test_rw,test_rp)
t20qc,t20qct,t20qap,t20qapt = process(q20,rw20,rp20)
t21qc,t21qct,t21qap,t21qapt = process(q21,rw21,rp21)

tokens = {
	"trqc":trqc,"trqct":trqct,"trqap":trqap,"trqapt":trqapt,
	"dvqc":dvqc,"dvqct":dvqct,"dvqap":dvqap,"dvqapt":dvqapt,
	"tsqc":tsqc,"tsqct":tsqct,"tsqap":tsqap,"tsqapt":tsqapt,
	"t20qc":t20qc,"t20qct":t20qct,"t20qap":t20qap,"t20qapt":t20qapt,
	"t21qc":t21qc,"t21qct":t21qct,"t21qap":t21qap,"t21qapt":t21qapt
}

with open("/data/lenam/cvst/extraction2/tokens_aqla.pkl","wb") as f:
	pickle.dump(tokens,f,protocol=-1)