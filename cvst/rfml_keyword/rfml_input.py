import pickle
import sys
import numpy as np
from transformers import AutoTokenizer
import re
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

FORMAT = '_enhanced_aqaa'
if len(sys.argv)!=3:
	print("python rfml_input.py <seed> <mode>")
	exit(1)
seed = sys.argv[1]
mode = int(sys.argv[2])

tr_runpath = f"/data/lenam/cvst/emb/pred_ti_tr_{seed}.pkl"
dv_runpath = f"/data/lenam/cvst/emb/pred_ti_dv_{seed}.pkl"
ts_runpath = f"/data/lenam/cvst/emb/pred_ti_ts_{seed}.pkl"
t20_runpath = f"/data/lenam/cvst/emb/pred_ti_t20_{seed}.pkl"
t21_runpath = f"/data/lenam/cvst/emb/pred_ti_t21_{seed}.pkl"
def squeeze(emb_dict):
	for topic in emb_dict:
		for turn in emb_dict[topic]:
			emb_dict[topic][turn] = emb_dict[topic][turn].squeeze()
	return emb_dict
print("Loading embedding dicts")
with open(tr_runpath,"rb") as f:
	tr_emb_dict = squeeze(pickle.load(f))
with open(dv_runpath,"rb") as f:
	dv_emb_dict = squeeze(pickle.load(f))
with open(ts_runpath,"rb") as f:
	ts_emb_dict = squeeze(pickle.load(f))
with open(t20_runpath,"rb") as f:
	t20_emb_dict = squeeze(pickle.load(f))
with open(t21_runpath,"rb") as f:
	t21_emb_dict = squeeze(pickle.load(f))

print("Loading data text")
with open(f"/data/lenam/cvst/extraction2/tokens{FORMAT}.pkl","rb") as f:
	tokens = pickle.load(f)
def stringify(d):
	dstr = dict()
	for k in d:
		dstr[str(k)] = d[k]
	return dstr
trqc, trqap = tokens['trqc'], tokens['trqap']
dvqc, dvqap = tokens['dvqc'], tokens['dvqap']
tsqc, tsqap = tokens['tsqc'], tokens['tsqap']
t20qc, t20qap = stringify(tokens['t20qc']), stringify(tokens['t20qap'])
t21qc, t21qap = stringify(tokens['t21qc']), stringify(tokens['t21qap'])

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

print("Loading CQR queries")
cqrseed = 0
with open(f"/data/lenam/cvst/rfml_keyword/generated/t20_1.out", "rb") as f:
	cqr20 = pickle.load(f)
with open(f"/data/lenam/cvst/rfml_keyword/generated/t21_1.out", "rb") as f:
	cqr21 = pickle.load(f)
with open(f"/data/lenam/cvst/rfml_keyword/generated/ts_1.out", "rb") as f:
	cqrts = pickle.load(f)
with open(f"/data/lenam/cvst/rfml_keyword/generated/tr_1.out", "rb") as f:
	cqrtr = pickle.load(f)


def interpret_convo(ti,reverse_voc):
	col = np.nonzero(np.array(ti))[0]
	weights = ti[col].tolist()
	d = {k: v for k, v in zip(col, weights)}
	sorted_d = {k: round(v, 2) for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
#    bow_rep = []
#    for k, v in sorted_d.items():
#        bow_rep.append((reverse_voc[k], round(v, 2)))
	return sorted_d #, bow_rep

def lemmatize(word_weights,lemmatizer=lemmatizer):
	lemmatized_word_weights = dict()
	for word in word_weights:
		if not (len(word)>1 and word[0].isupper() and word[1].isupper()):
			lemmatized_word = lemmatizer.lemmatize(word.lower())
		else:
			lemmatized_word = word
		if lemmatized_word not in lemmatized_word_weights or word_weights[word] > lemmatized_word_weights[lemmatized_word]:
			lemmatized_word_weights[lemmatized_word] = word_weights[word]
	return lemmatized_word_weights

def extract(text,sorted_d):
	clean_text = re.sub(r' +',' ',re.sub(r'[^a-zA-Z0-9 ]',' ',re.sub(r' \[SEP\] ','',text))).strip()
	words = clean_text.split()
	tmp = tokenizer(clean_text.lower(),return_offsets_mapping =True)
	input_ids, offset_mapping = tmp['input_ids'][1:-1], tmp['offset_mapping'][1:-1]
	token_weights = {start:sorted_d[input_id] if input_id in sorted_d else 0 for (input_id,(start,stop)) in zip(input_ids,offset_mapping)}
	offset_mapping_dict = {a:b for a,b in offset_mapping}
	words_offset = [0] + [i for i in range(1,len(clean_text)) if clean_text[i-1]==' ']
	word_weights = dict()
	for offset, word in zip(words_offset,words):
		word_weights[word] = 0
		while offset in offset_mapping_dict:
			word_weights[word] = max(word_weights[word], token_weights[offset])
			offset = offset_mapping_dict[offset]
	return word_weights

def filter(word_weights, topk=10, threshold=0.3):
	lemmatized_word_weights = lemmatize(word_weights)
	chosen = [word for word,weight in list(sorted(lemmatized_word_weights.items(),key=lambda x:x[1],reverse=True))[:topk] if weight > threshold]
	keywords = [word for word in lemmatized_word_weights if word in chosen]
	return keywords

def build_inputs(emb_dict,qc,qap,reverse_voc,cqrdict=None):
	inputs = dict()
	for topic in tqdm(emb_dict):
		inputs[topic] = dict()
		for turn in emb_dict[topic]:
			query = qc[topic][turn].split('\t',1)[0]
			sorted_d = interpret_convo(emb_dict[topic][turn],reverse_voc)
			qcatww = extract(qc[topic][turn], sorted_d)
			answers = ' '.join([a for q,a in qap[topic][turn].values()])
			qapww = extract(answers, sorted_d)
			if mode == 5 or mode == 9:
				keywords = ', '.join(filter({**qcatww,**qapww}, topk=5))
			elif mode == 8 or mode == 10:
				keywords = ', '.join(filter({**qcatww,**qapww}, topk=20, threshold=0.1))
			else:
				keywords = ', '.join(filter({**qcatww,**qapww}))
			if mode == 1 or mode == 9 or mode == 10:
				inputs[topic][turn] = query + ' Keywords: ' + keywords
			elif mode == 2:
				answer = list(qap[topic][turn].values())[-1][1] if len(qap[topic][turn]) > 0 else ""
				if len(answer)>1500:
					answer = answer[:1500].rsplit(' ',1)[0] + '... '
				inputs[topic][turn] = query + ' Context: ' + answer + ' Keywords: ' + keywords
			elif mode == 3:
				queries = ' '.join(qc[topic][turn].split('\t')[1].split(' [SEP] '))
				answer = list(qap[topic][turn].values())[-1][1] if len(qap[topic][turn]) > 0 else ""
				if len(answer)>1500:
					answer = answer[:1500].rsplit(' ',1)[0] + '... '
				inputs[topic][turn] = query + ' Context: ' + queries + ' ' + answer + ' Keywords: ' + keywords
			elif mode == 4 or mode == 5 or mode == 8:
				queries = ' '.join(qc[topic][turn].split('\t')[1].split(' [SEP] '))
				inputs[topic][turn] = query + ' Context: ' + queries + ' Keywords: ' + keywords
			elif mode == 6:
				cqr = cqrdict[topic][turn]
				inputs[topic][turn] = cqr + ' Keywords: ' + keywords
			else:
				print("Unknown mode")
				exit(1)
	return inputs

print("Building inputs")
tr_inputs = build_inputs(tr_emb_dict,trqc,trqap,reverse_voc,cqrtr)
dv_inputs = build_inputs(dv_emb_dict,dvqc,dvqap,reverse_voc,cqrtr)
ts_inputs = build_inputs(ts_emb_dict,tsqc,tsqap,reverse_voc,cqrts)
t20_inputs = build_inputs(t20_emb_dict,t20qc,t20qap,reverse_voc,cqr20)
t21_inputs = build_inputs(t21_emb_dict,t21qc,t21qap,reverse_voc,cqr21)

with open(f"/data/lenam/cvst/rfml_keyword/tr_inputs{mode}.pkl","wb") as f:
	pickle.dump(tr_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/dv_inputs{mode}.pkl","wb") as f:
	pickle.dump(dv_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/ts_inputs{mode}.pkl","wb") as f:
	pickle.dump(ts_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/t20_inputs{mode}.pkl","wb") as f:
	pickle.dump(t20_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/t21_inputs{mode}.pkl","wb") as f:
	pickle.dump(t21_inputs,f,protocol=-1)
