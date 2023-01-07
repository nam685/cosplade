from transformers import AutoModelForSeq2SeqLM
import pickle
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
from tqdm import tqdm
from rerank_data import *
import torch.nn.functional as F
import numpy as np
import json
import sys
from os.path import exists

if len(sys.argv)!=2:
	print("python rerank_tc.py <input>")
	exit(1)
input_type = sys.argv[1]

paths = {
	'raw':(
		"/data/lenam/cvst/tc/t20q.pkl",
		"/data/lenam/cvst/tc/t21q.pkl"
	),
	'golden':(
		"/data/lenam/cvst/tc/t20rw.pkl",
		"/data/lenam/cvst/tc/t21rw.pkl"
	),
	'rewrite':(
		"/data/lenam/cvst/reformulation/generated/tc20_3_3.out",
		"/data/lenam/cvst/reformulation/generated/tc21_3_3.out"
	),
	'keyword':(
		"/data/lenam/cvst/rfml_keyword/t20_inputs1.pkl",
		"/data/lenam/cvst/rfml_keyword/t21_inputs1.pkl"
	),
	'kw_pQs':(
		"/data/lenam/cvst/rfml_keyword/t20_inputs4.pkl",
		"/data/lenam/cvst/rfml_keyword/t21_inputs4.pkl"
	),
	'cqr_kw':(
		"/data/lenam/cvst/rfml_keyword/t20_inputs6.pkl",
		"/data/lenam/cvst/rfml_keyword/t21_inputs6.pkl"
	),
	'keysentence':(
		"/data/lenam/cvst/rfml_keyword/t20_inputs7.pkl",
		"/data/lenam/cvst/rfml_keyword/t21_inputs7.pkl"
	),
    'kw_pQs5':(
        "/data/lenam/cvst/rfml_keyword/t20_inputs5.pkl",
        "/data/lenam/cvst/rfml_keyword/t21_inputs5.pkl"
    ),
	'kw_pQs20':(
		"/data/lenam/cvst/rfml_keyword/t20_inputs8.pkl",
		"/data/lenam/cvst/rfml_keyword/t21_inputs8.pkl"
	),
    'kw_5':(
        "/data/lenam/cvst/rfml_keyword/t20_inputs9.pkl",
        "/data/lenam/cvst/rfml_keyword/t21_inputs9.pkl",
    ),
    'kw_20':(
        "/data/lenam/cvst/rfml_keyword/t20_inputs10.pkl",
        "/data/lenam/cvst/rfml_keyword/t21_inputs10.pkl",
    )
}

# I copy some code from https://github.com/castorini/pygaggle/blob/master/pygaggle/model/decode.py

query20path, query21path = paths[input_type]
with open(query20path,"rb") as f:
	query20 = pickle.load(f)

with open(query21path,"rb") as f:
	query21 = pickle.load(f)

t20passages_file = "fs_t20"
t21passages_file = "fs_t21"

#if exists(f"/data/lenam/cvst/rerank/rr20{input_type}.pkl"):
#	t20_ds = RerankDataset({'cache':f'rr20{input_type}'})
#else:
#	t20_ds = RerankDataset({'name':f'rr20{input_type}','queries':query20,'passages_file':t20passages_file})

if exists(f"/data/lenam/cvst/rerank/rr21{input_type}.pkl"):
	t21_ds = RerankDataset({'cache':f'rr21{input_type}'})
else:
	t21_ds = RerankDataset({'name':f'rr21{input_type}','queries':query21,'passages_file':t21passages_file})

print("Created datasets")

BATCH_SIZE=16
#t20_dl = DataLoader(t20_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=embed_smart_pad)
#t21_dl = DataLoader(t21_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=embed_smart_pad)

def rescore(model,data,device):
	input_ids = data['input_ids'].to(device=device,dtype=torch.long)
	attention_mask = data['attention_mask'].to(device=device,dtype=torch.long)
	encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
	decode_ids = torch.full((input_ids.size(0), 1), model.config.decoder_start_token_id, dtype=torch.long).to(input_ids.device)
	model_inputs = model.prepare_inputs_for_generation(
		decode_ids,
		encoder_outputs=encoder_outputs,
		past=None,
		attention_mask=attention_mask,
		use_cache=True)
	outputs = model(**model_inputs)
	logits = outputs[0][:, 0, :]
	true_false_proba = F.softmax(logits[:,[1176,6136]],dim=1)
	score = true_false_proba[:,0]
	return score

def rerank(model,loader,device):
	model.eval()
	rankings = dict()
	with torch.no_grad():
		for i,data in tqdm(enumerate(loader)):
			qids = np.array(data['qid'])
			docids = np.array(data['docid'])
			scores = rescore(model,data,device)
			for (qid,docid,score) in zip(qids,docids,scores):
				if qid not in rankings:
					rankings[qid] = dict()
				rankings[qid][docid] = score.item()
	for qid in rankings:
		rankings[qid] = {docid:score for (docid,score) in sorted(rankings[qid].items(),key=lambda x:x[1],reverse=True)}
	return rankings

#model = AutoModelForSeq2SeqLM.from_pretrained("castorini/monot5-base-msmarco-10k")
#model.to(device)

#rr20_6529 = rerank(model,t20_dl,device)
#with open(f"/data/lenam/eval/runs/rr20_6529_{input_type}.json",'w') as f:
#	json.dump(rr20_6529,f)

#rr21_6529 = rerank(model,t21_dl,device)
#with open(f"/data/lenam/eval/runs/rr21_6529_{input_type}.json",'w') as f:
#	json.dump(rr21_6529,f)
