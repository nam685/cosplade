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
from os.path import exists

BATCH_SIZE=16
model = AutoModelForSeq2SeqLM.from_pretrained("castorini/monot5-base-msmarco-10k")
model.to(device)

# I copy some code from https://github.com/castorini/pygaggle/blob/master/pygaggle/model/decode.py

with open("/data/lenam/cvst/cn/canard.pkl","rb") as f:
	cn = pickle.load(f)
cn_train, cn_dev, cn_test = cn
_, tr_rw, _ = cn_train
_, dv_rw, _ = cn_dev
_, ts_rw, _ = cn_test

if exists(f"/data/lenam/cvst/rerank/rrtrcn.pkl"):
	tr_ds = RerankDataset({'cache':'rrtrcn'},pre_tokenize=False,sampled=True)
else:
	tr_ds = RerankDataset({'name':'rrtrcn','queries':tr_rw,'passages_file':'fs_tr'},pre_tokenize=False,sampled=True)
tr_dl = DataLoader(tr_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=embed_smart_pad)

if exists(f"/data/lenam/cvst/rerank/rrdvcn.pkl"):
	dv_ds = RerankDataset({'cache':'rrdvcn'},pre_tokenize=False,sampled=True)
else:
	dv_ds = RerankDataset({'name':'rrdvcn','queries':dv_rw,'passages_file':'fs_dv'},pre_tokenize=False,sampled=True)
dv_dl = DataLoader(dv_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=embed_smart_pad)

if exists(f"/data/lenam/cvst/rerank/rrtscn.pkl"):
	ts_ds = RerankDataset({'cache':'rrtscn'},pre_tokenize=False,sampled=True)
else:
	ts_ds = RerankDataset({'name':'rrtscn','queries':ts_rw,'passages_file':'fs_ts'},pre_tokenize=False,sampled=True)
ts_dl = DataLoader(ts_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=embed_smart_pad)

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

rrdv_golden = rerank(model,dv_dl,device)
with open("/data/lenam/eval/runs/rrdvcn_golden.json",'w') as f:
	json.dump(rrdv_golden,f)

rrts_golden = rerank(model,ts_dl,device)
with open("/data/lenam/eval/runs/rrtscn_golden.json",'w') as f:
	json.dump(rrts_golden,f)

rrtr_golden = rerank(model,tr_dl,device)
with open("/data/lenam/eval/runs/rrtrcn_golden.json",'w') as f:
	json.dump(rrtr_golden,f)