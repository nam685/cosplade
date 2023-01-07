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
	print("python rerank_tc22.py <input>")
	exit(1)
input_type = sys.argv[1]

paths = {
	'keyword':(
		"/data/lenam/cvst/rfml_keyword/t22_inputs1.pkl"
	),
	'kw_pQs':(
		"/data/lenam/cvst/rfml_keyword/t22_inputs4.pkl"
	),
    'kw_pQs5':(
        "/data/lenam/cvst/rfml_keyword/t22_inputs5.pkl"
    ),
    'kw_pQs20':(
        "/data/lenam/cvst/rfml_keyword/t22_inputs8.pkl"
    )
}

# I copy some code from https://github.com/castorini/pygaggle/blob/master/pygaggle/model/decode.py

query22path = paths[input_type]
with open(query22path,"rb") as f:
	query22 = pickle.load(f)

t22passages_file = "fs_t22"

if exists(f"/data/lenam/cvst/rerank/rr22{input_type}.pkl"):
	t22_ds = RerankDataset({'cache':f'rr22{input_type}'})
else:
	t22_ds = RerankDataset({'name':f'rr22{input_type}','queries':query22,'passages_file':t22passages_file})

print("Created datasets")

BATCH_SIZE=16
t22_dl = DataLoader(t22_ds, shuffle=False, batch_size=BATCH_SIZE, collate_fn=embed_smart_pad)

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

#rr22_3405 = rerank(model,t22_dl,device)
#with open(f"/data/lenam/eval/runs/rr22_3405_{input_type}.json",'w') as f:
#	json.dump(rr22_3405,f)
