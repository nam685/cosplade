from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm
import numpy as np
import pickle
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import json

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
model.eval()
model.to(device)

with open("/data/lenam/topics/raw/2022_evaluation_topics_tree_v1.0.json","r") as tc22:
	with open("/data/lenam/topics/raw/2022_evaluation_topics_turn_ids.json","r") as ftids:
		_obj = json.load(tc22)
		obj = dict()
		for _topic in _obj:
			obj[_topic['number']] = dict()
			for _turn in _topic['turn']:
				obj[_topic['number']][_turn['number']] = {k:v for k,v in _turn.items() if k != 'number'}
		tids = json.load(ftids)
		rw = dict()

		for topic in tids:
			rw[int(topic)] = dict()
			for turn in tids[topic]:
				rw[int(topic)][turn] = obj[int(topic)][turn]['manual_rewritten_utterance']

def encode(d):
	term_importances = dict()
	for topic in tqdm(d):
		term_importances[topic] = dict()
		for turn in d[topic]:
			tokens = tokenizer(d[topic][turn])
			out = model(\
					torch.tensor(tokens['input_ids']).unsqueeze(0).to(device),\
					torch.tensor(tokens['attention_mask']).unsqueeze(0).to(device)\
				).logits
			values, _ = torch.max(torch.log(1 + torch.relu(out)) * torch.tensor(tokens["attention_mask"]).unsqueeze(-1).to(device), dim=1)
			ti = values.detach().cpu().numpy()
			term_importances[topic][turn] = ti
	return term_importances

ti_rw = encode(rw)

with open("tirw22.pkl","wb") as f:
	pickle.dump(ti_rw, f, protocol=-1)
