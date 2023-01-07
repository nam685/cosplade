import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm
import numpy as np
import pickle
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

year = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
model.eval()
model.to(device)
with open(f"treccast_{year}.pkl","rb") as f:
	tc = pickle.load(f)
test_q, test_rw, test_rp = tc
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.last_hidden_state.detach()
    return hook
h = model.bert.register_forward_hook(get_activation("bert"))
def encode(d):
	dense_embeddings = dict()
	term_importances = dict()
	for topic in tqdm(d):
		dense_embeddings[topic] = dict()
		term_importances[topic] = dict()
		for turn in d[topic]:
			tokens = tokenizer(d[topic][turn])
			out = model(\
					torch.tensor(tokens['input_ids']).unsqueeze(0).to(device),\
					torch.tensor(tokens['attention_mask']).unsqueeze(0).to(device)\
				).logits
			values, _ = torch.max(torch.log(1 + torch.relu(out)) * torch.tensor(tokens["attention_mask"]).unsqueeze(-1).to(device), dim=1)
			ti = values.detach().cpu().numpy()
			a = activation['bert'].detach().cpu().numpy()
			dense_embeddings[topic][turn] = a
			term_importances[topic][turn] = ti
	return dense_embeddings, term_importances

ts_emb_q, ts_ti_q = encode(test_q)
ts_emb_rw, ts_ti_rw = encode(test_rw)
ts_emb_rp, ts_ti_rp = encode(test_rp)

emb_test = ts_emb_q, ts_emb_rw, ts_emb_rp
ti_test = ts_ti_q, ts_ti_rw, ts_ti_rp

with open(f"emb{year[2:]}.pkl","wb") as f:
	pickle.dump(emb_test, f, protocol=-1)
with open(f"ti{year[2:]}.pkl","wb") as f:
	pickle.dump(ti_test, f, protocol=-1)
