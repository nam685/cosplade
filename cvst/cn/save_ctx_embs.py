from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from tqdm import tqdm
import numpy as np
import pickle
from scipy import sparse
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
model.eval()
model.to(device)
with open("canard_enhanced.pkl","rb") as f:
	cn = pickle.load(f)
cn_train, cn_dev, cn_test = cn
train_q, train_rw, train_rp = cn_train
dev_q, dev_rw, dev_rp = cn_dev
test_q, test_rw, test_rp = cn_test
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
			ti = values.detach().cpu().squeeze().numpy()
			a = activation['bert'].detach().cpu().numpy()
			dense_embeddings[topic][turn] = a
			term_importances[topic][turn] = sparse.csr_matrix(ti)
	return dense_embeddings, term_importances
tr_emb_q, tr_ti_q = encode(train_q)
tr_emb_rw, tr_ti_rw = encode(train_rw)
tr_emb_rp, tr_ti_rp = encode(train_rp)
dv_emb_q, dv_ti_q = encode(dev_q)
dv_emb_rw, dv_ti_rw = encode(dev_rw)
dv_emb_rp, dv_ti_rp = encode(dev_rp)
ts_emb_q, ts_ti_q = encode(test_q)
ts_emb_rw, ts_ti_rw = encode(test_rw)
ts_emb_rp, ts_ti_rp = encode(test_rp)

emb_train = tr_emb_q, tr_emb_rw, tr_emb_rp
emb_dev = dv_emb_q, dv_emb_rw, dv_emb_rp
emb_test = ts_emb_q, ts_emb_rw, ts_emb_rp
ti_train = tr_ti_q, tr_ti_rw, tr_ti_rp
ti_dev = dv_ti_q, dv_ti_rw, dv_ti_rp
ti_test = ts_ti_q, ts_ti_rw, ts_ti_rp

with open("emb_enhanced.pkl","wb") as f:
	pickle.dump((emb_train, emb_dev, emb_test), f, protocol=-1)
with open("ti_enhanced.pkl","wb") as f:
	pickle.dump((ti_train, ti_dev, ti_test), f, protocol=-1)
