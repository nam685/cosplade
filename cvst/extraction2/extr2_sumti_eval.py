from extr2_data import *
from extr2_sumti_model import *
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import pickle
import sys

if len(sys.argv)!=2:
	print("python extr2_sumti_eval.py <seed>")
seed = int(sys.argv[1])

VAL_BATCH_SIZE = 32
FORMAT = '_enhanced_aqla'

model_path = f"/data/lenam/cvst/model/{seed}"
model = Extr2SpladeSumti()
model.load_pretrained(model_path)
model.to(device)

def stringify(d):
	dstr = dict()
	for k in d:
		dstr[str(k)] = d[k]
	return dstr

with open(f"/data/lenam/cvst/extraction2/tokens{FORMAT}.pkl","rb") as f:
	tokens = pickle.load(f)

with open("/data/lenam/cvst/cn/ti.pkl","rb") as f:
	ti = pickle.load(f)

ti_train, ti_dev, ti_test = ti
_, tr_ti_rw, _ = ti_train
_, dv_ti_rw, _ = ti_dev
_, ts_ti_rw, _ = ti_test
with open("/data/lenam/cvst/cn/rw_tokens.pkl","rb") as f:
	tr_tks_rw, dv_tks_rw, ts_tks_rw = pickle.load(f)

with open("/data/lenam/cvst/tc/ti20.pkl","rb") as f:
	ti = pickle.load(f)

_, tc20_ti_rw, _ = ti
tc20_ti_rw = stringify(tc20_ti_rw)
with open("/data/lenam/cvst/tc/ti21.pkl","rb") as f:
	ti = pickle.load(f)

_, tc21_ti_rw, _ = ti
tc21_ti_rw = stringify(tc21_ti_rw)
with open("/data/lenam/cvst/tc/rw_tokens.pkl","rb") as f:
	t20_tks_rw, t21_tks_rw = pickle.load(f)

t20_tks_rw = stringify(t20_tks_rw)
t21_tks_rw = stringify(t21_tks_rw)

tr_ds = Extr2Dataset(tokens["trqct"],tokens["trqapt"])
tr_dl = DataLoader(tr_ds, batch_size=VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
dv_ds = Extr2Dataset(tokens["dvqct"],tokens["dvqapt"])
dv_dl = DataLoader(dv_ds, batch_size=VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
ts_ds = Extr2Dataset(tokens["tsqct"],tokens["tsqapt"])
ts_dl = DataLoader(ts_ds, batch_size=VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
tc20_ds = Extr2Dataset(tokens["t20qct"],tokens["t20qapt"])
tc20_dl = DataLoader(tc20_ds, batch_size=VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
tc21_ds = Extr2Dataset(tokens["t21qct"],tokens["t21qapt"])
tc21_dl = DataLoader(tc21_ds, batch_size=VAL_BATCH_SIZE, collate_fn=embed_smart_pad)

def vect_rwtk(d,device,dim=30522):
	v = dict()
	for topic in d:
		v[topic] = dict()
		for turn in d[topic]:
			a = torch.zeros(dim,dtype=torch.int,device=device)
			a[d[topic][turn]] = True
			v[topic][turn] = a
	return v

def eval(model, device, loader, supervision, golden_tokens_dict, partition="", name=""):
	golden_vectors = vect_rwtk(golden_tokens_dict, device)
	model.eval()
	with torch.no_grad():
		ti = {}
		for i,data in tqdm(enumerate(loader)):
			qcat_batch, qap_batch = data
			qcat_batch['qcat_input_ids'] = qcat_batch['qcat_input_ids'].to(device, dtype = torch.int)
			qcat_batch['qcat_attention_mask'] = qcat_batch['qcat_attention_mask'].to(device, dtype = torch.bool)
			qcat_batch['qcat_token_type_ids'] = qcat_batch['qcat_token_type_ids'].to(device, dtype = torch.int)
			topics = np.array(qcat_batch['topic'])
			turns = np.array(qcat_batch['turn'])
			qcat_batch['nb_context_items'] = qcat_batch['nb_context_items'].to(device, dtype = torch.int)
			qap_batch['qap_input_ids'] = qap_batch['qap_input_ids'].to(device, dtype = torch.int)
			qap_batch['qap_attention_mask'] = qap_batch['qap_attention_mask'].to(device, dtype = torch.bool)
			qap_batch['qap_token_type_ids'] = qap_batch['qap_token_type_ids'].to(device, dtype = torch.int)
			data = qcat_batch, qap_batch
			qcat_term_importance, qap_term_importance = model(data)
			term_importance = qcat_term_importance + qap_term_importance
			for i in range(len(topics)):
				if topics[i] not in ti:
					ti[topics[i]] = {}
				ti[topics[i]][turns[i]] = term_importance[i].detach().cpu().numpy()
	with open(f"/data/lenam/cvst/emb/pred_ti_{partition}_{name}.pkl","wb") as f:
		pickle.dump(ti,f,protocol=-1)
#eval(model, device, tr_dl, tr_ti_rw, tr_tks_rw, partition='tr', name=f"{seed}")
#eval(model, device, dv_dl, dv_ti_rw, dv_tks_rw, partition='dv', name=f"{seed}")
#eval(model, device, ts_dl, ts_ti_rw, ts_tks_rw, partition='ts', name=f"{seed}")
eval(model, device, tc20_dl, tc20_ti_rw, t20_tks_rw, partition='t20', name=f"{seed}")
eval(model, device, tc21_dl, tc21_ti_rw, t21_tks_rw, partition='t21', name=f"{seed}")
