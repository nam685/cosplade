import pickle
from tqdm import tqdm

with open("/data/lenam/cvst/reformulation/tc20_txt_3.pkl","rb") as f:
	rf20 = pickle.load(f)
with open("/data/lenam/cvst/reformulation/tc21_txt_3.pkl","rb") as f:
	rf21 = pickle.load(f)
with open("/data/lenam/cvst/reformulation/val_txt_3.pkl","rb") as f:
	val = pickle.load(f)
with open("/data/lenam/cvst/reformulation/train_txt_3.pkl","rb") as f:
	train = pickle.load(f)

def build_inputs(d):
	inputs = dict()
	for topic in tqdm(d):
		inputs[topic] = dict()
		for turn in d[topic]:
			q, ctx, rw = d[topic][turn]
			inputs[topic][turn] = q + ' Context: ' + ctx
	return inputs

def split_inputs(tr_inputs):
	dv_inputs = {k:v for k,v in list(tr_inputs.items())[4383:]}
	tr_inputs = {k:v for k,v in list(tr_inputs.items())[:4383]}
	return tr_inputs, dv_inputs

print("Building inputs")
tr_inputs = build_inputs(train)
tr_inputs, dv_inputs = split_inputs(tr_inputs)
ts_inputs = build_inputs(val)
t20_inputs = build_inputs(rf20)
t21_inputs = build_inputs(rf21)

with open(f"/data/lenam/cvst/rfml_keyword/tr_inputs7.pkl","wb") as f:
	pickle.dump(tr_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/dv_inputs7.pkl","wb") as f:
	pickle.dump(dv_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/ts_inputs7.pkl","wb") as f:
	pickle.dump(ts_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/t20_inputs7.pkl","wb") as f:
	pickle.dump(t20_inputs,f,protocol=-1)
with open(f"/data/lenam/cvst/rfml_keyword/t21_inputs7.pkl","wb") as f:
	pickle.dump(t21_inputs,f,protocol=-1)