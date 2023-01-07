import pickle
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
from tqdm import tqdm
from rerank_data import *
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoTokenizer
import wandb
from transformers import AutoModelForSeq2SeqLM

print("Loading data files")

with open(f"/data/lenam/cvst/rerank/rrdvcn_kw_pQs5.pkl","rb") as f:
	dv_ins ,dvcn = pickle.load(f)
with open(f"/data/lenam/cvst/rerank/rrtscn_kw_pQs5.pkl","rb") as f:
	ts_ins, tscn = pickle.load(f)
with open(f"/data/lenam/cvst/rerank/rrtrcn_kw_pQs5.pkl","rb") as f:
	tr_ins, trcn = pickle.load(f)
with open("/data/lenam/cvst/rerank/rr21kw_pQs5.pkl","rb") as f:
	t21_ins, t21_cdd = pickle.load(f)
#with open("/data/lenam/cvst/rerank/rr21kw_pQs.pkl","rb") as f:
#	t21_ins, t21_cdd = pickle.load(f)
#with open("/data/lenam/cvst/rerank/rr22kw_pQs.pkl","rb") as f:
#	t22_ins, t22_cdd = pickle.load(f)

with open("/data/lenam/eval/runs/rrdvcn_golden.json",'r') as f:
	rrdv_golden = json.load(f)
with open("/data/lenam/eval/runs/rrtscn_golden.json",'r') as f:
	rrts_golden = json.load(f)
with open("/data/lenam/eval/runs/rrtrcn_golden.json",'r') as f:
	rrtr_golden = json.load(f)
#with open("/data/lenam/eval/runs/rr20_6529_golden.json",'r') as f:
#	rrt20_golden = json.load(f)
#with open("/data/lenam/eval/runs/rr21_6529_golden.json",'r') as f:
#	rrt21_golden = json.load(f)

class RerankFinetuneDataset(Dataset):
	def __init__(self,instances,candidates,pre_tokenize=True, sample=False):
		self.pre_tokenize = pre_tokenize
		self.tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")	
		if sample:
			idx = np.random.choice(len(instances), size=len(instances)//40, replace=False)
			self.instances = np.array(instances)[idx]
		else:
			self.instances = instances
		self.candidates = candidates
	def __len__(self):
		return len(self.instances)
	def __getitem__(self,index):
		qid,docid = self.instances[index]
		if self.pre_tokenize:
			tokens = self.candidates[qid][docid]
		else:
			text = self.candidates[qid][docid]
			tokens = self.tokenizer(text,truncation=True,return_tensors='pt')
		return {**tokens,'qid':qid,'docid':docid}

print("Building datasets")

tr_ds = RerankFinetuneDataset(tr_ins+dv_ins+ts_ins, {**trcn,**dvcn,**tscn}, pre_tokenize=False)
#ptc_ds = RerankFinetuneDataset(t20_ins+t21_ins, {**t20_cdd,**t21_cdd})
t21_ds = RerankFinetuneDataset(t21_ins, t21_cdd)
#ts_ds = RerankFinetuneDataset(t22_ins, t22_cdd)

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

def train(model,loader,device,supervision,optimizer,wandb):
	model.train()
	for i,data in tqdm(enumerate(loader)):
		qids = np.array(data['qid'])
		docids = np.array(data['docid'])
		scores = rescore(model,data,device)
		targets = torch.tensor([supervision[qid][docid] for (qid,docid) in zip(qids,docids)], requires_grad=False).to(device)
		loss = F.mse_loss(scores,targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i%5 == 0:
			wandb.log({"train loss":loss.item()})

def eval(model,loader,device,supervision,wandb,name):
	model.eval()
	rankings = dict()
	with torch.no_grad():
		for i,data in tqdm(enumerate(loader)):
			qids = np.array(data['qid'])
			docids = np.array(data['docid'])
			scores = rescore(model,data,device)
			targets = torch.tensor([supervision[qid][docid] for (qid,docid) in zip(qids,docids)], requires_grad=False).to(device)
			loss = F.mse_loss(scores,targets)
			if i%5 == 0:
				wandb.log({"val loss":loss.item()})
			for i in range(len(qids)):
				if qids[i] not in rankings:
					rankings[qids[i]] = dict()
				rankings[qids[i]][docids[i]] = scores[i].item()
	with open(f"/data/lenam/eval/runs/{name}.json","w") as f:
		json.dump(rankings,f)

def inference(model,loader,device,wandb,name):
	model.eval()
	rankings = dict()
	with torch.no_grad():
		for i,data in tqdm(enumerate(loader)):
			qids = np.array(data['qid'])
			docids = np.array(data['docid'])
			scores = rescore(model,data,device)
			for i in range(len(qids)):
				if qids[i] not in rankings:
					rankings[qids[i]] = dict()
				rankings[qids[i]][docids[i]] = scores[i].item()
	with open(f"/data/lenam/eval/runs/{name}.json","w") as f:
		json.dump(rankings,f)

print("Setting up")

random_seed = np.random.choice(9999)
wandb.init(project="convers1")
config = wandb.config
config.SEED = random_seed
config.EPOCHS = 1
config.LEARNING_RATE = 1e-4
config.TRAIN_BATCH_SIZE = 16
config.VAL_BATCH_SIZE = 32
config.note = f'finetune kwpQs5 monot5 rerank'
print(random_seed,config.note)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

model = AutoModelForSeq2SeqLM.from_pretrained("castorini/monot5-base-msmarco-10k")
model.to(device)
wandb.watch(model, log="all")

tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE, collate_fn=embed_smart_pad)
t21_dl = DataLoader(t21_ds, shuffle=False, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
#ptc_dl = DataLoader(ptc_ds, shuffle=False, batch_size=config.TRAIN_BATCH_SIZE, collate_fn=embed_smart_pad)
#ts_dl = DataLoader(ts_ds, shuffle=False, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

print("Start training")

#model = AutoModelForSeq2SeqLM.from_pretrained(f"/data/lenam/cvst/model/{random_seed}")
#model.to(device)

for epoch in range(1,1+config.EPOCHS):
	train(model,tr_dl,device,{**rrtr_golden,**rrdv_golden,**rrts_golden},optimizer,wandb)
	inference(model,t21_dl,device,wandb, f'rrt21_kwpQs{random_seed}')
    #train(model,ptc_dl,device,{**rrt20_golden,**rrt21_golden},optimizer,wandb)
	#inference(model, ts_dl, device, wandb, f'rrt22_kwpQs{random_seed}')

model.save_pretrained(f"/data/lenam/cvst/model/{random_seed}")
