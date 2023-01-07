from transformers import AutoModelForSeq2SeqLM
import pickle
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import json
from transformers import AutoTokenizer
import wandb
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from rerank_data import *

print("Loading data files")

with open(f"/data/lenam/cvst/rerank/rrdvkwqs.pkl","rb") as f:
	dv_ins ,dvcn = pickle.load(f)
with open(f"/data/lenam/cvst/rerank/rrtskwqs.pkl","rb") as f:
	ts_ins, tscn = pickle.load(f)
with open(f"/data/lenam/cvst/rerank/rrtrkwqs.pkl","rb") as f:
	tr_ins, trcn = pickle.load(f)
with open("/data/lenam/cvst/rerank/rr20kw_pQs.pkl","rb") as f:
	t20_ins, t20_cdd = pickle.load(f)
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
	def __init__(self,instances,candidates,pre_tokenize=True):
		self.pre_tokenize = pre_tokenize
		self.tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")	
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

class RerankFinetunePairDataset(Dataset):
	def __init__(self,instances,candidates,pre_tokenize=True,nb_weak=1):
		self.pre_tokenize = pre_tokenize
		self.tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")	
		self.instances = list(set([qid for qid,docid in instances]))
		self.candidates = candidates
		self.nb_weak = nb_weak
	def __len__(self):
		return len(self.instances)
	def __getitem__(self,index):
		qid = self.instances[index]
		docids = list(self.candidates[qid].keys())
		strong_docid = np.random.choice(docids[:3])
		weak_docids = np.random.choice(docids[3:],size=self.nb_weak).tolist()
		docids = [strong_docid] + weak_docids
		pairs = [(qid,docid) for docid in docids]
		if self.pre_tokenize:
			tokens = [self.candidates[qid][docid] for docid in docids]
		else:
			tokens = [self.tokenizer(self.candidates[qid][docid],truncation=True,return_tensors='pt') for docid in docids]
		return {'pairs':pairs,'tokens':tokens}

def embed_smart_pad_pair(batch):
	padded_input_ids = pad_sequence([tokens['input_ids'].squeeze() for item in batch for tokens in item['tokens']], batch_first=True)
	padded_attention_mask = pad_sequence([tokens['attention_mask'].squeeze() for item in batch for tokens in item['tokens']], batch_first=True)
	pairs = [pair for item in batch for pair in item['pairs']]
	flattened_batch = [{
		'input_ids':padded_input_ids[i],
		'attention_mask':padded_attention_mask[i],
		'pairs':pairs[i]
	} for i in range(len(pairs))]
	return default_collate(flattened_batch)

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

def msemargin(scores, targets):
	scores_margin = scores[:,0] - scores[:,1]
	target_margin = targets[:,0] - targets[:,1]
	return F.mse_loss(target_margin, scores_margin)

def kldiv(scores, targets):
	scores_logproba = F.log_softmax(scores,dim=1)
	targets_proba = F.normalize(targets,p=1,dim=1)
	return F.kl_div(scores_logproba, targets_proba, reduction='batchmean')

def train(model,loader,device,supervision,optimizer,wandb,loss_fn,nb_weak):
	model.train()
	for i,data in tqdm(enumerate(loader)):
		pairs = np.array(data['pairs'])
		scores = rescore(model,data,device).reshape((-1,1+nb_weak))
		targets = torch.tensor([supervision[qid][docid] for (qid,docid) in zip(data['pairs'][0],data['pairs'][1])], requires_grad=False).to(device).reshape((-1,1+nb_weak))
		loss = loss_fn(scores,targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i%5 == 0:
			wandb.log({"train loss":loss.item()})

def eval_pair(model,loader,device,supervision,wandb,name,loss_fn,nb_weak):
	model.eval()
	with torch.no_grad():
		for i,data in tqdm(enumerate(loader)):
			pairs = np.array(data['pairs'])
			scores = rescore(model,data,device).reshape((-1,1+nb_weak))
			targets = torch.tensor([supervision[qid][docid] for (qid,docid) in zip(data['pairs'][0],data['pairs'][1])], requires_grad=False).to(device).reshape((-1,1+nb_weak))
			loss = loss_fn(scores,targets)
			if i%5 == 0:
				wandb.log({"val loss":loss.item()})

def inference(model,loader,device,name):
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

random_seed = 7632
wandb.init(project="convers1")
config = wandb.config
config.SEED = random_seed
config.EPOCHS = 0
config.LEARNING_RATE = 1e-4
config.TRAIN_BATCH_SIZE = 8
config.VAL_BATCH_SIZE = 16
config.NB_WEAK = 1
config.note = f'finetune pair kwpqs20 monot5 rerank msemargin'
print(random_seed,config.note)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

#model = AutoModelForSeq2SeqLM.from_pretrained("castorini/monot5-base-msmarco-10k")
model = AutoModelForSeq2SeqLM.from_pretrained(f"/data/lenam/cvst/model/{random_seed}")
model.to(device)
#wandb.watch(model, log="all")

print("Building datasets")

#tr_ds = RerankFinetunePairDataset(tr_ins+dv_ins+ts_ins, {**trcn,**dvcn,**tscn}, pre_tokenize=False, nb_weak=config.NB_WEAK)
t20_ds = RerankFinetuneDataset(t20_ins,t20_cdd)
#t21_ds = RerankFinetuneDataset(t21_ins,t21_cdd)
#ptc_ds = RerankFinetunePairDataset(t20_ins+t21_ins, {**t20_cdd,**t21_cdd}, pre_tokenize=True, nb_weak=config.NB_WEAK)
#ts_ds = RerankFinetuneDataset(t22_ins, t22_cdd)

#tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE, collate_fn=embed_smart_pad_pair)
t20_dl = DataLoader(t20_ds, shuffle=False, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
#t21_dl = DataLoader(t21_ds, shuffle=False, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
#ptc_dl = DataLoader(ptc_ds, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE, collate_fn=embed_smart_pad_pair)
#ts_dl = DataLoader(ts_ds, shuffle=False, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)

#optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

print("Start training")

#for epoch in range(1,1+config.EPOCHS):
	#train(model,tr_dl,device,{**rrtr_golden,**rrdv_golden,**rrts_golden},optimizer,wandb, msemargin, nb_weak=config.NB_WEAK)
inference(model,t20_dl,device, f'rrt21_kwqft_mm{random_seed}')
#inference(model,t21_dl,device, f'rrt21_kwqft_mm{random_seed}')
#inference(model, ts_dl, device, f'rrt22_kwqft_mm{random_seed}')

#model.save_pretrained(f"/data/lenam/cvst/model/{random_seed}")
