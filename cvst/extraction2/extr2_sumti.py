import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from torch import autograd

def vect_rwtk(d,device,dim=30522):
	v = dict()
	for topic in d:
		v[topic] = dict()
		for turn in d[topic]:
			a = torch.zeros(dim,dtype=torch.int,device=device)
			a[d[topic][turn]] = True
			v[topic][turn] = a
	return v

def cos_loss(target, prediction, l1_reg=0):
	reg = l1_reg * prediction.abs().sum(dim=-1).mean()
	cl = -(F.normalize(target,dim=-1) * F.normalize(prediction,dim=-1)).sum(dim=-1).mean()
	return reg, cl

def train(model, device, optimizer, loader, supervision, golden_tokens_dict, wandb):
	golden_vectors = vect_rwtk(golden_tokens_dict, device)
	model.train()
	for i,data in tqdm(enumerate(loader)):
		qcat_batch, qap_batch = data
		if torch.all(qcat_batch['nb_context_items'] == 0):
			continue
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

		#with autograd.detect_anomaly():
		qcat_term_importance, qap_term_importance = model(data)
		term_importance = qcat_term_importance + qap_term_importance
		targets = torch.tensor(np.concatenate(\
			[supervision[topic][turn] for (topic,turn) in zip(topics,turns)], axis=0), requires_grad=False).to(device)
		gold_tokens = torch.cat([golden_vectors[topic][turn].unsqueeze(0) for (topic,turn) in zip(topics,turns)], dim=0).to(device)
		gold_importance = targets * gold_tokens

		qcat_loss = F.mse_loss(qcat_term_importance, targets)
		qap_loss = F.mse_loss(qap_term_importance, targets)
		ti_loss = F.mse_loss(term_importance, targets)
		#reg, cl = cos_loss(targets, term_importance)
		gold_loss = torch.mean(torch.relu(gold_tokens * (gold_importance - qap_term_importance))**2)

		loss = ti_loss + 0.5*gold_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i%5 == 0:
			wandb.log({
				"max qap":qap_term_importance.max(),
				"max qcat":qcat_term_importance.max(),
				"train qcat loss":qcat_loss.item(),
				"train qap loss":qap_loss.item(),
				#"train qap loss 3":qap3_loss.item(),
				#"train flops":qap3_flops.item(),
				"train ti loss":ti_loss.item(),
				"train gold loss":gold_loss.item(),
				#"l1reg loss":reg.item(),
				#"cos loss":cl.item()
			})

def eval(model, device, loader, supervision, golden_tokens_dict, wandb, partition="", save_emb=False, name=None):
	golden_vectors = vect_rwtk(golden_tokens_dict, device)
	model.eval()
	with torch.no_grad():
		count_log = 0
		ti = {}
		qcat_ti = {}
		qap_ti = {}
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

			#with autograd.detect_anomaly():
			qcat_term_importance, qap_term_importance = model(data)
			term_importance = qcat_term_importance + qap_term_importance
			targets = torch.tensor(np.concatenate(\
				[supervision[topic][turn] for (topic,turn) in zip(topics,turns)], axis=0), requires_grad=False).to(device)
			gold_tokens = torch.cat([golden_vectors[topic][turn].unsqueeze(0) for (topic,turn) in zip(topics,turns)], dim=0).to(device)
			gold_importance = targets * gold_tokens

			qcat_loss = F.mse_loss(qcat_term_importance, targets)
			qap_loss = F.mse_loss(qap_term_importance, targets)
			ti_loss = F.mse_loss(term_importance, targets)
			#reg, cl = cos_loss(targets, term_importance)
			gold_loss = torch.mean(torch.relu(gold_tokens * (gold_importance - qap_term_importance))**2)

			loss = ti_loss + 0.5*gold_loss

			if i%5 == 0:
				wandb.log({
					"max qap":qap_term_importance.max(),
					"max qcat":qcat_term_importance.max(),
					"val qcat loss":qcat_loss.item(),
					"val qap loss":qap_loss.item(),
					#"val qap loss 3":qap3_loss.item(),
					#"val flops":qap3_flops.item(),
					"val ti loss":ti_loss.item(),
					"val gold loss":gold_loss.item(),
					#"l1reg loss":reg.item(),
					#"cos loss":cl.item()
				})

			if save_emb:
				for i in range(len(topics)):
					if topics[i] not in ti:
						ti[topics[i]] = {}
						qcat_ti[topics[i]] = {}
						qap_ti[topics[i]] = {}
					ti[topics[i]][turns[i]] = term_importance[i].detach().cpu().numpy()
					qcat_ti[topics[i]][turns[i]] = qcat_term_importance[i].detach().cpu().numpy()
					qap_ti[topics[i]][turns[i]] = qap_term_importance[i].detach().cpu().numpy()

	if save_emb:
		with open(f"/data/lenam/cvst/emb/pred_ti_{partition}_{name}.pkl","wb") as f:
			pickle.dump(ti,f,protocol=-1)
		with open(f"/data/lenam/cvst/emb/pred_qcat_ti_{partition}_{name}.pkl","wb") as f:
			pickle.dump(qcat_ti,f,protocol=-1)
		with open(f"/data/lenam/cvst/emb/pred_qap_ti_{partition}_{name}.pkl","wb") as f:
			pickle.dump(qap_ti,f,protocol=-1)
