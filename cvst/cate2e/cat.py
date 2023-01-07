import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def train(model, device, optimizer, loader, supervision, wandb, loss_fn):
	model.train()
	for i,data in tqdm(enumerate(loader)):
		input_ids = data['input_ids'].to(device, dtype = torch.long)
		mask = data['attention_mask'].to(device, dtype = torch.long)
		token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
		topics = np.array(data['topic'])
		turns = np.array(data['turn'])

		preds = model(input_ids=input_ids,attention_mask=mask,token_type_ids=token_type_ids)
		targets = torch.tensor(np.concatenate(\
			[supervision[topic][turn] for (topic,turn) in zip(topics,turns)], axis=0), requires_grad=False).to(device)

		loss = F.mse_loss(preds, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i%10 == 0:
			wandb.log({"train loss": loss.item()})

def eval(model, device, loader, supervision, wandb, save_emb, loss_fn):
	model.eval()
	with torch.no_grad():
		ti = {}
		for i,data in tqdm(enumerate(loader)):
			input_ids = data['input_ids'].to(device, dtype = torch.long)
			mask = data['attention_mask'].to(device, dtype = torch.long)
			token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
			topics = np.array(data['topic'])
			turns = data['turn'].numpy()

			preds = model(input_ids=input_ids,attention_mask=mask,token_type_ids=token_type_ids)
			targets = torch.tensor(np.concatenate(\
				[supervision[topic][turn] for (topic,turn) in zip(topics,turns)], axis=0), requires_grad=False).to(device)

			loss = F.mse_loss(preds, targets)

			if i%10 == 0:
				wandb.log({"val loss": loss.item()})

			if save_emb:
				for i in range(len(topics)):
					if topics[i] not in ti:
						ti[topics[i]] = {}
					emb = preds[i]
					ti[topics[i]][turns[i]] = emb.detach().cpu().numpy()
	return ti