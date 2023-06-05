import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

def stringify(d):
	dstr = dict()
	for k in d:
		dstr[str(k)] = d[k]
	return dstr

class Extr2Dataset(Dataset):
	def __init__(self,qctd,qaptd):
		self.instances = []
		self.qctd = stringify(qctd)
		self.qaptd = stringify(qaptd)
		for topic in self.qctd:
			for turn in self.qctd[topic]:
				#if turn > 1:
				self.instances.append((str(topic),turn))

	def __len__(self):
		return len(self.instances)
	
	def __getitem__(self,index):
		topic, turn = self.instances[index]

		qct = self.qctd[topic][turn]
		qcat_input_ids = torch.tensor(qct['input_ids'],requires_grad=False)
		qcat_attention_mask = torch.tensor(qct['attention_mask'],requires_grad=False)
		qcat_token_type_ids = torch.tensor(qct['token_type_ids'],requires_grad=False)

		qapl = self.qaptd[topic][turn]
		for a_turn in qapl:
			qapl[a_turn]['input_ids'] = torch.tensor(qapl[a_turn]['input_ids'],requires_grad=False)
			qapl[a_turn]['attention_mask'] = torch.tensor(qapl[a_turn]['attention_mask'],requires_grad=False)
			qapl[a_turn]['token_type_ids'] = torch.tensor(qapl[a_turn]['token_type_ids'],requires_grad=False)		
		return {
			'qcat_input_ids':qcat_input_ids,
			'qcat_attention_mask':qcat_attention_mask,
			'qcat_token_type_ids':qcat_token_type_ids,
			'topic':topic,
			'turn':turn,
			'nb_context_items':len(qapl),
			'qapl':qapl
		}

def embed_smart_pad(batch):
	padded_qcat_input_ids = pad_sequence([item['qcat_input_ids'] for item in batch], batch_first=True)
	padded_qcat_attention_mask = pad_sequence([item['qcat_attention_mask'] for item in batch], batch_first=True)
	padded_qcat_token_type_ids = pad_sequence([item['qcat_token_type_ids'] for item in batch], batch_first=True)
	q_batch = [{
		'qcat_input_ids':padded_qcat_input_ids[i],
		'qcat_attention_mask':padded_qcat_attention_mask[i],
		'qcat_token_type_ids':padded_qcat_token_type_ids[i],
		'topic':batch[i]['topic'],
		'turn':batch[i]['turn'],
		'nb_context_items':batch[i]['nb_context_items']
	} for i in range(len(batch))]
	collated_qbatch = default_collate(q_batch)

	if torch.all(collated_qbatch['nb_context_items'] == 0):
		collated_qapbatch = None
	else:
		padded_qap_input_ids = pad_sequence([item['qapl'][a_turn]['input_ids'] for item in batch for a_turn in item['qapl']], batch_first=True)
		padded_qap_attention_mask = pad_sequence([item['qapl'][a_turn]['attention_mask'] for item in batch for a_turn in item['qapl']], batch_first=True)
		padded_qap_token_type_ids = pad_sequence([item['qapl'][a_turn]['token_type_ids'] for item in batch for a_turn in item['qapl']], batch_first=True)
		qap_batch = [{
			'qap_input_ids':padded_qap_input_ids[i],
			'qap_attention_mask':padded_qap_attention_mask[i],
			'qap_token_type_ids':padded_qap_token_type_ids[i]
		}  for i in range(len(padded_qap_input_ids))]

		collated_qapbatch = default_collate(qap_batch)

	return collated_qbatch, collated_qapbatch
