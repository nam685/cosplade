import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np

class CatDataset(Dataset):
	def __init__(self,tokens_dict):
		self.instances = []
		self.tokens_dict = tokens_dict
		for topic in self.tokens_dict:
			for turn in self.tokens_dict[topic]:
				self.instances.append((topic,turn))

	def __len__(self):
		return len(self.instances)
	
	def __getitem__(self,index):
		topic, turn = self.instances[index]
		data = self.tokens_dict[topic][turn]
		input_ids = torch.tensor(data['input_ids'])
		attention_mask = torch.tensor(data['attention_mask'],requires_grad=False)
		token_type_ids = torch.tensor(data['token_type_ids'],requires_grad=False)
		return {\
			'input_ids':input_ids,\
			'attention_mask':attention_mask,\
			'token_type_ids':token_type_ids,\
			'topic':topic,\
			'turn':turn\
		}

def embed_smart_pad(batch):
	padded_input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
	padded_attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
	padded_token_type_ids = pad_sequence([item['token_type_ids'] for item in batch], batch_first=True)
	batch = [{\
		'input_ids':padded_input_ids[i],\
		'attention_mask':padded_attention_mask[i],\
		'token_type_ids':padded_token_type_ids[i],\
		'topic':batch[i]['topic'],\
		'turn':batch[i]['turn']\
	} for i in range(len(batch))]
	return default_collate(batch)