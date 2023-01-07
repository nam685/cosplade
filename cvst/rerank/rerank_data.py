import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

def read_queries(queries):
	d = dict()
	for topic in queries:
		for turn in queries[topic]:
			qid = f"{topic}_{turn}"
			d[qid] = queries[topic][turn]
	return d

class RerankDataset(Dataset):
	def __init__(self,data,pre_tokenize=True,sampled=False,instances=None):
		self.pre_tokenize = pre_tokenize
		self.tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco-10k")	
		if 'cache' in data:
			with open(f"/data/lenam/cvst/rerank/{data['cache']}.pkl","rb") as f:
				cache = pickle.load(f)
			self.instances = cache[0]
			self.candidates = cache[1]
		else:
			name = data['name']
			queries = data['queries']
			passages_file = data['passages_file']
			self.instances = []
			queries = read_queries(queries)
			self.candidates = dict()
			for qid in queries:
				self.candidates[qid] = dict()
			with open(f"/data/lenam/cvst/rerank/{passages_file}.txt",'r') as f:
				for line in tqdm(f):
					qid,docid,passage = line.strip().split('\t',2)
					query = queries[qid]
					if len(passage)>1500:
						passage = passage[:1500].rsplit(' ',1)[0] + '... '
					text = f"Query: {query} Document: {passage} Relevant:"
					if self.pre_tokenize:
						tokens = self.tokenizer(text,truncation=True,return_tensors='pt')
						self.candidates[qid][docid] = tokens
					else:
						self.candidates[qid][docid] = text
					self.instances.append((qid,docid))
			if sampled:
				if not instances is None:
					self.instances = instances
					self.candidates = self.sample2(self.candidates)
				self.instances, self.candidates = self.sample(self.instances, self.candidates)
			cache = (self.instances,self.candidates)
			with open(f"/data/lenam/cvst/rerank/{name}.pkl","wb") as f:
				pickle.dump(cache,f,protocol=-1)
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
	def sample(self,instances,candidates,n_strong=3,n_weak=7):
		sampled_instances = []
		sampled_candidates = dict()
		for qid in tqdm(self.candidates):
			docids = np.array(list(self.candidates[qid].keys()))
			chosen_docids = docids[:n_strong].tolist() + np.random.choice(docids[n_strong:],size=n_weak,replace=False).tolist()
			sampled_ranking = {docid:self.candidates[qid][docid] for docid in chosen_docids}
			sampled_candidates[qid] = sampled_ranking
		for qid in sampled_candidates:
			for docid in sampled_candidates[qid]:
				sampled_instances.append((qid,docid))
		return sampled_instances, sampled_candidates
	def sample2(self,candidates,n_strong=3,n_weak=7):
		sampled_candidates = dict()
		for (qid,docid) in tqdm(self.instances):
			if qid not in sampled_candidates:
				sampled_candidates[qid] = dict()
			sampled_candidates[qid][docid] = self.candidates[qid][docid]
		return sampled_candidates

def embed_smart_pad(batch):
	padded_input_ids = pad_sequence([item['input_ids'].squeeze() for item in batch], batch_first=True)
	padded_attention_mask = pad_sequence([item['attention_mask'].squeeze() for item in batch], batch_first=True)
	batch = [{
		'input_ids':padded_input_ids[i],
		'attention_mask':padded_attention_mask[i],
		'qid':batch[i]['qid'],
		'docid':batch[i]['docid'],
	} for i in range(len(batch))]
	return default_collate(batch)
