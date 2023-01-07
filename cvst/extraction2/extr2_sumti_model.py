import torch
from transformers import BertModel,BertConfig,AutoModelForMaskedLM
from torch.nn.utils.rnn import pad_sequence
from torch import cuda
import torch.nn.functional as F
device = 'cuda' if cuda.is_available() else 'cpu'
from torch import autograd
import pickle

class Extr2SpladeSumti(torch.nn.Module):
	def __init__(self):
		super(Extr2SpladeSumti,self).__init__()

		self.SPLADE_VERSION = "naver/splade-cocondenser-ensembledistil"
		self.qcat_encoder = AutoModelForMaskedLM.from_pretrained(self.SPLADE_VERSION)
		self.qap_encoder = AutoModelForMaskedLM.from_pretrained(self.SPLADE_VERSION)

	def forward(self, data):
		qcat_batch, qap_batch = data
		batch_size = qcat_batch['qcat_input_ids'].shape[0]
		ni = qcat_batch['nb_context_items']

		qcat_term_importance = self.qcat_encoder(
			input_ids=qcat_batch['qcat_input_ids'],
			attention_mask=qcat_batch['qcat_attention_mask'],
			token_type_ids=qcat_batch['qcat_token_type_ids']
		).logits
		qcat_term_importance = qcat_term_importance * qcat_batch['qcat_attention_mask'].unsqueeze(-1) # * (1-qcat_batch['qcat_token_type_ids']).unsqueeze(-1)
		qcat_term_importance = torch.max(torch.log(1 + torch.relu(qcat_term_importance)),dim=1).values

		qap_term_importance = self.qap_encoder(
			input_ids=qap_batch['qap_input_ids'],
			attention_mask=qap_batch['qap_attention_mask'],
			token_type_ids=qap_batch['qap_token_type_ids']
		).logits
		qap_term_importance = qap_term_importance * qap_batch['qap_attention_mask'].unsqueeze(-1) # * (1-qap_batch['qap_token_type_ids']).unsqueeze(-1)
		
		# Aggregate past turns
		#with autograd.detect_anomaly():
		cs = torch.cumsum(ni,dim=0)
		r = torch.arange(cs.max(), device=device).repeat(cs.size(0),1)
		encoding_indices = (r >= cs.unsqueeze(-1)).sum(dim=0)
		oh = F.one_hot(encoding_indices, num_classes=cs.size(0)).float()
		qap_term_importance = torch.movedim(torch.matmul(oh.T, torch.movedim(qap_term_importance,-1,0)),0,-1)
		ni[ni==0]=1
		qap_term_importance = qap_term_importance / ni[:,None,None]

		qap_term_importance = torch.max(torch.log(1 + torch.relu(qap_term_importance)),dim=1).values

		return qcat_term_importance, qap_term_importance

	def save_pretrained(self,path):
		self.qcat_encoder.save_pretrained(save_directory=f"{path}/splade_encoder")
		self.qap_encoder.save_pretrained(save_directory=f"{path}/cross_encoder")

	def load_pretrained(self,path):
		self.qcat_encoder = AutoModelForMaskedLM.from_pretrained(f"{path}/splade_encoder")
		self.qap_encoder = AutoModelForMaskedLM.from_pretrained(f"{path}/cross_encoder")