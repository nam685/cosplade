import torch
from transformers import AutoModelForMaskedLM

class CatSplade(torch.nn.Module):
	def __init__(self,query_only=False):
		super(CatSplade,self).__init__()
		self.query_only=query_only
		self.model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")

	def forward(self, input_ids, attention_mask, token_type_ids):
		output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
		if self.query_only:
			output = torch.max(\
				torch.log(1 + torch.relu(output)) * attention_mask.unsqueeze(-1) * token_type_ids.unsqueeze(-1),\
			dim=1).values
		else:
			output = torch.max(\
				torch.log(1 + torch.relu(output)) * attention_mask.unsqueeze(-1),\
			dim=1).values
		return output
