import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import wandb
from torch import cuda
from tqdm import tqdm
device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)

def my_collate(batch):
	padded_source_ids = pad_sequence([item['source_ids'] for item in batch], batch_first=True)
	padded_source_mask = pad_sequence([item['source_mask'] for item in batch], batch_first=True)
	padded_target_ids = pad_sequence([item['target_ids'] for item in batch], batch_first=True)
	batch = [{'source_ids':padded_source_ids[i],
			'topic':batch[i]['topic'],
			'turn':batch[i]['turn'],
			'source_mask':padded_source_mask[i],
			'target_ids':padded_target_ids[i]}
		for i in range(len(padded_source_ids))]
	return default_collate(batch)

class DatasetCQR(Dataset):
	def __init__(self, tkdict):
		self.instances = []
		self.tkdict = tkdict
		for topic in self.tkdict:
			for turn in self.tkdict[topic]:
				self.instances.append((topic,turn))

	def __len__(self):
		return len(self.instances)

	def __getitem__(self, index):
		topic, turn = self.instances[index]
		source, target = self.tkdict[topic][turn]
		return {
			"topic": topic,
			"turn": turn,
			"source_ids": torch.tensor(source['input_ids'], requires_grad=False),
			"source_mask": torch.tensor(source['attention_mask'], requires_grad=False),
			"target_ids": torch.tensor(target['input_ids'], requires_grad=False)
		}

def train(model, device, loader, optimizer, tokenizer, wandb):
	model.train()
	for i, data in tqdm(enumerate(loader, 0)):
		y = data['target_ids'].to(device, dtype=torch.long)
		decoder_input_ids = y[:, :-1].contiguous()
		labels = y[:, 1:].clone().detach()
		labels[y[:, 1:] == tokenizer.pad_token_id] = -100
		input_ids = data['source_ids'].to(device, dtype=torch.long)
		attention_mask = data['source_mask'].to(device, dtype=torch.long)

		outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
		loss = outputs[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i%10 == 0:
			wandb.log({"Train Loss": loss.item()})

def val(model, device, loader, tokenizer, wandb, save):
	model.eval()
	predictions = dict()
	with torch.no_grad():
		for i, data in tqdm(enumerate(loader, 0)):
			topics = np.array(data['topic'])
			turns = np.array(data['turn'])
			y = data['target_ids'].to(device, dtype=torch.long)
			input_ids = data['source_ids'].to(device, dtype=torch.long)
			attention_mask = data['source_mask'].to(device, dtype=torch.long)
			
			# Teacher forcing loss
			decoder_input_ids = y[:, :-1].contiguous()
			labels = y[:, 1:].clone().detach()
			labels[y[:, 1:] == tokenizer.pad_token_id] = -100
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
			loss = outputs[0]
			if i%10 == 0:
				wandb.log({"Val Loss": loss.item()})

			# Generate
			generated_ids = model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_length=100,
				num_beams=2,
			)
			generated_texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
			for topic, turn, text in zip(topics, turns, generated_texts):
				if topic not in predictions:
					predictions[topic] = dict()
				predictions[topic][turn] = text

	with open(f"/data/lenam/cvst/reformulation/generated/{save}","wb") as f:
		pickle.dump(predictions,f,protocol=-1)

random_seed = np.random.randint(9999)
wandb.init(project="convers1")
config = wandb.config
config.note = f't5cqr'
config.SEED = random_seed
config.EPOCHS = 6
config.LEARNING_RATE = 1e-4
config.TRAIN_BATCH_SIZE = 16
config.VAL_BATCH_SIZE = 32
config.TYPE = 3
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

with open(f"/data/lenam/cvst/reformulation/train_tk_3.pkl","rb") as f:
	train_tk = pickle.load(f)
with open(f"/data/lenam/cvst/reformulation/val_tk_3.pkl","rb") as f:
	val_tk = pickle.load(f)
with open(f"/data/lenam/cvst/reformulation/tc20_tk_3.pkl","rb") as f:
	tc20_tk = pickle.load(f)
with open(f"/data/lenam/cvst/reformulation/tc21_tk_3.pkl","rb") as f:
	tc21_tk = pickle.load(f)

model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.to(device)
wandb.watch(model, log="all")

train_ds = DatasetCQR(train_tk)
val_ds = DatasetCQR(val_tk)
tc20_ds = DatasetCQR(tc20_tk)
tc21_ds = DatasetCQR(tc21_tk)

train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=my_collate)
val_dl = DataLoader(val_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=my_collate)
tc20_dl = DataLoader(tc20_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=my_collate)
tc21_dl = DataLoader(tc21_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=my_collate)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

for epoch in range(1, 1+config.EPOCHS):
	train(model, device, train_dl, optimizer, tokenizer, wandb)
	val(model, device, val_dl, tokenizer, wandb, save=f"cn_test_3_{epoch}.out")
	val(model, device, tc20_dl, tokenizer, wandb, save=f"tc20_3_{epoch}.out")
	val(model, device, tc21_dl, tokenizer, wandb, save=f"tc21_3_{epoch}.out")

model.save_pretrained(save_directory=f"/data/lenam/cvst/reformulation/model/{random_seed}_3_{epoch}.pt")