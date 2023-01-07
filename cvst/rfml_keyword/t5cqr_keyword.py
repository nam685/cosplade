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
	def __init__(self, source_dict, target_dict):
		self.instances = []
		self.tkdict = dict()
		for topic in tqdm(source_dict):
			self.tkdict[topic] = dict()
			for turn in source_dict[topic]:
				self.tkdict[topic][turn] = (tokenizer(source_dict[topic][turn],truncation=True),tokenizer(target_dict[topic][turn]))
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
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True
			)
			generated_texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
			for topic, turn, text in zip(topics, turns, generated_texts):
				if topic not in predictions:
					predictions[topic] = dict()
				predictions[topic][turn] = text

	with open(f"/data/lenam/cvst/rfml_keyword/generated/{save}","wb") as f:
		pickle.dump(predictions,f,protocol=-1)

random_seed = np.random.randint(9999)
wandb.init(project="convers1")
config = wandb.config
config.note = f't5cqrkeyword_la'
config.SEED = random_seed
config.EPOCHS = 2
config.LEARNING_RATE = 1e-4
config.TRAIN_BATCH_SIZE = 16
config.VAL_BATCH_SIZE = 32
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

def stringify(d):
	dstr = dict()
	for k in d:
		dstr[str(k)] = d[k]
	return dstr
with open("/data/lenam/cvst/cn/canard.pkl","rb") as f:
	cn = pickle.load(f)
cn_train, cn_dev, cn_test = cn
_, train_rw, _ = cn_train
_, dev_rw, _ = cn_dev
_, test_rw, _ = cn_test
with open("/data/lenam/cvst/tc/treccast_2020.pkl","rb") as f:
	tc20 = pickle.load(f)
_, rw20, _ = tc20
rw20 = stringify(rw20)
with open("/data/lenam/cvst/tc/treccast_2021.pkl","rb") as f:
	tc21 = pickle.load(f)
_, rw21, _ = tc21
rw21 = stringify(rw21)

with open(f"/data/lenam/cvst/rfml_keyword/tr_inputs3.pkl","rb") as f:
	tr_inputs = pickle.load(f)
with open(f"/data/lenam/cvst/rfml_keyword/dv_inputs3.pkl","rb") as f:
	dv_inputs = pickle.load(f)
with open(f"/data/lenam/cvst/rfml_keyword/ts_inputs3.pkl","rb") as f:
	ts_inputs = pickle.load(f)
with open(f"/data/lenam/cvst/rfml_keyword/t20_inputs3.pkl","rb") as f:
	t20_inputs = pickle.load(f)
with open(f"/data/lenam/cvst/rfml_keyword/t21_inputs3.pkl","rb") as f:
	t21_inputs = pickle.load(f)

model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.to(device)
wandb.watch(model, log="all")

train_ds = DatasetCQR({**tr_inputs,**dv_inputs},{**train_rw,**dev_rw})
ts_ds = DatasetCQR(ts_inputs, test_rw)
tc20_ds = DatasetCQR(t20_inputs, rw20)
tc21_ds = DatasetCQR(t21_inputs, rw21)

train_dl = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, collate_fn=my_collate)
ts_dl = DataLoader(ts_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=my_collate)
tc20_dl = DataLoader(tc20_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=my_collate)
tc21_dl = DataLoader(tc21_ds, batch_size=config.VAL_BATCH_SIZE, shuffle=False, collate_fn=my_collate)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

for epoch in range(1, 1+config.EPOCHS):
	train(model, device, train_dl, optimizer, tokenizer, wandb)
	val(model, device, train_dl, tokenizer, wandb, save=f"tr_{epoch}.out")
	val(model, device, ts_dl, tokenizer, wandb, save=f"ts_{epoch}.out")
	val(model, device, tc20_dl, tokenizer, wandb, save=f"t20_{epoch}.out")
	val(model, device, tc21_dl, tokenizer, wandb, save=f"t21_{epoch}.out")

model.save_pretrained(save_directory=f"/data/lenam/cvst/rfml_keyword/model/{random_seed}_{epoch}.pt")