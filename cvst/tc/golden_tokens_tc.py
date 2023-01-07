import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")

with open("/data/lenam/cvst/tc/treccast_2020.pkl","rb") as f:
	t20 = pickle.load(f)
_, rw20, _ = t20
with open("/data/lenam/cvst/tc/treccast_2021.pkl","rb") as f:
	t21 = pickle.load(f)
_, rw21, _ = t21

def process(rw):
	tokens = dict()
	for topic in tqdm(rw):
		tokens[topic] = dict()
		for turn in rw[topic]:
			tokens[topic][turn] = tokenizer(rw[topic][turn]).input_ids
	return tokens

with open("/data/lenam/cvst/tc/rw_tokens.pkl","wb") as f:
	pickle.dump((process(rw20),process(rw21)),f,protocol=-1)