import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")

with open("/data/lenam/cvst/cn/canard.pkl","rb") as f:
	cn = pickle.load(f)
cn_train, cn_dev, cn_test = cn
_, train_rw, _ = cn_train
_, dev_rw, _ = cn_dev
_, test_rw, _ = cn_test

def process(rw):
	tokens = dict()
	for topic in tqdm(rw):
		tokens[topic] = dict()
		for turn in rw[topic]:
			tokens[topic][turn] = tokenizer(rw[topic][turn]).input_ids
	return tokens

with open("/data/lenam/cvst/cn/rw_tokens.pkl","wb") as f:
	pickle.dump((process(train_rw),process(dev_rw),process(test_rw)),f,protocol=-1)