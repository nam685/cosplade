import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
import json

with open("/data/lenam/topics/raw/2022_evaluation_topics_tree_v1.0.json","r") as tc22:
	with open("/data/lenam/topics/raw/2022_evaluation_topics_turn_ids.json","r") as ftids:
		_obj = json.load(tc22)
		obj = dict()
		for _topic in _obj:
			obj[_topic['number']] = dict()
			for _turn in _topic['turn']:
				obj[_topic['number']][_turn['number']] = {k:v for k,v in _turn.items() if k != 'number'}
		tids = json.load(ftids)
		rw = dict()

		for topic in tids:
			rw[int(topic)] = dict()
			for turn in tids[topic]:
				rw[int(topic)][turn] = obj[int(topic)][turn]['manual_rewritten_utterance']

def process(rw):
	tokens = dict()
	for topic in tqdm(rw):
		tokens[topic] = dict()
		for turn in rw[topic]:
			tokens[topic][turn] = tokenizer(rw[topic][turn]).input_ids
	return tokens

with open("/data/lenam/cvst/tc/rw22_tokens.pkl","wb") as f:
	pickle.dump(process(rw),f,protocol=-1)