import json
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
from tqdm import tqdm
import pickle

# All previous queries, last previous answer

with open("raw/2022_automatic_evaluation_topics_tree_v1.0.json","r") as tc22:
	_obj = json.load(tc22)
	obj = dict()
	for _topic in _obj:
		obj[_topic['number']] = dict()
		for _turn in _topic['turn']:
			obj[_topic['number']][_turn['number']] = {k:v for k,v in _turn.items() if k != 'number'}
	prev_queries = dict()
	qcat = dict()
	qcat_tokens = dict()
	qa_pairs = dict()
	qa_pairs_tokens = dict()
	for topic in tqdm(obj):
		prev_queries[topic] = dict()
		qcat[topic] = dict()
		qcat_tokens[topic] = dict()
		qa_pairs[topic] = dict()
		qa_pairs_tokens[topic] = dict()
		for turn in obj[topic]:
			if obj[topic][turn]['participant'] == 'User':
				query = obj[topic][turn]['utterance']
				if 'parent' in obj[topic][turn]:
					prev_answer = obj[topic][obj[topic][turn]['parent']]['response']
					prev_query_turn = obj[topic][obj[topic][turn]['parent']]['parent']
					prev_query = obj[topic][prev_query_turn]['utterance']
					prev_queries[topic][turn] = prev_queries[topic][prev_query_turn] + [prev_query]
					qa_pairs[topic][turn] = {0:(query, prev_answer)}
					qa_pairs_tokens[topic][turn] = {0:tokenizer(query, prev_answer)}
				else:
					prev_queries[topic][turn] = []
					qa_pairs[topic][turn] = dict()
					qa_pairs_tokens[topic][turn] = dict()
				qcat[topic][turn] = query + '\t' + " [SEP] ".join(prev_queries[topic][turn])
				qcat_tokens[topic][turn] = tokenizer(query," [SEP] ".join(prev_queries[topic][turn]))
				

tokens = {
	"t22qc":qcat,"t22qct":qcat_tokens,"t22qap":qa_pairs,"t22qapt":qa_pairs_tokens
}
print(tokens)

with open("/data/lenam/cvst/extraction2/tokens22_aqla.pkl","wb") as f:
	pickle.dump(tokens,f,protocol=-1)