import json
import pickle

with open("/data/lenam/topics/raw/2021_manual_evaluation_topics_v1.0.json","r") as f:
	tc = json.load(f)

q = dict()
rw = dict()
a = dict()
for topic in tc:
	topic_number = topic['number']
	turns = topic['turn']
	q[topic_number] = dict()
	rw[topic_number] = dict()
	a[topic_number] = dict()
	for turn in turns:
		turn_number = turn['number']
		q[topic_number][turn_number] = turn['raw_utterance']
		rw[topic_number][turn_number] = turn['manual_rewritten_utterance']
		a[topic_number][turn_number] = turn['passage']

with open("treccast_2021.pkl","wb") as f:
	pickle.dump((q,rw,a),f,protocol=-1)
