import pandas as pd
import pickle
import re

def parse_canard(filename):
	df = pd.read_json(filename)
	question_dict = dict()
	rewrite_dict = dict()
	response_dict = dict()

	for history,topic,question,turn,rewrite in df.itertuples(index=False):
		if topic not in question_dict:
			question_dict[topic] = dict()
			rewrite_dict[topic] = dict()
		question_dict[topic][turn] = question
		rewrite_dict[topic][turn] = rewrite

		if turn == 1:
			title = history[0] + ', ' + history[1]
			words_title = set(re.split('[^a-zA-Z]',title.lower()))
			words_q1 = set(re.split('[^a-zA-Z]',rewrite.lower()))
			if len(words_title.intersection(words_q1)):
				q1 = title + ". " + rewrite
			question_dict[topic][turn] = q1
			rewrite_dict[topic][turn] = q1
	
	topic_length = {topic:len(question_dict[topic]) for topic in question_dict}
	
	i = 0
	while i < len(df):
		_,topic,_,_,_ = tuple(df.iloc[i])
		num_turns = topic_length[topic]
		history,_,_,_,_ = tuple(df.iloc[i+num_turns-1])
		response_dict[topic] = {}
		response_dict[topic][0] = history[0] + ', ' + history[1]
		for j in range(1,num_turns):
			response_dict[topic][j] = history[1 + 2*j]
		i+=num_turns
	
	return question_dict, rewrite_dict, response_dict

dev = parse_canard(f"/data/lenam/topics/raw/CANARD_Release/dev.json")
test = parse_canard(f"/data/lenam/topics/raw/CANARD_Release/test.json")
train = parse_canard(f"/data/lenam/topics/raw/CANARD_Release/train.json")
with open("canard.pkl","wb") as f:
	pickle.dump((train,dev,test),f,protocol=-1)