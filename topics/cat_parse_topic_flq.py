import json


with open("/data/lenam/topics/raw/2020_manual_evaluation_topics_v1.0.json","r") as tc20:
		with open("cat/tc20_flq.tsv","w") as fr:
			obj = json.load(tc20)
			for topic in obj:
				topic_id = topic['number']
				for turn in topic['turn']:
					turn_id = str(topic_id) + "_" + str(turn['number'])
					if int(turn['number']) == 1:
						history_raw = []
						first_query = turn['raw_utterance']
					elif int(turn['number']) == 2:
						history_raw = [last_query]
					else:
						history_raw = [first_query, last_query]
					fr.write(f"{turn_id}\t{' [SEP] '.join(history_raw)}\t{turn['raw_utterance']}\n")
					last_query = turn['raw_utterance']

with open("/data/lenam/topics/raw/2021_manual_evaluation_topics_v1.0.json","r") as tc21:
		with open("cat/tc21_flq.tsv","w") as fr:
			obj = json.load(tc21)
			for topic in obj:
				topic_id = topic['number']
				history_raw=[]
				for turn in topic['turn']:
					turn_id = str(topic_id) + "_" + str(turn['number'])
					if int(turn['number']) == 1:
						history_raw = []
						first_query = turn['raw_utterance']
					elif int(turn['number']) == 2:
						history_raw = [last_query]
					else:
						history_raw = [first_query, last_query]
					fr.write(f"{turn_id}\t{' [SEP] '.join(history_raw)}\t{turn['raw_utterance']}\n")
					last_query = turn['raw_utterance']

with open("/data/lenam/topics/raw/CANARD_Release/dev.json","r") as cn_dev:
	with open("cat/cn_dev_flq.tsv","w") as ft:
		obj = json.load(cn_dev)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			history = turn['History'][2::2]
			history = [history[0], history[-1]] if len(history) > 1 else history
			text = " [SEP] ".join(turn['History'][:2] + history) + "\t" + turn['Question']
			ft.write(f"{id}\t{text}\n")

with open("/data/lenam/topics/raw/CANARD_Release/test.json","r") as cn_test:
	with open("cat/cn_test_flq.tsv","w") as ft:
		obj = json.load(cn_test)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			history = turn['History'][2::2]
			history = [history[0], history[-1]] if len(history) > 1 else history
			text = " [SEP] ".join(turn['History'][:2] + history) + "\t" + turn['Question']
			ft.write(f"{id}\t{text}\n")

with open("/data/lenam/topics/raw/CANARD_Release/train.json","r") as cn_train:
	with open("cat/cn_train_flq.tsv","w") as ft:
		obj = json.load(cn_train)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			history = turn['History'][2::2]
			history = [history[0], history[-1]] if len(history) > 1 else history
			text = " [SEP] ".join(turn['History'][:2] + history) + "\t" + turn['Question']
			ft.write(f"{id}\t{text}\n")
