import json


with open("raw/2020_manual_evaluation_topics_v1.0.json","r") as tc20:
		with open("cat/tc20.tsv","w") as fr:
			obj = json.load(tc20)
			for topic in obj:
				topic_id = topic['number']
				history_raw=[]
				for turn in topic['turn']:
					turn_id = str(topic_id) + "_" + str(turn['number'])
					fr.write(f"{turn_id}\t{' [SEP] '.join(history_raw)}\t{turn['raw_utterance']}\n")
					history_raw.append(turn['raw_utterance'])

with open("raw/2021_manual_evaluation_topics_v1.0.json","r") as tc21:
		with open("cat/tc21.tsv","w") as fr:
			obj = json.load(tc21)
			for topic in obj:
				topic_id = topic['number']
				history_raw=[]
				for turn in topic['turn']:
					turn_id = str(topic_id) + "_" + str(turn['number'])
					fr.write(f"{turn_id}\t{' [SEP] '.join(history_raw)}\t{turn['raw_utterance']}\n")
					history_raw.append(turn['raw_utterance'])

with open("raw/CANARD_Release/dev.json","r") as cn_dev:
	with open("cat/cn_dev.tsv","w") as ft:
		obj = json.load(cn_dev)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			text = " [SEP] ".join(turn['History'][:2] + turn['History'][2::2]) + "\t" + turn['Question']
			ft.write(f"{id}\t{text}\n")

with open("raw/CANARD_Release/test.json","r") as cn_test:
	with open("cat/cn_test.tsv","w") as ft:
		obj = json.load(cn_test)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			text = " [SEP] ".join(turn['History'][:2] + turn['History'][2::2]) + "\t" + turn['Question']
			ft.write(f"{id}\t{text}\n")

with open("raw/CANARD_Release/train.json","r") as cn_train:
	with open("cat/cn_train.tsv","w") as ft:
		obj = json.load(cn_train)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			text = " [SEP] ".join(turn['History'][:2] + turn['History'][2::2]) + "\t" + turn['Question']
			ft.write(f"{id}\t{text}\n")
