import json

with open("raw/2020_manual_evaluation_topics_v1.0.json","r") as tc20:
	with open("tsv/tc20_target.tsv","w") as ft:
		with open("tsv/tc20_raw.tsv","w") as fr:
			obj = json.load(tc20)
			for topic in obj:
				topic_id = topic['number']
				for turn in topic['turn']:
					turn_id = str(topic_id) + "_" + str(turn['number'])
					ft.write(f"{turn_id}\t{turn['manual_rewritten_utterance']}\n")
					fr.write(f"{turn_id}\t{turn['raw_utterance']}\n")

with open("raw/2021_manual_evaluation_topics_v1.0.json","r") as tc21:
	with open("tsv/tc21_target.tsv","w") as ft:
		with open("tsv/tc21_raw.tsv","w") as fr:
			obj = json.load(tc21)
			for topic in obj:
				topic_id = topic['number']
				for turn in topic['turn']:
					turn_id = str(topic_id) + "_" + str(turn['number'])
					ft.write(f"{turn_id}\t{turn['manual_rewritten_utterance']}\n")
					fr.write(f"{turn_id}\t{turn['raw_utterance']}\n")

with open("raw/CANARD_Release/dev.json","r") as cn_dev:
	with open("tsv/cn_dev_target.tsv","w") as ft:
		obj = json.load(cn_dev)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			query = turn['History'][0] + " " + turn['History'][1] + " - " + turn['Rewrite']
			ft.write(f"{id}\t{query}\n")

with open("raw/CANARD_Release/test.json","r") as cn_test:
	with open("tsv/cn_test_target.tsv","w") as ft:
		obj = json.load(cn_test)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			query = turn['History'][0] + " " + turn['History'][1] + " - " + turn['Rewrite']
			ft.write(f"{id}\t{query}\n")

with open("raw/CANARD_Release/train.json","r") as cn_train:
	with open("tsv/cn_train_target.tsv","w") as ft:
		obj = json.load(cn_train)
		for turn in obj:
			id = turn['QuAC_dialog_id'] + "_" + str(turn['Question_no'])
			query = turn['History'][0] + " " + turn['History'][1] + " - " + turn['Rewrite']
			ft.write(f"{id}\t{query}\n")

