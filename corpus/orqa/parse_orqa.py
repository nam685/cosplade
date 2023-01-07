import json

with open("all_blocks.txt","r") as f:
	with open("orqa.tsv","w") as out:
		for line in f:
			obj = json.loads(line)
			out.write(f"{obj['id']}\t{obj['title']}. {obj['text']}\n")