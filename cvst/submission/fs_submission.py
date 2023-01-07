import sys
import json

submission = {
	"run_name" : "MLIA_DAC_splade",
	"run_type" :  "automatic"
}

with open(f"/data/lenam/eval/runs/6529_22.json",'r') as sf:
	scores = json.load(sf)

turns_dict = dict()
with open(f"/data/lenam/cvst/rerank/ss_6529_22.txt",'r') as tf:
	for i,line in enumerate(tf):
		rank = 1+i%1000
		qid, docid, text = line.strip().split('\t',2)
		if qid not in turns_dict:
			turns_dict[qid] = []
		turns_dict[qid] += [
			{
				"text": text,
				"rank": rank,
				"provenance": [
					{
						"id": docid,
						"text": text,
						"score": scores[qid][docid]
					}
				]
			}
		]

submission["turns"] = [{"turn_id":qid, "responses":turns_dict[qid]} for qid in turns_dict]

with open(f"/data/lenam/cvst/submission/fs_t22.json",'w') as f:
	json.dump(submission,f)
