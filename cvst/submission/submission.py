import sys
import json
import re
if len(sys.argv)!= 3:
    print("python submission.py <name> <runname>")
    exit(1)
name = sys.argv[1]
runname=sys.argv[2]
submission = {
    "run_name" : f"{runname}",
    "run_type" :  "automatic"
}

with open(f"/data/lenam/eval/runs/{name}.json",'r') as sf:
    scores = json.load(sf)

turns_dict = dict()
with open(f"/data/lenam/cvst/rerank/ss_{name}.txt",'r') as tf:
    for i,line in enumerate(tf):
        rank = 1+i%1000
        qid, docid, text = line.strip().split('\t',2)
        _docid = re.sub('MARCO_msmarco_doc','MARCO',docid)
        if qid not in turns_dict:
            turns_dict[qid] = []
        turns_dict[qid] += [
            {
                "text": text,
                "rank": rank,
                "provenance": [
                    {
                        "id": _docid,
                        "text": text,
                        "score": scores[qid][docid]
                    }
                ]
            }
        ]

submission["turns"] = [{"turn_id":qid, "responses":turns_dict[qid]} for qid in turns_dict]

with open(f"/data/lenam/cvst/submission/{name}.json",'w') as f:
    json.dump(submission,f)
