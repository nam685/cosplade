import json
import sys

if len(sys.argv)!=4:
    print("python eval <adhoc|conversation> <run> <corpus>")
    exit(1)
typ = sys.argv[1]
run = sys.argv[2]
corpus = sys.argv[3]

id_dict = {}
with open(f"/data/lenam/corpus/{corpus}/{corpus}_ids","r") as idf:
	for i,l in enumerate(idf):
		id_dict[i] = l.strip()

with open(f"/data/lenam/retrieval/{typ}/{run}.out/other_dataset/run.json", "r") as runf:	
	run_obj = json.load(runf)
	out_obj = {}
	for qid in run_obj.keys():
		if qid[0] == '1':
			_qid = qid[:3]+'_'+qid[3:]
		else:
			_qid = qid[:2]+'_'+qid[2:]
		out_obj[_qid] = {}
		for docid in run_obj[qid]:
			out_obj[_qid][id_dict[int(docid)].split('-')[0]] = run_obj[qid][docid]

	with open(f"/data/lenam/eval/runs/{run}.json","w") as outf:
		json.dump(out_obj,outf)
