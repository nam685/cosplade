import json
import sys

if len(sys.argv)!=2:
	print("python rm_pid_run <run>")
	exit(1)
run = sys.argv[1]

with open(f"/data/lenam/eval/runs/{run}.json","r") as f:
	runobj = json.load(f)

def trim_pid(ranking):
	trimmed_ranking = dict()
	for passageid,score in sorted(ranking.items(),key=lambda item: item[1], reverse=True)[:3000]:
		docid = passageid.rsplit('-',1)[0]
		if docid not in trimmed_ranking:
			trimmed_ranking[docid] = score
	return trimmed_ranking
	
runobj_trimmed = {turn:trim_pid(ranking) for turn,ranking in runobj.items()}
with open(f"/data/lenam/eval/runs/{run}_trimmed.json","w") as f:
	json.dump(runobj_trimmed,f)
