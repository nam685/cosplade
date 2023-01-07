import json

d = {}
with open("2020qrels.txt","r") as qr20:
	for line in qr20:
		qid, _, docid, score = line.split()
		if qid not in d:
			d[qid] = {}
		d[qid][docid] = int(score)
	json.dump(d,open("qr20.json","w"))


d = {}
with open("trec-cast-qrels-docs.2021.qrel","r") as qr21:
	for line in qr21:
		qid, _, docid, score = line.split()
		if qid not in d:
			d[qid] = {}
		d[qid][docid] = int(score)
	json.dump(d,open("qr21.json","w"))