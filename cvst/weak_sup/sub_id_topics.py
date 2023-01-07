with open("/data/lenam/topics/tsv/cn_dev_target.tsv",'r') as infile:
	with open("cn_dev_target.tsv",'w') as outfile:
		with open("cn_dev_ids",'w') as idfile:
			for i,line in enumerate(infile):
				tid,text=line.strip().split('\t',1)
				outfile.write(f"{i}\t{text}\n")
				idfile.write(f"{tid}\n")
with open("/data/lenam/topics/tsv/cn_test_target.tsv",'r') as infile:
	with open("cn_test_target.tsv",'w') as outfile:
		with open("cn_test_ids",'w') as idfile:
			for i,line in enumerate(infile):
				tid,text=line.strip().split('\t',1)
				outfile.write(f"{i}\t{text}\n")
				idfile.write(f"{tid}\n")
with open("/data/lenam/topics/tsv/cn_train_target.tsv",'r') as infile:
	with open("cn_train_target.tsv",'w') as outfile:
		with open("cn_train_ids",'w') as idfile:
			for i,line in enumerate(infile):
				tid,text=line.strip().split('\t',1)
				outfile.write(f"{i}\t{text}\n")
				idfile.write(f"{tid}\n")
