import re
with open("msdoc1.tsv.trecweb",'r') as f:
	with open("msdoc1.tsv","w") as out:
		passage_next=False
		for line in f:
			if line[:7]=="<DOCNO>":
				docid=line[7:-9]
			elif line[:7]=="<TITLE>":
				title=line[7:-9]
			elif line[:12]=="<passage id=":
				pid = int(line[12:-2])
				passage_next=True
			elif passage_next:
				passage = line.rstrip()
				out.write(f"{docid}-{pid}\t{title}. {passage}\n")
				passage_next=False