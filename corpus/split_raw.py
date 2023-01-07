import sys

if len(sys.argv)!=2:
	print("python split_raw <name>")
	exit(1)
name = sys.argv[1]

size = 2048000
with open(f"{name}/raw.tsv",'r') as source:
	for line in source:
		i, passasge = line.rstrip().split('\t',1)
		if int(i)%size==0:
			try:
				dest.close()
			except:
				None
			dest = open(f"{name}/raw/{name}_{i}",'w')
		dest.write(line)