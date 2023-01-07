import pickle
import sys

if len(sys.argv) != 2:
	print("python offset_tool.py <name>")
	exit()
name = sys.argv[1]

def _blocks(files, size=524288):
	while True:
		b = files.read(size)
		if not b: break
		yield b

def len_file(path):
	with open(path, "r", encoding='utf8') as f:
		return sum(bl.count("\n") for bl in _blocks(f))

def save_offset(name):
	offset_dict = {}
	with open(f"{name}/{name}.tsv", 'r',encoding='utf-8') as f:
		offset = f.tell()
		while True:
			line = f.readline()
			if len(line)==0:
				break
			docid = line.split('\t',1)[0]
			offset_dict[docid] = offset
			offset = f.tell()
	with open(f"{name}/{name}.offset","wb") as f:
		pickle.dump(offset_dict,f)
	return offset_dict

save_offset(name)
