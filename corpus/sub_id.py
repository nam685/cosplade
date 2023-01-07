from tqdm import tqdm
import sys
if len(sys.argv)!=4:
    print("python sub_id <source_tsv> <dest_tsv> <id_file>")
    exit(1)
source_tsv = sys.argv[1]
dest_tsv = sys.argv[2]
id_file = sys.argv[3]
i = 0
with open(source_tsv,'r') as source:
    with open(dest_tsv,'w') as dest:
        with open(id_file,'w') as idf:
            for line in tqdm(source):
                try:
                    docid, passage = line.rstrip().split('\t',1)
                    dest.write(f"{i}\t{passage}\n")
                    idf.write(f"{docid}\n")
                    i+=1
                    prevline=line
                except:
                    print('===============')
                    print(prevline)
                    print(line)
