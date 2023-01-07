from tqdm import tqdm
import pyterrier as pt
pt.init()

def kilt_generate():
    with open("kilt.tsv", 'r') as corpusfile:
        for l in tqdm(corpusfile,total=17108378):
            docno, passage = l.split("\t",1)
            if len(passage) > 0:
                yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./kilt_index", meta=['docno'], meta_lengths=[20])
print("Indexing 17108378 KILT passages")
indexref = iter_indexer.index(kilt_generate())
