import pyterrier as pt
pt.init()
from tqdm import tqdm

def msd2_generate():
    with open("msdoc2.tsv", 'r') as corpusfile:
        for l in tqdm(corpusfile,total=97112074):
            try:
                docno, passage = l.split("\t",1)
                if len(passage) > 0:
                    yield {'docno' : docno, 'text' : passage}
            except:
                continue

iter_indexer = pt.IterDictIndexer("./msd2_index", meta=['docno'], meta_lengths=[40])
print("Indexing 97112074 MSMarco Doc 2 passages")
indexref = iter_indexer.index(msd2_generate())
