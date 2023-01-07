from tqdm import tqdm
import pyterrier as pt
pt.init()

def msd1_generate():
    with open("msdoc1.tsv", 'r') as corpusfile:
        for l in tqdm(corpusfile,total=22526377):
            docno, passage = l.split("\t",1)
            yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./msd1_index", meta=['docno'], meta_lengths=[20])
print("Indexing 22526377 MSMarco Doc 1 passages")
indexref = iter_indexer.index(msd1_generate())
