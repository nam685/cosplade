from tqdm import tqdm
import pyterrier as pt
pt.init()

def wapo_generate():
    with open("wapo.tsv", 'r') as corpusfile:
        for l in tqdm(corpusfile,total=2963130):
            docno, passage = l.split("\t",1)
            if len(passage) > 0:
                yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./wapo_index", meta=['docno'], meta_lengths=[48])
print("Indexing 2963130 WAPO passages")
indexref = iter_indexer.index(wapo_generate())
