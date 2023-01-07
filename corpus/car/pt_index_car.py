from tqdm import tqdm
import pyterrier as pt
pt.init()

def treccar_generate():
    with open("car.tsv", 'r') as corpusfile:
        for l in tqdm(corpusfile,total=29794697):
            docno, passage = l.split("\t",1)
            if len(passage) > 0:
                yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./car_index", meta=['docno'], meta_lengths=[64])
print("Indexing 29794697 CAR passages")
indexref = iter_indexer.index(treccar_generate())
