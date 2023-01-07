import pyterrier as pt
pt.init()

def orqa_generate():
    with open("orqa.tsv", 'r') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t",1)
            yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./orqa_index", meta=['docno'], meta_lengths=[64])
indexref = iter_indexer.index(orqa_generate())