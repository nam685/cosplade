import pyterrier as pt
pt.init()

def msp_generate():
    with open("msp.tsv", 'r') as corpusfile:
        for l in corpusfile:
            docno, passage = l.split("\t",1)
            yield {'docno' : docno, 'text' : passage}

iter_indexer = pt.IterDictIndexer("./msp_index", meta=['docno', 'text'], meta_lengths=[10, 4096])
indexref = iter_indexer.index(msp_generate())