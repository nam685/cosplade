# cosplade
CoSPLADE: Contextualizing SPLADE for Conversational Information Retrieval

Source code for my 3-months research internship work at ISIR.
It contains experiments that work, and also those that don't.



folders content (I only describe folders containing code for CoSPLADE. Other folders contain code for unsuccessful experiements):

/corpus: contains python and bash scripts for corpus preprocessing, and indexing for splade. An example (MSMarco Document v2 corpus) is included.

/topics: python scripts to preprocess questions in topics

/cvst: CoSPLADE
- /cvst/cn: preprocessing for CANARD queries
- /cvst/tc: preprocessing for TREC CAsT queries
- /cvst/extraction2: training first stage ranker (python source code with filenames containing "sumti")
- /cvst/emb: saved query embeddings of first stage
- /cvst/visualie_ti_vectors.py: interpret embeddings
- /cvst/convers_retrieve.sh: script to run first stage ranking
- /cvst/rfml_keyword: extract keywords from first stage ranker for reranker
- /cvst/rerank: training reranker
- /cvst/submission: generate submission files for TREC CAsT

/eval: python scripts to postprocess and evaluate retrieved results
- /eval/runs: ranked list of documents for each query
- /eval/res: evaluation metrics

/qrel: qrel files

If you use this code, please cite:

@inproceedings{DBLP:conf/ecir/HaiGFNPS23,

author = {Nam Le Hai and
Thomas Gerald and
Thibault Formal and
Jian{-}Yun Nie and
Benjamin Piwowarski and
Laure Soulier},

editor = {Jaap Kamps and
Lorraine Goeuriot and
Fabio Crestani and
Maria Maistro and
Hideo Joho and
Brian Davis and
Cathal Gurrin and
Udo Kruschwitz and
Annalina Caputo},

title = {CoSPLADE: Contextualizing {SPLADE} for Conversational Information Retrieval},

booktitle = {Advances in Information Retrieval - 45th European Conference on Information
Retrieval, {ECIR} 2023, Dublin, Ireland, April 2-6, 2023, Proceedings, Part {I}},
series = {Lecture Notes in Computer Science},

volume = {13980},

pages = {537--552},

publisher = {Springer},

year = {2023},

doi = {10.1007/978-3-031-28244-7_34},

biburl = {https://dblp.org/rec/conf/ecir/HaiGFNPS23.bib},

}

