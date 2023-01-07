import json
import pytrec_eval
import sys
import numpy as np
if len(sys.argv)!=3:
    print("python eval <run> <qrel>")
    exit(1)
runstr = sys.argv[1]
qrelstr = sys.argv[2]

def threshold(qrel,threshold=2):
    new_qrel = dict()
    for turn in qrel:
        new_qrel[turn] = dict()
        for docid in qrel[turn]:
            if qrel[turn][docid] >= 2:
                new_qrel[turn][docid] = 1
            else:
                new_qrel[turn][docid] = 0
    return new_qrel

def cut(run,cutoff):
    new_run = dict()
    for turn in run:
        new_run[turn] = {docid:score for docid,score in list(run[turn].items())[:cutoff]}
    return new_run

def ndcgcut(run_obj, qrel, k):
    res = {}
    for qid in run_obj:
        if qid not in qrel:
            continue
        s = [(qrel[qid][i] if i in qrel[qid] else 0) for i,v in list(run_obj[qid].items())[:k]]
        m = sum([v>0 for v in qrel[qid].values()])
        if m == 0:
            res[qid] = 0
            continue
        dcg = 0
        idcg = 0
        for i in range(k):
            dcg += s[i] / np.log2(i+2)
        for i in range(min(k,m)):
            idcg += 1 / np.log2(i+2)
        res[qid] = dcg/idcg
    agg = 0
    for qid in res:
        agg += res[qid]
    agg /= len(res)
    var = 0
    for qid in res:
        var += (res[qid] - agg)**2
    std = var**0.5 / len(res)
    return res, agg, std

with open(f"/data/lenam/eval/runs/{runstr}.json", "r") as runf:
    with open(f"/data/lenam/qrel/{qrelstr}.json","r") as qrelf:
        run_obj = json.load(runf)
        if qrelstr == "qr21":
            run_obj = cut(run_obj, 500)

        qrel = json.load(qrelf)
        qrel = threshold(qrel,2)

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall', 'map_cut', 'recip_rank', 'ndcg_cut'})
        res = evaluator.evaluate(run_obj)

        for x in res:
            metrics = res[x].keys()
            break
        agg = {}
        for metric in metrics:
            agg[metric] = 0
            for x in res:
                agg[metric] += res[x][metric]
            agg[metric] /= len(res)
        std = {}
        for metric in metrics:
            var = 0
            count = 0
            for x in res:
                count += 1
                var += (res[x][metric] - agg[metric])**2
            std[metric] = var**0.5 / count

        manualndcg3, agg3, std3 = ndcgcut(run_obj, qrel, 3)
        manualndcg5, agg5, std5 = ndcgcut(run_obj, qrel, 5)

        trec_agg = {}
        if qrelstr == "qr21":
            trec_agg["recip_rank"] = agg["recip_rank"]
            trec_agg["recip_rank_std"] = std["recip_rank"]
            trec_agg["recall_500"] = agg["recall_500"]
            trec_agg["recall_500_std"] = std["recall_500"]
            trec_agg["ndcg_cut_5"] = agg["ndcg_cut_5"]
            trec_agg["ndcg_cut_5_std"] = std["ndcg_cut_5"]
            trec_agg["ndcg_cut_3_manual"] = agg3
            trec_agg["ndcg_cut_3_std_manual"] = std3
            trec_agg["ndcg_cut_5_manual"] = agg5
            trec_agg["ndcg_cut_5_std_manual"] = std5
            trec_agg["ndcg_cut_500"] = agg["ndcg_cut_500"]
            trec_agg["ndcg_cut500_std"] = std["ndcg_cut_500"]
            trec_agg["map_cut_500"] = agg["map_cut_500"]
            trec_agg["map_cut_500_std"] = std["map_cut_500"]
        else:
            trec_agg["recip_rank"] = agg["recip_rank"]
            trec_agg["recip_rank_std"] = std["recip_rank"]
            trec_agg["recall_1000"] = agg["recall_1000"]
            trec_agg["recall_1000_std"] = std["recall_1000"]
            trec_agg["ndcg_cut_5"] = agg["ndcg_cut_5"]
            trec_agg["ndcg_cut_5_std"] = std["ndcg_cut_5"]
            trec_agg["ndcg_cut_3_manual"] = agg3
            trec_agg["ndcg_cut_3_std_manual"] = std3
            trec_agg["ndcg_cut_5_manual"] = agg5
            trec_agg["ndcg_cut_5_std_manual"] = std5
            trec_agg["ndcg_cut_1000"] = agg["ndcg_cut_1000"]
            trec_agg["ndcg_cut1000_std"] = std["ndcg_cut_1000"]
            trec_agg["map_cut_1000"] = agg["map_cut_1000"]
            trec_agg["map_cut_1000_std"] = std["map_cut_1000"]

        with open(f"/data/lenam/eval/res/res_{runstr}.json","w") as out:
            json.dump({"all":res,"agg":agg,"trec_agg":trec_agg}, out)
