import json
import sys

if len(sys.argv)!=4:
    print("python eval <run1> <run2> <out>")
    exit(1)
run1 = sys.argv[1]
run2 = sys.argv[2]
out = sys.argv[3]

with open(f"/data/lenam/eval/runs/{run1}.json", "r") as runf1:
    with open(f"/data/lenam/eval/runs/{run2}.json", "r") as runf2:
        run_obj1 = json.load(runf1)
        run_obj2 = json.load(runf2)
        res = {}
        count_error = 0
        for qid in set().union(run_obj1, run_obj2):
            l1 = run_obj1[qid]
            l2 = run_obj2[qid]
            l = {**l1,**l2}
            res[qid] = dict()
            for k,v in sorted(l.items(),key=lambda item: item[1], reverse=True)[:3000]:
                if k not in res[qid]:
                    res[qid][k] = v
                if len(res[qid]) >= 1000:
                    break
            print(len(res[qid]))
            if len(res[qid]) == 0:
                print(qid)
                count_error += 1
        print(f"There are {count_error} empty vectors")
with open(f"/data/lenam/eval/runs/{out}.json", "w") as f:
    json.dump(res,f)
