import pickle
from tqdm import tqdm
from rerank_data import *
import json
from os.path import exists
import sys

if len(sys.argv)!=2:
	print("python rr_cn.py <input>")
	exit(1)
input_type = sys.argv[1]

paths = {
	"keyword":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs1.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs1.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs1.pkl"
	),
	"kw_pQs":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs4.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs4.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs4.pkl"
	),
	"kw_pQs5":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs5.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs5.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs5.pkl"
	),
	"cqr_kw":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs6.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs6.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs6.pkl"
	),
	"keysentence":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs7.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs7.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs7.pkl"
	),
	"kw_pQs20":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs8.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs8.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs8.pkl"
	),
    "kw_5":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs9.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs9.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs9.pkl"
	),
    "kw_20":(
		"/data/lenam/cvst/rfml_keyword/tr_inputs10.pkl",
		"/data/lenam/cvst/rfml_keyword/dv_inputs10.pkl",
		"/data/lenam/cvst/rfml_keyword/ts_inputs10.pkl"
	),


}

with open("/data/lenam/cvst/rerank/rrdvcnkw.pkl","rb") as f:
	dvi,_ = pickle.load(f)
with open("/data/lenam/cvst/rerank/rrtscnkw.pkl","rb") as f:
	tsi,_ = pickle.load(f)
with open("/data/lenam/cvst/rerank/rrtrcnkw.pkl","rb") as f:
	tri,_ = pickle.load(f)

querytrpath, querydvpath, querytspath = paths[input_type]
with open(querytrpath,"rb") as f:
	tr_q = pickle.load(f)

with open(querydvpath,"rb") as f:
	dv_q = pickle.load(f)

with open(querytspath,"rb") as f:
	ts_q = pickle.load(f)

tr_ds = RerankDataset({"name":f"rrtrcn_{input_type}","queries":tr_q,"passages_file":"fs_tr"},pre_tokenize=False,sampled=True,instances=tri)
dv_ds = RerankDataset({"name":f"rrdvcn_{input_type}","queries":dv_q,"passages_file":"fs_dv"},pre_tokenize=False,sampled=True,instances=dvi)
ts_ds = RerankDataset({"name":f"rrtscn_{input_type}","queries":ts_q,"passages_file":"fs_ts"},pre_tokenize=False,sampled=True,instances=tsi)
