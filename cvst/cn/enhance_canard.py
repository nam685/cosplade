import pickle
from tqdm import tqdm
with open("canard.pkl","rb") as f:
	cn = pickle.load(f)
cn_train, cn_dev, cn_test = cn
train_q, train_rw, train_rp = cn_train
dev_q, dev_rw, dev_rp = cn_dev
test_q, test_rw, test_rp = cn_test

with open("/data/lenam/cvst/weak_sup/cn_dev_orqa.tsv","r") as f:
	for line in f:
		qid, score, short_answer, long_answer = line.strip().split('\t',3)
		topic, turn = qid.rsplit('_',1)
		dev_rp[topic][int(turn)] = long_answer
cn_dev = dev_q, dev_rw, dev_rp
with open("/data/lenam/cvst/weak_sup/cn_test_orqa.tsv","r") as f:
	for line in f:
		qid, score, short_answer, long_answer = line.strip().split('\t',3)
		topic, turn = qid.rsplit('_',1)
		test_rp[topic][int(turn)] = long_answer
cn_test = test_q, test_rw, test_rp
with open("/data/lenam/cvst/weak_sup/cn_train_orqa.tsv","r") as f:
	for line in f:
		qid, score, short_answer, long_answer = line.strip().split('\t',3)
		topic, turn = qid.rsplit('_',1)
		train_rp[topic][int(turn)] = long_answer
cn_train = train_q, train_rw, train_rp
with open("canard_enhanced.pkl","wb") as f:
	pickle.dump((cn_train,cn_dev,cn_test),f,protocol=-1)
