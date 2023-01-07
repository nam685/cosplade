from extr2_sumti import *
from extr2_data import *
from extr2_sumti_model import *
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
import wandb
import pickle

random_seed = np.random.choice(9999)
wandb.init(project="convers1")
config = wandb.config
config.SEED = random_seed
config.EPOCHS = 1
config.QCAT_LEARNING_RATE = 2e-5
config.QAP_LEARNING_RATE = 3e-5
config.TRAIN_BATCH_SIZE = 16
config.VAL_BATCH_SIZE = 32
config.FORMAT = '_enhanced_aqla'
config.note = f'extr2_sumti'
print(random_seed,config.note)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

model = Extr2SpladeSumti()
model.to(device)
wandb.watch(model, log="all")

def stringify(d):
	dstr = dict()
	for k in d:
		dstr[str(k)] = d[k]
	return dstr

with open(f"/data/lenam/cvst/extraction2/tokens{config.FORMAT}.pkl","rb") as f:
	tokens = pickle.load(f)
#with open(f"/data/lenam/cvst/extraction2/tokens22_aqla.pkl","rb") as f:
#	tokens22 = pickle.load(f)

with open("/data/lenam/cvst/cn/ti.pkl","rb") as f:
	ti = pickle.load(f)
ti_train, ti_dev, ti_test = ti
_, tr_ti_rw, _ = ti_train
_, dv_ti_rw, _ = ti_dev
_, ts_ti_rw, _ = ti_test
with open("/data/lenam/cvst/cn/rw_tokens.pkl","rb") as f:
	tr_tks_rw, dv_tks_rw, ts_tks_rw = pickle.load(f)

with open("/data/lenam/cvst/tc/ti20.pkl","rb") as f:
	ti = pickle.load(f)
_, tc20_ti_rw, _ = ti
tc20_ti_rw = stringify(tc20_ti_rw)
with open("/data/lenam/cvst/tc/ti21.pkl","rb") as f:
	ti = pickle.load(f)
_, tc21_ti_rw, _ = ti
tc21_ti_rw = stringify(tc21_ti_rw)
#with open("/data/lenam/cvst/tc/tirw22.pkl","rb") as f:
#	tc22_ti_rw = pickle.load(f)
#tc22_ti_rw = stringify(tc22_ti_rw)

with open("/data/lenam/cvst/tc/rw_tokens.pkl","rb") as f:
	t20_tks_rw, t21_tks_rw = pickle.load(f)
#with open("/data/lenam/cvst/tc/rw22_tokens.pkl","rb") as f:
#	t22_tks_rw = pickle.load(f)
t20_tks_rw = stringify(t20_tks_rw)
t21_tks_rw = stringify(t21_tks_rw)
#t22_tks_rw = stringify(t22_tks_rw)

tr_ds = Extr2Dataset({**tokens["t20qct"],**tokens["t21qct"],**tokens["trqct"],**tokens["dvqct"],**tokens["tsqct"]},{**tokens["t20qapt"],**tokens["t21qapt"],**tokens["trqapt"],**tokens["dvqapt"],**tokens["tsqapt"]})
tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE, collate_fn=embed_smart_pad)
tc20_ds = Extr2Dataset(tokens["t20qct"],tokens["t20qapt"])
tc20_dl = DataLoader(tc20_ds, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
tc21_ds = Extr2Dataset(tokens["t21qct"],tokens["t21qapt"])
tc21_dl = DataLoader(tc21_ds, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
#tc22_ds = Extr2Dataset(tokens22["t22qct"],tokens22["t22qapt"])
#tc22_dl = DataLoader(tc22_ds, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)

optimizer = torch.optim.Adam(params=[
	{'params':model.qcat_encoder.parameters(), 'lr': config.QCAT_LEARNING_RATE},
	{'params':model.qap_encoder.parameters(), 'lr': config.QAP_LEARNING_RATE}
])

for epoch in range(1,1+config.EPOCHS):
	train(model, device, optimizer, tr_dl, {**tr_ti_rw,**dv_ti_rw,**ts_ti_rw,**tc20_ti_rw,**tc21_ti_rw}, {**tr_tks_rw,**dv_tks_rw,**ts_tks_rw,**t20_tks_rw,**t21_tks_rw}, wandb)
	save = (epoch in [1,2])
	eval(model, device, tc20_dl, tc20_ti_rw, t20_tks_rw, wandb, partition='t20', save_emb=save, name=f"{random_seed}_{epoch}")
	eval(model, device, tc21_dl, tc21_ti_rw, t21_tks_rw, wandb, partition='t21', save_emb=save, name=f"{random_seed}_{epoch}")
	#eval(model, device, tc22_dl, tc22_ti_rw, t22_tks_rw, wandb, partition='t22', save_emb=save, name=f"{random_seed}_{epoch}")

model.save_pretrained(f"/data/lenam/cvst/model/{random_seed}")

