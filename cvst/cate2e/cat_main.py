import pickle
from torch import cuda
import wandb
from cat_model import *
from cat_data import *
from cat import *
import numpy as np
device = 'cuda' if cuda.is_available() else 'cpu'

random_seed = np.random.randint(9999)
wandb.init(project="convers1")
config = wandb.config
config.note = f'cat'
config.SEED = random_seed
config.EPOCHS = 8
config.LEARNING_RATE = 1e-5
config.TRAIN_BATCH_SIZE = 32
config.VAL_BATCH_SIZE = 64
config.LOSS = 'mse'
config.QUERY_ONLY_MLM=True
config.FORMAT = '_aqaa'
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

model = CatSplade(config.QUERY_ONLY_MLM)
model.to(device)
wandb.watch(model, log="all")

with open(f"/data/lenam/cvst/cate2e/cat_tokens{config.FORMAT}.pkl","rb") as f:
	cn_train,cn_dev,cn_test,tc20,tc21 = pickle.load(f)

with open("/data/lenam/cvst/cn/ti.pkl","rb") as f:
	ti = pickle.load(f)
ti_train, ti_dev, ti_test = ti
_, tr_ti_rw, _ = ti_train
_, dv_ti_rw, _ = ti_dev
_, ts_ti_rw, _ = ti_test
with open("/data/lenam/cvst/tc/ti20.pkl","rb") as f:
	ti = pickle.load(f)
_, tc20_ti_rw, _ = ti
with open("/data/lenam/cvst/tc/ti21.pkl","rb") as f:
	ti = pickle.load(f)
_, tc21_ti_rw, _ = ti

tr_ds = CatDataset({**cn_train,**cn_dev})
tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE, collate_fn=embed_smart_pad)
ts_ds = CatDataset(cn_test)
ts_dl = DataLoader(ts_ds, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
tc20_ds = CatDataset(tc20)
tc20_dl = DataLoader(tc20_ds, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)
tc21_ds = CatDataset(tc21)
tc21_dl = DataLoader(tc21_ds, batch_size=config.VAL_BATCH_SIZE, collate_fn=embed_smart_pad)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

'''
print("Let's try to overfit a small set")
mini_ds = CatDataset(cn_dev)
mini_dl = DataLoader(mini_ds, shuffle=True, batch_size=config.TRAIN_BATCH_SIZE, collate_fn=embed_smart_pad)
for epoch in range(32):
	train(model, device, optimizer, mini_dl, dv_ti_rw, config.TEMPERATURE, wandb)
'''

for epoch in range(config.EPOCHS):
	train(model, device, optimizer, tr_dl, {**tr_ti_rw,**dv_ti_rw}, wandb, config.LOSS)
	eval(model, device, ts_dl, ts_ti_rw, wandb, save_emb=False, loss_fn=config.LOSS)
	tc20_ti_pred = eval(model, device, tc20_dl, tc20_ti_rw, wandb, save_emb=True, loss_fn=config.LOSS)
	tc21_ti_pred = eval(model, device, tc21_dl, tc21_ti_rw, wandb, save_emb=True, loss_fn=config.LOSS)
	if (1+epoch) % 4 == 0:
		with open(f"/data/lenam/cvst/emb/pred_cat_q_20_{random_seed}_{1+epoch}.pkl","wb") as f:
			pickle.dump(tc20_ti_pred,f,protocol=-1)
		with open(f"/data/lenam/cvst/emb/pred_cat_q_21_{random_seed}_{1+epoch}.pkl","wb") as f:
			pickle.dump(tc21_ti_pred,f,protocol=-1)
