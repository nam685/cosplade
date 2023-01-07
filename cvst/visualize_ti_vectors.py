import sys
import pickle
import numpy as np
from transformers import AutoTokenizer

if len(sys.argv) != 2:
    print("python visualize_ti_vectors.py <name>")
    exit(1)
name = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

with open(f"/data/lenam/cvst/emb/{name}.pkl","rb") as f:
    emb_dict = pickle.load(f)

print(emb_dict.keys())

count_turns = 0
count_nonzeros = 0
for topic in emb_dict:
    for turn in emb_dict[topic]:
        ti = emb_dict[topic][turn].squeeze()
        emb_dict[topic][turn] = ti
        count_turns += 1
        nonzeros = np.nonzero(np.array(ti))[0]
        if len(nonzeros) == 0:
            print(f"Empty vector at topic {topic} turn {turn}")
        if len(nonzeros) > 30000:
            print(f"Suspicious vector at topic {topic} turn {turn}")
        count_nonzeros += len(nonzeros)

def interpret_convo(ti,reverse_voc):
    col = np.nonzero(np.array(ti))[0]
    print("number of actual dimensions: ", len(col))
    weights = ti[col].tolist()
    d = {k: v for k, v in zip(col, weights)}
    sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
    bow_rep = []
    for k, v in sorted_d.items():
        bow_rep.append((reverse_voc[k], round(v, 2)))
    print("SPLADE BOW rep:\n", bow_rep)

print(f"L0_d: {count_nonzeros/count_turns}")

examine = input("examine individual turn? [y/n]")
while examine == 'y':
    topic = input("topic")
    topic = topic if topic in emb_dict else int(topic)
    turn = int(input("turn"))

    interpret_convo(emb_dict[topic][turn].squeeze(),reverse_voc)
    examine = input("examine individual turn? [y/n]")
