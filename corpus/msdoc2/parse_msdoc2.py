import re
import jsonlines
from tqdm import tqdm
for i in range(110):
    with open(f"tsv/MARCO_v2_{i}.tsv","w") as out:
        with jsonlines.open(f"/data/lenam/corpus/jsonlines/MARCO_v2_{i}.jsonl") as reader:
            for article in tqdm(reader):
                aid = article['id']
                title = re.sub('\n','',re.sub(r'[^\x20-\x7f]',r'',article['title']))
                for passage in article['contents']:
                    pid = aid + '-' + str(passage['id'])
                    text = re.sub('\n','',re.sub(r'[^\x20-\x7f]',r'',passage['body']))
                    out.write(f"{pid}\t{title}. {text}\n")
