import re
with open("treccar.tsv","w") as out:
    with open("dedup.articles-paragraphs.cbor.xml","r") as f:
        title_next=False
        body_next=False
        for line in f:
            if line[:7]=="<DOCNO>":
                docid=line[7:-9]
            if title_next:
                title=line.rstrip()
                title_next=False
            if line[:7]=="<TITLE>":
                title_next=True
            if body_next:
                body=line.rstrip()
                out.write(f"{docid}\t{title}. {body}\n")
                body_next=False
            if line[:6]=="<BODY>":
                body_next=True