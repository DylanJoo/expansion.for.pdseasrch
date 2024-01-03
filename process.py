from tqdm import tqdm
import json

fout = open('data/mlsr_corpus/25000/corpus.new.jsonl', 'w')

with open('data/mlsr_corpus/25000/corpus.jsonl', 'r') as f:
    for line in tqdm(f):
        data = json.loads(line.strip())
        docid = data['doc_id']
        contents = data.get('title', "") + " " + data.get('description', "")
        vector = data.get('vector')
        fout.write(json.dumps({
            "id": docid, 
            "contents": contents, 
            "vector": vector
        }, ensure_ascii=False)+'\n')  

