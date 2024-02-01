import collections

def load_query(path: str):
    data = collections.defaultdict(str)
    with open(path, 'r') as f:
        for line in f:
            if path.endswith('tsv'):
                qid, qtext = line.split('\t')
                data[str(qid.strip())] = qtext.strip()
            else:
                raise ValueError("Invalid data extension.")
    return data

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]