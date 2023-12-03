import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", type=str, default='sample.qrels')
    args = parser.parse_args()

    data = pd.read_csv(args.tsv, header=None, delimiter='\t')

