import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir_path = "~/dev/simple-qa-on-kb/ferhan_simple_qa_rnn/exploration"
data_path = os.path.join(dir_path, "freebase-FB2M.txt")

df = pd.read_table(data_path, "\t", header=None, names=["mid", "relation", "object"])

print(df.describe())

rel_counts = df['relation'].value_counts()
print(rel_counts.describe())
# print(rel_counts[rel_counts <= 10].describe())
# print(rel_counts[rel_counts > 10].describe())
# rel_counts.plot(logy=True)

mid_counts = df['mid'].value_counts()
print(mid_counts.describe())
# print(mid_counts[mid_counts <= 10].describe())
# print(mid_counts[mid_counts > 10].describe())
# mid_counts.plot(logy=True)
