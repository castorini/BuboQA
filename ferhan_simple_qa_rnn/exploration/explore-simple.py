import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir_path = "~/dev/simple-qa-on-kb/ferhan_simple_qa_rnn/exploration"
data_path = os.path.join(dir_path, "simpleqa-entity-linking-numbered-dataset.txt")

df = pd.read_table(data_path, " %%%% ", engine="python", header=None, names=["lineid", "mid", "relation", "object", "question", "entity_names"])

print(df.describe())

rel_counts = df['relation'].value_counts()
print(rel_counts[rel_counts <= 10].describe())
print(rel_counts[rel_counts > 10].describe())
rel_counts.plot()

mid_counts = df['mid'].value_counts()
print(mid_counts[mid_counts <= 10].describe())
print(mid_counts[mid_counts > 10].describe())
mid_counts.plot()
