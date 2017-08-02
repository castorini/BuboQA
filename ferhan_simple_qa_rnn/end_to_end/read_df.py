import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

df = pd.read_pickle("end_df.pkl")
X = df[['ent_score', 'rel_score']]
y = df['label']

X_scaled = preprocessing.scale(X)
classifier = LogisticRegression()
classifier.fit(X_scaled, y)
print("classifier weights: {}".format(classifier.coef_))

pred_proba = classifier.predict_proba(X_scaled)
df['pred_proba_0'] = pred_proba[:,0]
df['pred_proba_1'] = pred_proba[:,1]
# score = classifier.score(X_scaled, y)
# print("mean accuracy: {}".format(score))

df_eval = df[['lineid', 'label', 'pred_proba_0', 'pred_proba_1']]
df_eval = df_eval.groupby('lineid').apply(lambda x: x.sort_values('pred_proba_0'))
df_eval = df_eval.groupby('lineid').first()

correct_count = df_eval['label'].sum()
