import pandas as pd
df = pd.DataFrame.from_csv('../working/out.csv').head(200000)
dictionary = dict()
for token in df['word']:
  dictionary[token] = token
print(len(dictionary.keys()))
