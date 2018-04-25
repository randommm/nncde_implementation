import pandas as pd

# Data from https://github.com/vgeorge/pnad-2015/tree/master/dados
df = pd.read_csv("dbs/sgemm_product.csv")

dfs = []
for i, ind in enumerate(range(14, 18)):
    dfs.append(pd.concat([df.iloc[:, :14], df.iloc[:, ind]], axis=1))
    columns = list(dfs[i].columns)
    columns[-1] = "run (ms)"
    dfs[i].columns = columns

df = pd.concat(dfs, join='inner')
