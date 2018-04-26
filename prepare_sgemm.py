import pandas as pd

# Data from https://github.com/vgeorge/pnad-2015/tree/master/dados
df = pd.read_csv("dbs/sgemm_product.csv")

mean_col = df.iloc[:, -1] + df.iloc[:, -2]
mean_col += df.iloc[:, -3] + df.iloc[:, -4]
mean_col /= 4
df = pd.concat([df.iloc[:, :14], mean_col], axis=1)
columns = list(df.columns)
columns[-1] = "run (ms)"
df.columns = columns
