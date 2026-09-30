import pandas as pd
from build_cost_matrix import load_cost_matrix

C, days, meta = load_cost_matrix("cost_matrix_output")   # csak a mappa, "cost_matrix" nélkül!
labels = [d.date() for d in days]

df = pd.DataFrame(C, index=labels, columns=labels)
df.to_csv("cost_matrix_readable.csv")
print(df.iloc[:10, :10])