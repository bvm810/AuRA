import sys
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

path = sys.argv[1]
df = pd.read_csv(path)

val_size = max(1, min(1024, int(len(df) * 0.01)))

train_idx, val_idx = train_test_split(df.index, test_size=val_size, random_state=SEED)
df["split"] = "train"
df.loc[val_idx, "split"] = "validation"

df.to_csv(path, index=False)