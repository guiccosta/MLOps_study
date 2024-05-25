from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

data = load_iris(as_frame=True)
df = data.frame

print(df.info())