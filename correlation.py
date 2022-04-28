
import pandas as pd


path_to_dataset = 'data/dump_esp8266-244085.csv'

threshold = 30

df = pd.read_csv(path_to_dataset)

print(df.corr(method='pearson'))