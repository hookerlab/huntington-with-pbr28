import pandas as pd

df1 = pd.read_csv('./data/participants_with_labels.tsv', sep='\t')
df2 = pd.read_csv('./data/regional_volumes_in_mm3.csv')
df2['label_plot'] = df2.index.values + 1

df = pd.merge(df1, df2, how='outer', on='subject_id', suffixes=('', '_drop'))
df = df.drop(columns=['intracraneal_volume_drop'])
df.to_csv('./data/participants_with_labels_and_vols.csv')
