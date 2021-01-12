import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('./metadata.csv')

# get the number of missing data points per column
count_missing_values = df.isnull().sum()

# how many total missing values?
total_cells = np.product(df.shape)
total_missing = count_missing_values.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100

# create a new colum, published_time_parsed, with the parsed publish_times
df['published_time_parsed'] = pd.to_datetime(df['publish_time'], format='%Y/%m/%d')

drop_columns = ['cord_uid', 'sha', 'source_x', 'doi', 'pmcid', 'pubmed_id', 'license', 'publish_time',
                'journal', 'mag_id', 'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files',
                's2_id']

df = df.drop(drop_columns, axis=1)
df = df.dropna()

stopwords = ['vaccine', 'mrna', 'mRNA', 'therapeutics']
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words=stopwords)
dtm = cv.fit_transform(df['abstract'])

LDA = LatentDirichletAllocation(n_components=7, random_state=42)

print(LDA.fit(dtm))
print(done)

# print(df.head())
