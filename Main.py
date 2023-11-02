# -*- coding: utf-8 -*-


#Basic Libraries
import numpy as np
import pandas as pd



#Text Handling Libraries
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

df = pd.read_csv('BigBasket Products.csv',index_col='index')

df.isnull().sum()

print('Total Null Data')
null_count = df.isnull().sum().sum()
total_count = np.product(df.shape)
print("{:.2f}".format(null_count/total_count * 100))
df = df.dropna()
null_count = df.isnull().sum().sum()
total_count = np.product(df.shape)
print("{:.2f}".format(null_count/total_count * 100))

df2 = df.copy()
rmv_spc = lambda a:a.strip()
get_list = lambda a:list(map(rmv_spc,re.split('& |, |\*|\n', a)))
for col in ['category', 'sub_category', 'type']:
    df2[col] = df2[col].apply(get_list)

def cleaner(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

for col in ['category', 'sub_category', 'type','brand']:
    df2[col] = df2[col].apply(cleaner)
def couple(x):
    return ' '.join(x['category']) + ' ' + ' '.join(x['sub_category']) + ' '+x['brand']+' ' +' '.join( x['type'])
df2['soup'] = df2.apply(couple, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['product'])

def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return df2['product'].iloc[movie_indices]

def find(name,number):
    new_rec = get_recommendations(name, cosine_sim2).values
    df1 = pd.DataFrame({'product':new_rec})
    result = pd.merge(df1, df2, on='product', how='left')
    return result[['product', 'category', 'brand', 'sale_price', 'rating', 'description']].head(number)

find('Cadbury Perk - Chocolate Bar',5 )

import streamlit as st


def similar(target_word):
    # Find similar words within the DataFrame
    similar_words = df2[df2['product'].str.contains(target_word, case=False, na=False)]
    df = pd.DataFrame(similar_words)
    return df['product'].to_numpy()


st.image('dataset-cover.jpg')


try:
    st.text('Enter Product')
except:
    # Handle the exception and display a warning message
    st.warning("No Product Found")


val = st.text_input("Name")
song = st.selectbox("Pick one", similar(val))
inp = st.slider('No of Recommender Products', 0, 40, 5)


if st.button('Similar products'):
    value = find('Cadbury Perk - Chocolate Bar',5 )
    st.dataframe(value)
    st.button("Reset", type="primary")
else:
    st.write('')

