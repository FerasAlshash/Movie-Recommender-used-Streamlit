import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer , util
 
st.title('Movie Recommendition app!')
st.write('This is a movie recommendation app using Streamlit and Python')
 
data = pd.read_csv('movies.csv')
 
movie = st.text_input('Enter your favourite movie', key='movie')
 
#3- Semantic Search
 
model = SentenceTransformer('all-MiniLM-L6-v2')
def semantic_search( movie_title, df ):
 
    movie_titles = data['movie_name'].tolist()
    title_embedding = model.encode(movie_title , convert_to_tensor=True)
    movie_embeddings = model.encode(movie_titles , convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(title_embedding, movie_embeddings)
 
    best_similar = similarity.argmax().item() # index of the best similar movie
    print(best_similar)
 
    best_similar_title = movie_titles[best_similar]
 
    best_similar_score = df.iloc[best_similar]['imdb_rating']
    return best_similar_title, best_similar_score
 
 
if st.button("Movie IMDB Score !"):
    if movie :
        movie_title,score = semantic_search (movie , data)
        st.write(score)
        st.write(movie_title)
    else:
        st.write('Movie doesnt exist !')
 
 
st.write(data)