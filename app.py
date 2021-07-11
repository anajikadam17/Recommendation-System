import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, url_for, request
app = Flask(__name__)

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity
    
def get_movie_name():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

def rcmd(m, data, similarity):
    m = m.lower()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

def similar_movie(movie, df, similarity):
    rc = rcmd(movie, df, similarity)
    if type(rc)==type('string'):
        #print(rc)
        return [rc]
    else:
        #print(rc)
        rc = [(str(i+1)+'. '+j.capitalize()) for i,j in enumerate(rc)]
        return rc
 
def get_movies_name():
    df = pd.read_csv('data.csv')
    movies_names = list(set(df['title']))
    return movies_names

def Recom_movies_ByMovie(df, movie):
    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())                    # Ratings dataframe by groupby title by rating
    ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count()) # Add rating counts columns
    moviemat = df.pivot_table(index='user_id',columns='title',values='rating')      # Pivot table for movie title and user_id
    user_ratings = moviemat[movie]                                         # for particular given movie
    similar_to_movie = moviemat.corrwith(user_ratings)                     # find (similar movie) corr with given movie dataframe
    corr_movie = pd.DataFrame(similar_to_movie,columns=['Correlation'])    # give name to column Correlation
    corr_movie.dropna(inplace=True)                                        # Drop null value raws
    corr_movie = corr_movie.join(ratings['num of ratings'])                # join column num of ratings from ratings dataframe
    # final correlated movie with given movie for only count of rating greater than 100
    final_corr_movie = corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)
    # find out list of five recommended movie by given movie based 
    rec = final_corr_movie.index[1:11].tolist()
    return rec
    
df, similarity = create_similarity()
df1 = pd.read_csv('data.csv')
@app.route('/')
@app.route('/index')
def index():
    suggestion = get_movie_name()
    return render_template('index.html', suggestions=suggestion)

@app.route('/index1')
def index1():
    suggestion = get_movies_name()
    return render_template('index1.html', suggestions=suggestion)
    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        movie_title = request.form.get("movie_title")
        print(movie_title)
        similar_movies = similar_movie(movie_title,df, similarity)
        return render_template("result.html", movie_title = movie_title, similar_movies = similar_movies)
    return render_template("result.html", movie_title = None)

@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    if request.method == 'POST':
        movie_title = request.form.get("movie_title")
        print(movie_title)
        similar_movies = Recom_movies_ByMovie(df1, movie_title)
        return render_template("result2.html", movie_title = movie_title, similar_movies = similar_movies)
    return render_template("result2.html", movie_title = None)
    
if __name__ == '__main__':
    app.run(debug=True)

