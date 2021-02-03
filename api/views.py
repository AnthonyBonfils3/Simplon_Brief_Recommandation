#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:52:12 2021

@author: Bonfils Anthony
"""

from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from api.fonction import create_ap, get_recommandation, get_X
from api.fonction import get_ratings_df, fit_model, add_new_user, preprocessing


## load DataFrame and merge them
ap, artists = create_ap()
ap = preprocessing(ap)
artist_names = ap.sort_values("artistID")["name"].unique()

#############################################################
##########################         ##########################
##########################   APP   ##########################
##########################         ##########################
#############################################################

app = Flask(__name__)

## page 1 : index
@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        pass
    letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
                   'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                   'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] 
    letter = request.form.getlist("Lettre")
    artists_names = sorted(list(artist_names))
    return render_template('index.html', artists_names=artists_names[60:-725], letter_list=letter_list)

## page 2 : results
@app.route('/result', methods = ['POST'])
def result():
    ## liste retournée par page index
    user_artist_list = request.form.getlist("research")
    print(user_artist_list)
    ## Pivot Matrix rating    
    ratings_df = get_ratings_df(ap)

    ## compute the new ratings_df matrix with the new user
    ratings_df, user = add_new_user(ratings_df, user_artist_list, artist_names, ap)
    ## update the new user list                            
    user_ids = ratings_df.index.values 
    ## compute matrix X
    X = get_X(ratings_df) 

    ## compute model
    model = fit_model(X)

    ## Make predictions
    top_items_pred = get_recommandation(user, model, user_ids, ap, ratings_df,artist_names, n_reco=10)
    reco = [[artist, artists[artists['name']==artist].url.values[0]] for artist in top_items_pred if artist not in user_artist_list]
    reco = reco[:20]

    return render_template('result.html', selection = user_artist_list, recommandations=reco)


if __name__ == "__main__":
    app.run()
    

    