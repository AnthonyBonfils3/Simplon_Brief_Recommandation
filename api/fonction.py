import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM

def create_ap():
    plays = pd.read_csv('/home/anthony/Documents/Briefs/2_Block_janv_fev/20210112_recommendation_system/api/static/lastfm/user_artists.dat', sep='\t') ## Dataset of relation between an artist 

    ## an user and the number listening 
    artists = pd.read_csv('/home/anthony/Documents/Briefs/2_Block_janv_fev/20210112_recommendation_system/api/static/lastfm//artists.dat', sep='\t', usecols=['id','name']) ## id	name	url	pictureURL

    # Merge artist and user pref data
    ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
    ap = ap.rename(columns={"weight": "playCount"})

    # Group artist by name
    artist_rank = ap.groupby(['name']) \
        .agg({'userID' : 'count', 'playCount' : 'sum'}) \
        .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
        .sort_values(['totalPlays'], ascending=False)

    artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']

    # Merge into ap matrix
    ap = ap.join(artist_rank, on="name", how="inner") \
        .sort_values(['playCount'], ascending=False)

    return ap

def preprocessing(ap):
    # Preprocessing
    pc = ap.playCount
    play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
    ap = ap.assign(playCountScaled=play_count_scaled)
    return ap

def get_ratings_df(ap):
    # Build a user-artist rating matrix 
    ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')
    return ratings_df

def get_X(ratings_df):
    ratings = ratings_df.fillna(0).values

    # Build a sparse matrix
    X = csr_matrix(ratings)
    Xcoo = X.tocoo()
    return Xcoo

def add_new_user(ratings_df, select, artist_names, ap):
    user_ids = ratings_df.index.values
    new_user = max(user_ids)+1
    new_user_artist = np.zeros(len(artist_names))
    i=0
    for artist in artist_names:
        if artist in select:
            print(artist)
            new_user_artist[i] = ap.playCountScaled[ap['name']==artist] \
                .mean()
        i +=1  
    ratings_df.loc[new_user] = new_user_artist
    return ratings_df, new_user

def fit_model(X):
    learn_rate = 0.05
    nb_epochs = 25
    k = 10
    loss = 'warp-kos'
    nb_comp = 20
    model = LightFM(learning_rate=learn_rate, k=k, loss=loss, 
                 random_state = 42, no_components=nb_comp)
    model.fit(X, epochs=nb_epochs, num_threads=2)
    return model

def get_recommandation(userID, model, user_ids, ap, n_reco=10):
    artist_names = ap.sort_values("artistID")["name"].unique()
    ratings_df = get_ratings_df(ap)
    n_users, n_items = ratings_df.shape
    liste_user_idx = list(user_ids)
    idx = liste_user_idx.index(userID)
    scores = model.predict(idx, np.arange(n_items))
    top_items_pred = artist_names[np.argsort(-scores)]
    return top_items_pred
    


if __name__ == '__main__':
    pass
