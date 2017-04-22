from bs4 import BeautifulSoup
import requests
import re
from collections import namedtuple
import numpy as np
import pandas as pd
import os.path
import sys
#import pylab as pl
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import webbrowser

num_display = 1    # Number of games to display at a time
num_reviews = 6    # Minimum reviews per game
api_calls   = 700  # Number of api calls
shrinkage   = .1   # Shrinkage coef.: degree to which # of reviews reweights
step_size   = .5   # Size of feedback adjustments
calc_table  = 0    # 1 = save to database; 0 = load from database

def main(platforms,feedback):
    # SQL database setup
    #username = 'ubuntu'
    username = 'ubuntu'
    password = 'password'
    #host = 'nextgame-psql-db.c0dlhwlq1k1s.us-west-2.rds.amazonaws.com'
    host = 'localhost'
    #dbname = 'nextgame_psql_db'
    dbname = 'games_db'
    #port = '5432'
    #engine = create_engine('postgresql://%s:%s@%s:%s/%s'%(username,password,host,port,dbname),echo=True)
    engine = create_engine('postgresql://%s:%s@%s/%s'%(username,password,host,dbname),echo=True)
    #engine = create_engine('postgresql://%s@%s/%s'%(username,password,host,dbname),echo=True)
    print('\n**connected**\n')
    if (calc_table == 1) & (feedback is None):
        print('\n**0**\n')
        # Get the names of all games on supplied platforms
        names_and_plats = get_game_names()
        print('\n**1**\n')
        # For each game name, fetch its scores & info
        Games = list()
        for i in range(0,len(names_and_plats.names)):
            g = get_game_info(names_and_plats.names[i])
            if g != -1:
                this_game = [g.title, g.critic_names, g.critic_scores, g.working_score, names_and_plats.plats[i], names_and_plats.covers[i], names_and_plats.genres[i], names_and_plats.summaries[i]]
                Games.append(this_game)
        print('\n**2**\n')
        # Calculate weighted ratings
        Games = calc_weighted_rating(Games)
        print('\n**3**\n')
        # Save Games to Dataframe
        Games_DF = pd.DataFrame(columns=('title', 'critic_names', 'critic_scores', 'working_score', 'platforms', 'cover_url', 'genres', 'summaries'))
        ind = 0
        for game in Games:
            c_names = '--'.join(game[1])
            c_score = '--'.join(str(i) for i in game[2])
            platfms = '--'.join(str(i) for i in game[4])
            g_names = '--'.join(game[6])
            Games_DF.loc[ind] = [game[0],c_names,c_score,game[3],platfms,game[5],g_names,game[7]]
            ind = ind + 1
        print('\n**4**\n')
        # Save Dataframe to SQL database
        Games_DF.to_sql('Games', engine, if_exists='replace', index=False)
        print('\n**5**\n')
    if feedback is None:
        Games_DF = pd.read_sql_table('Games', engine)
        Games_DF.to_sql('Games_temp', engine, if_exists='replace', index=False)
        found_DF = pd.DataFrame(list(), columns=['found'])
        found_DF.to_sql('found', engine, if_exists='replace', index=False)
        ret = adjust_scores(Games_DF,'y0',platforms,1,list())
        Games_DF = ret.Games_DF
        recs = get_recommendations(Games_DF,platforms,ret.found)
        return recs
    else:
        Games_DF_temp = pd.read_sql_table('Games_temp', engine)
        found_DF = pd.read_sql_table('found', engine)
        ret = adjust_scores(Games_DF_temp,feedback,platforms,0,found_DF['found'].tolist())
        Games_DF_temp = ret.Games_DF
        Games_DF_temp.to_sql('Games_temp', engine, if_exists='replace', index=False)
        found_DF = pd.DataFrame(ret.found, columns=['found'])
        found_DF.to_sql('found', engine, if_exists='replace', index=False)
        recs = get_recommendations(Games_DF_temp,platforms,ret.found)
        return recs

def get_game_names():

    # Set up igdb.com API call
    headers={
        "X-Mashape-Key": "uaTzloq2VumshkWJBpZzh7fD7RNqp1ZpWMZjsnK6D7waL44GNQ",
        "Accept": "application/json"
    }
    req1 = "https://igdbcom-internet-game-database-v1.p.mashape.com/genres/?fields=id%2Cname&limit=50"
    req2 = "https://igdbcom-internet-game-database-v1.p.mashape.com/games/?fields=name%2Csummary%2Crelease_dates%2Ccover%2Cgenres&limit=50&offset="

    # Get genres
    genre_result = requests.get(req1, headers=headers)
    ids = re.findall ('"id":(.*?),', genre_result.text, re.DOTALL)
    gnames = re.findall ('"name":"(.*?)"', genre_result.text, re.DOTALL)
    genre_dict = dict()
    for i in range(0,len(ids)):
        genre_dict[ids[i]] = gnames[i]

    # Get game names with igdb.com API
    names = list()
    plats = list()
    covers = list()
    genres = list()
    summaries = list()
    for o in range(0,api_calls):
        req = req2 + str(o*50)
        result = requests.get(req, headers=headers)
        txt = re.findall ('"name"(.*?)"height"', result.text, re.DOTALL)
        for t in range(0,len(txt)):
            nam = re.findall (':"(.*?)","summary"', txt[t], re.DOTALL)
            summ = re.findall ('"summary":"(.*?)"', txt[t], re.DOTALL)
            plts = re.findall ('"platform":(.*?),', txt[t], re.DOTALL)
            genre = re.findall ('"genres":(.*?)"', txt[t], re.DOTALL)
            if len(nam) > 0:
                if len(summ) > 0:
                    if (len(summ[0]) > 150) & (len(genre) > 0) & (',' not in nam[0]) & (('41' in plts) | ('48' in plts) | ('49' in plts)): # If this game is on PS4, Wii U, or XB1
                        cover = re.findall ('"cover"(.*?)"width"', txt[t], re.DOTALL)
                        cloudinary_id = re.findall ('"cloudinary_id":"(.*?)"', cover[0], re.DOTALL)
                        names.append(nam[0])
                        dirty_summ = summ[0]
                        if len(dirty_summ) > 650:
                            dirty_summ = dirty_summ[:650]
                        dirty_summ = dirty_summ[:dirty_summ.rfind(".")] + '.'
                        dirty_summ = dirty_summ.replace("&amp;","&")
                        dirty_summ = dirty_summ.replace("â","'")
                        clean_summ = dirty_summ.replace("\\n"," ")
                        summaries.append(clean_summ)
                        plats.append(list(map(int, plts)))
                        covers.append('https://images.igdb.com/igdb/image/upload/t_cover_big/' + cloudinary_id[0] + '.jpg')
                        tosplit = genre[0]
                        genres_split = tosplit[1:-2].split(",")
                        these_genres = list()
                        for g in genres_split:
                            g_clean = re.sub("[^0-9]", "", g)
                            these_genres.append(genre_dict[g_clean])
                        genres.append(these_genres)

    # Return sorted names
    nt = namedtuple('nt', ['names', 'plats', 'covers', 'genres', 'summaries'])
    ret = nt(names, plats, covers, genres, summaries)
    return ret

def get_game_info(name):

    nameX = name.lower().replace(" ","-")
    link = "https://www.igdb.com/games/" + nameX + "/reviews"
    r = requests.get(link)
    if r.status_code == 200:
        data = r.text
        soup = BeautifulSoup(data, "lxml")
        nums = re.findall(r'\d+%', soup.prettify())
        if len(nums) >= num_reviews:
            cn = re.findall ('<span data-reactid="(.*?)</span>', soup.prettify(), re.DOTALL)
            scores = list()
            cnames = list()
            for n in nums:
                scores.append(int(n[:-1]))
            for c in cn:
                cx = c[15:-9]
                cnames.append(cx[:-1])
            Game = namedtuple('Game', ['title', 'critic_names', 'critic_scores', 'working_score'])
            g = Game(name,cnames,scores,np.mean(scores))
            return g
        else:
            return -1
    else:
        return -1

def calc_weighted_rating(Games):

    num_votes = list()
    for this_game in Games:
        num_votes.append(len(this_game[1]))

    all_ratings = list()
    for this_game in Games:
        # R = mean vote for the game
        R = this_game[3]
        # V = number of votes for the game
        V = len(this_game[1])
        # N = mean number of votes across the whole report
        N = np.mean(num_votes)
        # S = scaled shrinkage coef
        S = shrinkage * N
        weighted_rating = R * ((V+S) / (N+S))
        this_game[3] = weighted_rating
        all_ratings.append(weighted_rating)

    # Normalize
    for this_game in Games:
        this_game[3] = this_game[3] / max(all_ratings)

    return Games

def get_recommendations(Games_DF,platforms,found):

    # Sort games by score
    sorted_DF = Games_DF.sort_values(['working_score'], ascending=[False])

    # Plot working scores
    #sanity_plot(sorted_DF['working_score'].tolist())

    # Display highest scored games
    names = list()
    covers = list()
    genres = list()
    glink = list()
    amazn = list()
    summary = list()
    i = 0
    score_of_rec = 0
    while len(names) < num_display:
        plats = sorted_DF.iloc[i]['platforms']
        plats_split = plats.split("--")
        if not (set(int(j) for j in plats_split).isdisjoint(set(platforms))):
            # Name
            names.append(sorted_DF.iloc[i]['title'])
            # Picture
            covers.append(sorted_DF.iloc[i]['cover_url'])
            # Genres
            genres.append(sorted_DF.iloc[i]['genres'].replace("--",", "))
            # Google Link
            glink.append('http://www.google.com/search?q=' + sorted_DF.iloc[i]['title'].replace(" ","+"))
            # Amazon Link
            amazn.append('http://www.amazon.com/s?url=search-alias%3Daps&field-keywords=' + sorted_DF.iloc[i]['title'].replace(" ","+"))
            # Summary
            summary.append(sorted_DF.iloc[i]['summaries'])
            # Score
            score_of_rec = float(sorted_DF.iloc[i]['working_score'])

        i = i + 1

    nt = namedtuple('nt', ['names','covers','genres','glink','amazn','summary','found','score'])
    ret = nt(names, covers, genres, glink, amazn, summary, found, score_of_rec)
    return ret

def adjust_scores(Games_DF,feedback,platforms,initialize,found):

    # Calculate similarities
    games = list(Games_DF.title)
    sims_in_line = list()
    prenorm_ARDs = list()
    for x in range(0,len(games)):
        pna_row = list()
        for y in range(0,len(games)):
            if x == y:
                pna_row.append(0);
            else:
                # PreNorm Average Review Difference
                cns = list(Games_DF.critic_names)
                cnamesX = cns[x].split("--")
                cnamesY = cns[y].split("--")
                scs = list(Games_DF.critic_scores)
                scoresX = scs[x].split("--")
                scoresY = scs[y].split("--")
                reviewed_both = [val for val in cnamesX if val in cnamesY]
                diffs = list()
                for c in reviewed_both:
                    indX = [i for i, j in enumerate(cnamesX) if j == c]
                    indY = [i for i, j in enumerate(cnamesY) if j == c]
                    diffs.append(int(scoresX[indX[0]]) - int(scoresY[indY[0]]))
                if len(diffs) > 0:
                    pna_row.append(abs(np.mean(diffs)))
                else:
                    pna_row.append(1) # No overlapping reviews => disimilar
                ##
        sims_in_line.extend(pna_row)
        prenorm_ARDs.append(pna_row)

    # Normalize
    max_ard = max(sims_in_line)
    ARDs = prenorm_ARDs / max_ard

    similarities = list()
    for x in range(0,len(games)):
        sim_row = list()
        for y in range(0,len(games)):
            if x == y:
                sim_row.append(1);
            else:
                # Percent Genre Overlapping (PGO)
                grs = list(Games_DF.genres)
                genresX = grs[x].split("--")
                genresY = grs[y].split("--")
                num_shared = len([val for val in genresX if val in genresY])
                Xonly = len(genresX) - num_shared
                Yonly = len(genresY) - num_shared
                PGO = num_shared / (Xonly + Yonly + num_shared)
                ##

                # Average Review Difference (ARD)
                ARD = 1 - ARDs[x][y]
                ##

                # Similarity equation
                SIM = (ARD * PGO) / (ARD + PGO)
                ##

                sim_row.append(SIM)
        similarities.append(sim_row)

    if initialize != 1:

        # Parse feedback
        pf = parse_feedback(Games_DF,feedback,platforms,found)
        found = pf.found

        # Save for bipartite graph
        Games_DF_old = Games_DF.copy()

        # Adjust scores
        for x in range(0,(len(games)-1)):
            if list(Games_DF.title)[x] == pf.fb_game:
                fb_game_ind = x
        for x in range(0,(len(games)-1)):
            if list(Games_DF.title)[x] != pf.fb_game:
                pre_score = Games_DF.iloc[x]['working_score']
                similarity = similarities[x][fb_game_ind]
                if pre_score != 0:
                    # Adjustment equation
                    post_score = pre_score + (similarity * pf.fb_valu * step_size)
                    ##
                    Games_DF.ix[x,'working_score'] = post_score
        Games_DF.ix[fb_game_ind,'working_score'] = 0
        #bipartite(Games_DF_old,Games_DF,platforms)

    nt = namedtuple('nt', ['Games_DF','found'])
    return nt(Games_DF, found)

def parse_feedback(Games_DF,feedback,platforms,found):

    # Sort games by score
    sorted_DF = Games_DF.sort_values(['working_score'], ascending=[False])

    # Display highest scored games
    names = list()
    covers = list()
    i = 0
    while len(names) < num_display:
        plats = sorted_DF.iloc[i]['platforms']
        plats_split = plats.split("--")
        if not (set(int(j) for j in plats_split).isdisjoint(set(platforms))):
            names.append(sorted_DF.iloc[i]['title'])
        i = i + 1
    fb_game = names[0]
    if feedback[0] == 'a':
        fb_valu = 2
    elif feedback[0] == 'b':
        fb_valu = 1
        found.append(fb_game)
    elif feedback[0] == 'c':
        fb_valu = -1
    elif feedback[0] == 'd':
        fb_valu = -2
    nt = namedtuple('nt', ['fb_game','fb_valu','found'])
    return nt(fb_game, fb_valu, found)

def sanity_plot(working_scores):

    pass
    #pl.hist(working_scores, bins='auto')
    #pl.title('Game Score Distribution')
    #pl.xlabel('Rating')
    #pl.ylabel('Count')
    #i = 1
    #fname = '/Users/Caleb/Downloads/fig_' + str(i) + '.png'
    #while os.path.isfile(fname):
    #    i = i + 1
    #    fname = '/Users/Caleb/Downloads/fig_' + str(i) + '.png'
    #pl.savefig(fname)
    #pl.close()

def bipartite(old_DF,new_DF,platforms):

    s_old_DF = old_DF.sort_values(['working_score'], ascending=[False])
    s_new_DF = new_DF.sort_values(['working_score'], ascending=[False])

    n = 15
    old_names = list()
    i = 0
    while len(old_names) < n:
        plats = s_old_DF.iloc[i]['platforms']
        plats_split = plats.split("--")
        if not (set(int(j) for j in plats_split).isdisjoint(set(platforms))):
            old_names.append(s_old_DF.iloc[i]['title'])
        i = i + 1

    new_names = list()
    i = 0
    while len(new_names) < n:
        plats = s_new_DF.iloc[i]['platforms']
        plats_split = plats.split("--")
        if not (set(int(j) for j in plats_split).isdisjoint(set(platforms))):
            new_names.append(s_new_DF.iloc[i]['title'])
        i = i + 1

    for x in range(0,len(old_names)):
        pass
        #print(str(x) + '. ' + old_names[x] + ' -> ' + new_names[x])

if __name__ == '__main__':
    #PS4  code = 48
    #WiiU code = 41
    #XB1  code = 49
    main([48,41,49], None)
