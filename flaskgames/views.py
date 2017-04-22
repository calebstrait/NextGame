from flask import Flask, render_template, request, session
from flaskgames import app
import next_game

app.secret_key = '2435#$5@#45#$5345'

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/output')
def games_output():
    feedback = request.args.get('fb')
    PS4 = request.args.get('PS4')
    WiiU = request.args.get('WiiU')
    XB1 = request.args.get('XB1')
    platforms = list()
    if type(PS4).__name__ == 'str':
        platforms.append(48)
    if type(WiiU).__name__ == 'str':
        platforms.append(41)
    if type(XB1).__name__ == 'str':
        platforms.append(49)
    if len(platforms) == 0:
        plats = session.get('session_plats', None)
        platforms = plats.split("--")
        platforms = list(map(int, platforms))
    else:
        plats = '--'.join(str(i) for i in platforms)
        session['session_plats'] = plats

    recs = next_game.main(platforms,feedback)

    found_dict = dict()
    j = 0
    found = recs.found
    for f in found:
        fou = dict()
        fou['title'] = f
        fou['glink'] = 'http://www.google.com/search?q=' + f.replace(" ","+")
        fou['amazn'] = 'http://www.amazon.com/s?url=search-alias%3Daps&field-keywords=' + f.replace(" ","+")
        found_dict[j] = fou
        j = j + 1

    return render_template("output.html", pic = recs.covers[0], title = recs.names[0], genres = recs.genres[0], glink = recs.glink[0], amazn = recs.amazn[0], summary = recs.summary[0], found_dict = found_dict, recscore = recs.score)

@app.route('/about')
def about():
    return render_template("about.html")
