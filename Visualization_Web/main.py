from flask import Flask, render_template
from data import Data
from court import CourtShape
#
app = Flask(__name__)
data = Data()
court_shapes = CourtShape()

@app.route('/')
@app.route('/<string:chosen_name>')
# @app.route('/profile/<name>')
def profile(chosen_name='Kobe'):
    if not chosen_name: chosen_name = 'Kobe'
    for d in data:
        if d['player_name'] == chosen_name:
            data_piece = d
    return render_template('profile.html', player=chosen_name, data=data_piece)


# showing the distribution on the court
@app.route('/layout')
def distribution(chosen_name=None):
    if not chosen_name: chosen_name = 'Kobe'
    print(data)
    for d in data:
        if d['player_name'] == chosen_name:
            data_piece = d

    return render_template('distribution.html', court=court_shapes, data=data_piece)


@app.route('/trending')
def create():
    return render_template('trending.html')

#
# @app.route('/')
# def main():
#     return render_template('stats.html')
@app.route('/percent')
def percent():
    return  render_template('stats.html')

if __name__ == "__main__":
    app.run()