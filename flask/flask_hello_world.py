# flask hello world rest api tutorial from
# https://realpython.com/flask-connexion-rest-api/

from flask import Flask, render_template

# create instance
app = Flask(__name__, template_folder='templates')


# create route at '/'
@app.route('/')
def home():
    '''at localhost:5000/'''
    return render_template('home.html')



if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)