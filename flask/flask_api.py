# flask hello world rest api tutorial from
# https://realpython.com/flask-connexion-rest-api/

from flask import Flask, render_template
import connexion

# create instance
# app = Flask(__name__, template_folder='templates')
app = connexion.FlaskApp(__name__, specification_dir='./')
app.add_api('swagger.yml', options={'swagger_ui':True}) # read config



# create route at '/'
@app.route('/')
def home():
    '''at localhost:5000/'''
    return render_template('home.html')



if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

    # localhost:5000/api/people <-- for get call
    # localhost:5000/api/ui <-- for ui
    # pip install connexion"[swagger-ui]"
    # https://connexion.readthedocs.io/en/latest/quickstart.html