import logging

import flask
import tensorflow as tf
from flasgger import Swagger
from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from src.translate_spanish import translate_sentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# NOTE this import needs to happen after the logger is configured


# Initialize the Flask application
application = Flask(__name__)

application.config['ALLOWED_EXTENSIONS'] = set(['pdf'])
application.config['CONTENT_TYPES'] = {"pdf": "application/pdf"}
application.config["Access-Control-Allow-Origin"] = "*"


CORS(application)

swagger = Swagger(application)

modelfile = 'models/model_v0.h5'
graph = tf.get_default_graph()

def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp

@application.route('/')
def my_form():
    return render_template('index.html')

@application.route('/', methods=['POST', 'GET'])
def translate_s():
    """Run translation given text.
        ---
        parameters:
          - name: body
            in: body
            schema:
              id: text
              required:
                - text
              properties:
                text:
                  type: string
            description: the required text for POST method
            required: true
        definitions:
          SentimentResponse:
          Project:
            properties:
              status:
                type: string
              ml-result:
                type: object
        responses:
          40x:
            description: Client error
          200:
            description: Translated sentences in English
            examples:
                          [
{
  "status": "success",
  "english": "i be pleased"
},
{
  "status": "error",
  "message": "Exception caught"
},
]
        """
    global graph
    with graph.as_default():
      text = request.form['text']
      if text is None:
          return Response("No text provided.", status=400)
      else:
          english_trans = translate_sentences(modelfile, text)
          return flask.jsonify({"status": "success", "english": english_trans})

if __name__ == '__main__':
    application.run(debug=True, use_reloader=True)