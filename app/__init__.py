from flask_api import FlaskAPI
from flask import request, jsonify
from flask_cors import CORS

def create_app():
    app = FlaskAPI(__name__, instance_relative_config=True)

    CORS(app)

    @app.route('/qa/', methods=['POST'])
    def ask():
        if request.method == "POST":
            question = str(request.data.get('question'))
            candidate = str(request.data.get('candidate'))

            #TODO: answer id in the request probably

            if question and candidate:

                #TODO: calculate probability

                probability = 0.456

                response = jsonify({
                    'answerid': '100001',
                    'probability': probability
                })

                response.status_code = 200

                return response

    return app