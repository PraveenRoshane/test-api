from flask import Flask, request, jsonify

from body_language import process_video_and_predict

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/analyze_body_language', methods=['POST'])
def analyze_body_language():
    data = request.json
    video_url = data['video_url']
    result = process_video_and_predict(video_url)
    return jsonify({"result": result})


if __name__ == '__main__':
    app.run()
