from flask import Flask, jsonify
from em import get_dataset, gmm

app = Flask(__name__)

@app.route('/')
def index():
    outcome = None
    dataset = get_dataset(num_of_samples=30)
    model, thetas = gmm(dataset)
    return jsonify({
        "Biases": {
            "Coin A": thetas[0],
            "Coin B": thetas[1]
        },
        "msg" : "Please note that the parameters are sorted."        
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)