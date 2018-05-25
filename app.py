import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from network import Network

app = Flask(__name__)

hyper_param = np.load('./data/model.npz', encoding='bytes')
NN = Network(hyper_param['weights'], hyper_param['biases'])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/mnist', methods=['POST'])
def api():
    inpt = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(784, 1)
    out = NN.feedforward(inpt)
    print(out.argmax())
    
    return jsonify({'payload': str(out.argmax())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
