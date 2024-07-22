from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def receive_data():
    data = request.get_json()
    x = data['x']
    y = data['y']
    z = data['z']
    pitch = data['pitch']
    roll = data['roll']
    yaw = data['yaw']
    
    # Do something with the data
    print(f"Received data: x={x}, y={y}, z={z}, pitch={pitch}, roll={roll}, yaw={yaw}")
    
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)