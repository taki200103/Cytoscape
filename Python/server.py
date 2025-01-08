from flask import Flask
import subprocess

app = Flask(__name__)

@app.route('/run_node_embedding', methods=['GET'])
def run_node_embedding():
    try:
        # Chạy file node_embedding.py
        subprocess.run(['python', 'E:\\OOP\\Project_OOP\\Python\\node_embedding.py'], check=True)
        return "Node Embedding completed successfully", 200
    except subprocess.CalledProcessError as e:
        return f"Error running node_embedding.py: {str(e)}", 500

@app.route('/run_add_view', methods=['GET'])
def run_add_view():
    try:
        # Chạy file add_view.py
        subprocess.run(['python', 'E:\\OOP\\Project_OOP\\Python\\add_view.py'], check=True)
        return "Add View completed successfully", 200
    except subprocess.CalledProcessError as e:
        return f"Error running add_view.py: {str(e)}", 500
    
@app.route('/run_link_prediction', methods=['GET'])
def run_link_prediction():
    try:
        # Chạy file link_prediction.py
        subprocess.run(['python', 'E:\\OOP\\Project_OOP\\Python\\link_prediction.py'], check=True)
        return "Link Prediction completed successfully", 200
    except subprocess.CalledProcessError as e:
        return f"Error running link_prediction.py: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
