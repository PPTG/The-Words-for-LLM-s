from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import os

app = Flask(__name__)
API_URL = os.environ.get("API_URL", "http://api:8000/api")

# Main routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/keywords')
def keywords():
    return render_template('keywords.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# API routes for keywords
@app.route('/api/keywords/list')
def list_keywords():
    try:
        response = requests.get(f"{API_URL}/keywords")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/keywords/add', methods=['POST'])
def add_keyword():
    try:
        data = {
            "keyword": request.form['keyword'],
            "webhook_id": request.form['webhook_id'],
            "backend_type": request.form['backend_type'],
            "description": request.form['description']
        }
        response = requests.post(f"{API_URL}/keywords", json=data)
        return jsonify({"status": "success", "data": response.json()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/keywords/edit/<int:id>', methods=['POST'])
def edit_keyword(id):
    try:
        data = {
            "keyword": request.form['keyword'],
            "webhook_id": request.form['webhook_id'],
            "backend_type": request.form['backend_type'],
            "description": request.form['description']
        }
        response = requests.put(f"{API_URL}/keywords/{id}", json=data)
        return jsonify({"status": "success", "data": response.json()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/keywords/delete/<int:id>')
def delete_keyword(id):
    try:
        response = requests.delete(f"{API_URL}/keywords/{id}")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API routes for config
@app.route('/api/config/list')
def list_config():
    try:
        response = requests.get(f"{API_URL}/config")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config/update/<string:id>', methods=['POST'])
def update_config(id):
    try:
        data = {
            "id": id,
            "value": request.form['value'],
            "description": request.form.get('description', '')
        }
        response = requests.put(f"{API_URL}/config/{id}", json=data)
        return jsonify({"status": "success", "data": response.json()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health endpoint
@app.route('/api/health')
def health():
    try:
        response = requests.get(f"{API_URL}/health")
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "services": {
                "llama": "error",
                "ollama": "error", 
                "n8n": "error",
                "flowise": "error"
            },
            "config": {},
            "keywords": {"count": 0}
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)