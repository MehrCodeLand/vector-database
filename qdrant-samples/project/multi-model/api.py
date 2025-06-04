
from flask import Flask, request, jsonify
from main_service import ImageSearchService
import os

app = Flask(__name__)

# Initialize the search service
search_service = ImageSearchService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Animal Image Search API"})

@app.route('/initialize', methods=['POST'])
def initialize_database():
    try:
        success = search_service.initialize_database()
        if success:
            return jsonify({
                "status": "success",
                "message": "Database initialized successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to initialize database"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error initializing database: {str(e)}"
        }), 500

@app.route('/search', methods=['POST'])
def search_images():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "status": "error",
                "message": "Query parameter is required"
            }), 400

        query = data['query']
        limit = data.get('limit', 5)
        
        if limit > 20:  # Limit maximum results
            limit = 20

        results = search_service.search_by_query(query, limit)
        
        return jsonify({
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing search: {str(e)}"
        }), 500

@app.route('/search/<query>', methods=['GET'])
def search_images_get(query):
    try:
        limit = request.args.get('limit', 5, type=int)
        
        if limit > 20:
            limit = 20

        results = search_service.search_by_query(query, limit)
        
        return jsonify({
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing search: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)