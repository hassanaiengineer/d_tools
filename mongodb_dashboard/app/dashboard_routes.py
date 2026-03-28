from flask import Blueprint, jsonify, Response, request
from app.utils import get_filtered_data
from app.visualizations import generate_weekly_trends, generate_heatmap, calculate_key_metrics

# Create Blueprint for the dashboard
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route("/", methods=["GET"])
def home():
    return "Welcome to the Taxi Fleet Dashboard API!"

@dashboard_bp.route("/api/key-metrics", methods=["GET"])
def key_metrics():
    try:
        # Retrieve the limit parameter from the query string
        limit = int(request.args.get("limit", 0))  # Default to 0 if not provided
        data = get_filtered_data(limit=limit)
        if not data:
            return jsonify({"error": "No data retrieved from the database. Check your query or data availability."}), 500
        metrics = calculate_key_metrics(data)
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Failed to calculate key metrics: {str(e)}"}), 500

@dashboard_bp.route("/api/weekly-trends", methods=["GET"])
def weekly_trends():
    try:
        # Retrieve the limit parameter from the query string
        limit = int(request.args.get("limit", 0))  # Default to 0 if not provided
        data = get_filtered_data(limit=limit)
        if not data:
            return jsonify({"error": "No data retrieved from the database. Check your query or data availability."}), 500
        weekly_trend_chart = generate_weekly_trends(data)
        return jsonify(weekly_trend_chart)
    except Exception as e:
        return jsonify({"error": f"Failed to generate weekly trends: {str(e)}"}), 500

@dashboard_bp.route("/api/heatmap-data", methods=["GET"])
def heatmap_data():
    try:
        # Retrieve the limit parameter from the query string
        limit = int(request.args.get("limit", 0))  # Default to 0 if not provided
        data = get_filtered_data(limit=limit)
        if not data:
            return jsonify({"error": "No data retrieved from the database. Check your query or data availability."}), 500
        heatmap = generate_heatmap(data)
        return Response(heatmap, content_type="text/html")
    except Exception as e:
        return jsonify({"error": f"Failed to generate heatmap: {str(e)}"}), 500
