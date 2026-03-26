from flask import Flask, render_template, redirect, url_for, request, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import folium
from folium.plugins import HeatMap

# Initialize Flask
app = Flask(__name__)
load_dotenv()
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY")

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["FahrlyQPP"]
collection = db["rides"]
users_collection = db["users"]

# Flask-Login User class
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Global variables for dashboard data
key_metrics = {}
daily_revenue_plot = ""
hourly_distribution_plot = ""
heatmap_plot = ""

# Fetch and process data
def fetch_and_process_data(limit_for_heatmap=100):
    global key_metrics, daily_revenue_plot, hourly_distribution_plot, heatmap_plot

    time_threshold = int((datetime.now() - timedelta(days=180)).timestamp())
    query = {"datestamp": {"$gte": time_threshold}, "accepted": True}
    projection = {
        "date": 1,
        "datestamp": 1,
        "price": 1,
        "from": 1,
        "to": 1,
        "city": 1,
        "driver": 1,
        "system": 1
    }

    # Fetch all records for daily revenue and hourly distribution
    full_data = []
    try:
        full_data_cursor = collection.find(query, projection)
        for record in full_data_cursor:
            try:
                full_data.append(record)
            except Exception as e:
                print(f"Skipping invalid record: {e}")
    except Exception as e:
        print(f"Error fetching data: {e}")

    # Fetch limited records for heatmap
    limited_data = []
    try:
        limited_cursor = collection.find(query, projection).limit(limit_for_heatmap)
        for record in limited_cursor:
            try:
                limited_data.append(record)
            except Exception as e:
                print(f"Skipping invalid record in heatmap: {e}")
    except Exception as e:
        print(f"Error fetching limited data for heatmap: {e}")

    # Convert full data to DataFrame
    if full_data:
        df = pd.DataFrame(full_data)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['datetime'] = pd.to_datetime(df['datestamp'], unit='s')

        # Calculate key metrics
        key_metrics.update({
            "total_rides": len(df),
            "total_revenue": round(df['price'].sum(), 2),
            "average_price": round(df['price'].mean(), 2),
            "median_price": round(df['price'].median(), 2),
        })

        # Daily Revenue Plot
        daily_revenue = df.groupby(df['datetime'].dt.date)['price'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        fig1 = px.line(daily_revenue, x='date', y='revenue', title='Daily Revenue Trend')
        fig1.update_layout(xaxis_title="Date", yaxis_title="Revenue (€)")
        daily_revenue_plot = fig1.to_html(full_html=False)

        # Hourly Distribution Plot
        df['hour'] = df['datetime'].dt.hour
        hourly_stats = df.groupby('hour').agg({'price': ['count', 'sum']}).reset_index()
        hourly_stats.columns = ['hour', 'rides', 'revenue']
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(
            go.Bar(x=hourly_stats['hour'], y=hourly_stats['rides'], name="Number of Rides"),
            secondary_y=False
        )
        fig2.add_trace(
            go.Scatter(x=hourly_stats['hour'], y=hourly_stats['revenue'], mode='lines', name="Revenue (€)"),
            secondary_y=True
        )
        fig2.update_layout(title="Hourly Distribution of Rides and Revenue")
        fig2.update_xaxes(title_text="Hour of Day")
        fig2.update_yaxes(title_text="Number of Rides", secondary_y=False)
        fig2.update_yaxes(title_text="Revenue (€)", secondary_y=True)
        hourly_distribution_plot = fig2.to_html(full_html=False)

    # Heatmap (limited records)
    locations = []
    for record in limited_data:
        if record.get('from') and record.get('to'):
            # Replace with actual geocoding logic (Google Maps API, etc.)
            locations.append([50.1109, 8.6821])  # Dummy coordinates (Frankfurt)
    heatmap = folium.Map(location=[50.1109, 8.6821], zoom_start=10)
    HeatMap(locations).add_to(heatmap)
    heatmap_plot = heatmap._repr_html_()


# Flask routes
@app.route("/")
@login_required
def home():
    return render_template("home.html", key_metrics=key_metrics)

@app.route("/daily-revenue")
@login_required
def daily_revenue():
    return render_template("visualization.html", title="Daily Revenue", plot=daily_revenue_plot)

@app.route("/hourly-distribution")
@login_required
def hourly_distribution():
    return render_template("visualization.html", title="Hourly Distribution", plot=hourly_distribution_plot)

@app.route("/heatmap")
@login_required
def heatmap():
    return render_template("visualization.html", title="Heatmap", plot=heatmap_plot)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            login_user(User(user["_id"]))
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if users_collection.find_one({"email": email}):
            flash("Email already registered", "danger")
            return redirect(url_for("register"))
        hashed_password = generate_password_hash(password, method='sha256')
        users_collection.insert_one({"email": email, "password": hashed_password})
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    print("Fetching and processing data...")
    fetch_and_process_data(limit_for_heatmap=100)
    app.run(debug=True)
