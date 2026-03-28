from flask import Flask, render_template, redirect, url_for, request, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import folium
from folium.plugins import HeatMap

# Initialize Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = "your_secret_key_here"  # Replace with a secure key

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Flask-Login User class
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

# Dummy user data (static username/password for demo purposes)
DUMMY_USERS = {
    "admin": "password123"  # Username: admin, Password: password123
}

@login_manager.user_loader
def load_user(user_id):
    if user_id in DUMMY_USERS:
        return User(user_id)
    return None

# Global variables for dashboard data
key_metrics = {}
daily_revenue_plot = ""
hourly_distribution_plot = ""
heatmap_plot = ""

# Dummy data for 100 records
def fetch_and_process_data(limit=100):
    global key_metrics, daily_revenue_plot, hourly_distribution_plot, heatmap_plot

    # Simulated data for demonstration
    date_range = pd.date_range(datetime.now() - timedelta(days=30), periods=limit, freq='D')
    revenue_data = pd.DataFrame({
        'datetime': date_range,
        'price': [round(x * 1.5, 2) for x in range(1, limit + 1)],
        'hour': [x % 24 for x in range(1, limit + 1)]
    })

    # Key Metrics
    key_metrics.update({
        "total_rides": limit,
        "total_revenue": revenue_data['price'].sum(),
        "average_price": revenue_data['price'].mean(),
        "median_price": revenue_data['price'].median(),
    })

    # Daily Revenue Plot
    daily_revenue = revenue_data.groupby(revenue_data['datetime'].dt.date)['price'].sum().reset_index()
    daily_revenue.columns = ['date', 'revenue']
    fig1 = px.line(daily_revenue, x='date', y='revenue', title='Daily Revenue Trend')
    fig1.update_layout(xaxis_title="Date", yaxis_title="Revenue (€)")
    daily_revenue_plot = fig1.to_html(full_html=False)

    # Hourly Distribution Plot
    hourly_stats = revenue_data.groupby('hour').agg({'price': ['count', 'sum']}).reset_index()
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

    # Heatmap Plot (Dummy)
    heatmap = folium.Map(location=[50.1109, 8.6821], zoom_start=10)
    HeatMap([[50.1109 + i * 0.001, 8.6821 + i * 0.001] for i in range(limit)]).add_to(heatmap)
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
        username = request.form.get("username")
        password = request.form.get("password")
        if username in DUMMY_USERS and DUMMY_USERS[username] == password:
            login_user(User(username))
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    print("Fetching and processing data...")
    fetch_and_process_data(limit=100)
    app.run(debug=True)
