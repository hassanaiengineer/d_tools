import plotly.graph_objects as go
import folium
from datetime import datetime

def calculate_key_metrics(data):
    try:
        total_rides = len(data)
        total_revenue = sum(ride["price"] for ride in data)
        average_price = total_revenue / total_rides if total_rides > 0 else 0

        high_demand_zones = []
        for ride in data:
            high_demand_zones.append({
                "latitude": ride["latitude"],
                "longitude": ride["longitude"],
                "count": 1  # Example count, modify based on your aggregation logic
            })

        return {
            "total_rides": total_rides,
            "average_price": average_price,
            "high_demand_zones": high_demand_zones[:5]  # Limit to top 5
        }
    except Exception as e:
        print(f"Error calculating key metrics: {str(e)}")
        return {}

def generate_weekly_trends(data):
    try:
        # Group data by week and calculate revenue
        weekly_data = {}
        for ride in data:
            week = datetime.utcfromtimestamp(ride["datestamp"]).isocalendar()[1]
            weekly_data[week] = weekly_data.get(week, 0) + ride["price"]

        x = list(weekly_data.keys())
        y = list(weekly_data.values())

        # Create Plotly chart
        fig = go.Figure(data=[go.Bar(x=x, y=y)])
        fig.update_layout(title="Weekly Revenue Trends", xaxis_title="Week", yaxis_title="Revenue")
        return fig.to_dict()
    except Exception as e:
        print(f"Error generating weekly trends: {str(e)}")
        return {}

def generate_heatmap(data):
    try:
        # Create heatmap using Folium
        map_center = [data[0]["latitude"], data[0]["longitude"]] if data else [0, 0]
        m = folium.Map(location=map_center, zoom_start=12)

        for ride in data:
            folium.CircleMarker(
                [ride["latitude"], ride["longitude"]],
                radius=5,
                weight=2,
                popup=f"Price: {ride['price']}"
            ).add_to(m)

        return m._repr_html_()
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return "<h1>Error generating heatmap</h1>"
