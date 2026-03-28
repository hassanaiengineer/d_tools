from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from bson import json_util
import requests
import folium
from folium.plugins import HeatMap

class FleetAnalytics:
    def __init__(self):
        self.data = None
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load environment variables
        load_dotenv()
        self.mongo_uri = os.getenv("MONGO_URI")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Initialize MongoDB connection
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client["FahrlyQPP"]
        self.collection = self.db["rides"]

    def fetch_data(self, limit=None):
        """Fetch last 6 months of ride data."""
        try:
            time_threshold = int((datetime.now() - timedelta(days=180)).timestamp())
            query = {
                "datestamp": {"$gte": time_threshold},
                "accepted": True
            }
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
            cursor = self.collection.find(query, projection)
            if limit:
                cursor = cursor.limit(limit)

            self.data = list(cursor)
            print(f"Successfully processed {len(self.data)} records")
            return len(self.data)
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return 0

    def get_coordinates(self, address):
        """Fetch coordinates using Google Maps Geocoding API."""
        try:
            # Append a default city/country context to vague addresses
            address_with_context = f"{address}, Germany"  # Example context
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address_with_context}&key={self.google_api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "OK":
                    location = data["results"][0]["geometry"]["location"]
                    return location["lat"], location["lng"]
                elif data["status"] == "ZERO_RESULTS":
                    print(f"Failed to get coordinates for address '{address}': ZERO_RESULTS")
                else:
                    print(f"Failed to get coordinates for address '{address}': {data['status']}")
            else:
                print(f"Failed to connect to API for address '{address}'. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error fetching coordinates: {str(e)}")
        return None, None


    def calculate_key_metrics(self):
        """Calculate key business metrics."""
        try:
            if not self.data:
                return None

            df = pd.DataFrame(self.data)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['datetime'] = pd.to_datetime(df['datestamp'], unit='s')

            metrics = {
                "total_rides": len(df),
                "total_revenue": df['price'].sum(),
                "average_price": df['price'].mean(),
                "median_price": df['price'].median()
            }
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return None

    def generate_visualizations(self):
        """Generate and save visualizations."""
        try:
            if not self.data:
                return None

            df = pd.DataFrame(self.data)
            df['datetime'] = pd.to_datetime(df['datestamp'], unit='s')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

            # Daily Revenue Trend
            daily_revenue = df.groupby(df['datetime'].dt.date)['price'].sum().reset_index()
            daily_revenue.columns = ['date', 'revenue']
            fig1 = px.line(daily_revenue, x='date', y='revenue', title='Daily Revenue Trend')
            fig1.update_layout(xaxis_title="Date", yaxis_title="Revenue (€)")
            fig1.write_html(f"{self.output_dir}/daily_revenue.html")

            # Hourly Distribution
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
            fig2.write_html(f"{self.output_dir}/hourly_distribution.html")

            print("Visualizations saved.")
            return True
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            return False

    def generate_heatmap(self):
        """Generate a heatmap for ride locations."""
        try:
            if not self.data:
                return None

            locations = []
            invalid_records = 0

            for record in self.data:
                try:
                    from_address = record.get("from", "")
                    if from_address:
                        lat, lng = self.get_coordinates(from_address)
                        if lat is not None and lng is not None:
                            locations.append([lat, lng])
                        else:
                            invalid_records += 1
                except Exception as e:
                    print(f"Error processing record: {str(e)}")
                    invalid_records += 1

            if not locations:
                print(f"No valid coordinates for heatmap. Invalid records: {invalid_records}")
                return False

            # Create heatmap
            heatmap = folium.Map(location=[50.1109, 8.6821], zoom_start=10)
            HeatMap(locations).add_to(heatmap)
            heatmap.save(f"{self.output_dir}/heatmap.html")
            print("Heatmap saved.")
            return True
        except Exception as e:
            print(f"Error generating heatmap: {str(e)}")
            return False

def main():
    analytics = FleetAnalytics()

    print("Fetching data...")
    record_count = analytics.fetch_data(limit=100)
    if record_count == 0:
        print("No data retrieved. Exiting.")
        return

    print("\nCalculating metrics...")
    metrics = analytics.calculate_key_metrics()
    if metrics:
        print("\nKey Performance Indicators:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("\nGenerating visualizations...")
    analytics.generate_visualizations()

    print("\nGenerating heatmap...")
    analytics.generate_heatmap()

if __name__ == "__main__":
    main()
