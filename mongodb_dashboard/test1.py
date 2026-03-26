from pymongo import MongoClient
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import folium
from folium.plugins import HeatMap

class FleetAnalytics:
    def __init__(self):
        self.data = None
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize MongoDB connection
        self.client = MongoClient(
            "mongodb://rides_user:rides123456789@116.202.82.222:27017/FahrlyQPP?authSource=admin"
        )
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
                print("No data available for heatmap generation.")
                return None

            # Debugging: Print 'from' and 'to' fields
            print("\nInspecting 'from' and 'to' fields in the dataset:")
            for record in self.data[:5]:  # Print the first 5 records
                print(f"Record ID: {record.get('_id')}")
                print(f"From Field: {record.get('from')}")
                print(f"To Field: {record.get('to')}")
                print("\n")

            locations = []
            invalid_records = 0

            for record in self.data:
                try:
                    # Extract latitude and longitude
                    lat = None
                    lon = None

                    if record.get('from') and isinstance(record['from'], dict):
                        lat = record['from'].get('latitude') or record['from'].get('lat')
                        lon = record['from'].get('longitude') or record['from'].get('lon')
                    elif record.get('to') and isinstance(record['to'], dict):
                        lat = record['to'].get('latitude') or record['to'].get('lat')
                        lon = record['to'].get('longitude') or record['to'].get('lon')

                    if lat is not None and lon is not None:
                        locations.append([lat, lon])
                    else:
                        invalid_records += 1
                except KeyError as e:
                    print(f"KeyError for record ID {record.get('_id')}: {str(e)}")
                    invalid_records += 1
                    continue

            if not locations:
                print(f"No valid coordinates for heatmap. Invalid records: {invalid_records}")
                return False

            # Create heatmap
            print(f"Valid locations for heatmap: {len(locations)}")
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
    record_count = analytics.fetch_data(limit=1000)
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
