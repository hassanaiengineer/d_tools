import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

class FleetOptimizer:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.prepare_data()

    def prepare_data(self):
        # Extract time-based features
        self.df['hour'] = self.df['time'].dt.hour
        self.df['day_of_week'] = self.df['time'].dt.dayofweek
        self.df['week'] = self.df['time'].dt.isocalendar().week
        self.df['month'] = self.df['time'].dt.month

        # Group data by different time periods
        self.hourly_demand = self.df.groupby('hour')['price'].agg(['count', 'mean']).reset_index()
        self.weekly_demand = self.df.groupby('week')['price'].agg(['count', 'sum', 'mean']).reset_index()
        self.daily_demand = self.df.groupby('day_of_week')['price'].agg(['count', 'mean']).reset_index()

    def get_peak_hours(self):
        """Identify peak hours based on ride frequency and revenue"""
        peak_hours = self.hourly_demand.nlargest(5, 'count')
        return {
            'hours': peak_hours['hour'].tolist(),
            'demand': peak_hours['count'].tolist(),
            'avg_price': peak_hours['mean'].tolist()
        }

    def predict_weekly_demand(self, forecast_weeks=4):
        """Predict demand for next n weeks"""
        # Prepare features for weekly prediction
        X = self.weekly_demand[['week']]
        y = self.weekly_demand['count']

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Generate future weeks
        last_week = X['week'].max()
        future_weeks = np.array(range(last_week + 1, last_week + forecast_weeks + 1))
        future_weeks_df = pd.DataFrame(future_weeks, columns=['week'])

        # Make predictions
        predictions = model.predict(future_weeks_df)

        return {
            'weeks': future_weeks.tolist(),
            'predicted_demand': predictions.tolist()
        }

    def get_location_hotspots(self):
        """Identify high-demand areas"""
        # Group by location (rounded coordinates for clustering)
        self.df['lat_rounded'] = self.df['latitude'].round(3)
        self.df['lng_rounded'] = self.df['longitude'].round(3)
        
        hotspots = self.df.groupby(['lat_rounded', 'lng_rounded']).agg({
            'price': ['count', 'mean'],
            'time': 'count'
        }).reset_index()
        
        # Find top 10 hotspots
        top_hotspots = hotspots.nlargest(10, ('time', 'count'))
        
        return [{
            'latitude': row['lat_rounded'],
            'longitude': row['lng_rounded'],
            'ride_count': row[('time', 'count')],
            'avg_price': row[('price', 'mean')]
        } for _, row in top_hotspots.iterrows()]

    def get_revenue_optimization_suggestions(self):
        """Generate revenue optimization suggestions"""
        peak_hours = self.get_peak_hours()
        hotspots = self.get_location_hotspots()
        weekly_patterns = self.daily_demand.to_dict('records')

        return {
            'peak_hours': peak_hours,
            'hotspots': hotspots,
            'weekly_patterns': weekly_patterns,
            'suggestions': [
                f"Peak demand hours are between {peak_hours['hours'][0]}:00 and {peak_hours['hours'][-1]}:00",
                f"Highest average fares during {self.hourly_demand.nlargest(1, 'mean')['hour'].iloc[0]}:00",
                f"Found {len(hotspots)} major hotspots for ride concentration",
                f"Best performing day: {self.daily_demand.nlargest(1, 'mean')['day_of_week'].iloc[0]}"
            ]
        }