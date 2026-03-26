// Global variables for charts
let peakHoursChart = null;
let forecastChart = null;
let hotspotMap = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    fetchFleetInsights();
    // Refresh data every 5 minutes
    setInterval(fetchFleetInsights, 300000);
}

function fetchFleetInsights() {
    fetch('/api/fleet-insights')
        .then(response => response.json())
        .then(data => {
            updateKPIs(data);
            updatePeakHoursChart(data.peak_hours);
            updateForecastChart(data.forecast);
            updateHotspotMap(data.hotspots);
            updateOptimizationSuggestions(data.optimization);
        })
        .catch(error => console.error('Error fetching insights:', error));
}

function updateKPIs(data) {
    document.getElementById('totalRevenue').textContent = `$${data.optimization.total_revenue.toLocaleString()}`;
    document.getElementById('totalRides').textContent = data.optimization.total_rides.toLocaleString();
    document.getElementById('avgFare').textContent = `$${data.optimization.avg_fare.toFixed(2)}`;
    document.getElementById('peakHours').textContent = data.peak_hours.hours.map(h => `${h}:00`).join(', ');
}

function updatePeakHoursChart(peakData) {
    const trace = {
        x: peakData.hours.map(hour => `${hour}:00`),
        y: peakData.demand,
        type: 'bar',
        name: 'Ride Demand',
        marker: {
            color: 'rgb(59, 130, 246)'
        }
    };

    const layout = {
        margin: { t: 20, r: 20, l: 40, b: 40 },
        xaxis: { title: 'Hour of Day' },
        yaxis: { title: 'Number of Rides' }
    };

    Plotly.newPlot('peakHoursChart', [trace], layout);
}

function updateForecastChart(forecast) {
    const trace = {
        x: forecast.weeks.map(week => `Week ${week}`),
        y: forecast.predicted_demand,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Demand',
        line: {
            color: 'rgb(16, 185, 129)'
        }
    };

    const layout = {
        margin: { t: 20, r: 20, l: 40, b: 40 },
        xaxis: { title: 'Week' },
        yaxis: { title: 'Predicted Rides' }
    };

    Plotly.newPlot('forecastChart', [trace], layout);
}

function updateHotspotMap(hotspots) {
    if (!hotspotMap) {
        hotspotMap = L.map('hotspotMap').setView([hotspots[0].latitude, hotspots[0].longitude], 12);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(hotspotMap);
    } else {
        hotspotMap.eachLayer((layer) => {
            if (layer instanceof L.Circle) {
                hotspotMap.removeLayer(layer);
            }
        });
    }

    hotspots.forEach(spot => {
        L.circle([spot.latitude, spot.longitude], {
            color: 'rgb(239, 68, 68)',
            fillColor: '#ef4444',
            fillOpacity: 0.5,
            radius: Math.sqrt(spot.ride_count) * 50
        }).bindPopup(`
            <div class="text-sm">
                <p class="font-bold">Hotspot Details</p>
                <p>Total Rides: ${spot.ride_count}</p>
                <p>Average Fare: $${spot.avg_price.toFixed(2)}</p>
            </div>
        `).addTo(hotspotMap);
    });
}

function updateOptimizationSuggestions(optimization) {
    const container = document.getElementById('optimizationSuggestions');
    container.innerHTML = '';
    
    optimization.suggestions.forEach(suggestion => {
        const div = document.createElement('div');
        div.className = 'bg-blue-50 border-l-4 border-blue-500 p-4';
        div.innerHTML = `<p class="text-blue-700">${suggestion}</p>`;
        container.appendChild(div);
    });
}

function refreshData() {
    fetchFleetInsights();
}

function exportData() {
    fetch('/api/export-data')
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `fleet-analytics-${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        })
        .catch(error => console.error('Error exporting data:', error));
}