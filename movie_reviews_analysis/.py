from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Specify the path to your Python file
file_path = '/content/drive/My Drive/movie_reviews_analysis.py'  # Update this path to your actual file

# Change the directory to the location of the script
os.chdir('/content/drive/My Drive/')  # Update this to the folder containing your script

