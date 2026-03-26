# DTSool Projects Repository

This repository contains various AI, Data Science, and Software Development projects created during work at DTSool. 
It consists of multiple sub-projects, each contained in its own directory with a dedicated environment and source code.

## Projects Overview

- **Areal Images**: A collection of aerial PNG images.
- **Assistive_Gym**: Reinforcement learning environments for assistive robotics.
- **BPM**: Video and image processing code for BPM tasks.
- **Example**: Web application with Flask and image orientation logic.
- **Inspre**: Python backend application containing `app.py` and `inspre.py`.
- **LLM-Chatbot**: A QA Chatbot utilizing Large Language Models.
- **Massachusetts Roads Dataset**: UNet model implementation (training, evaluation) for the Massachusetts Roads Dataset.
- **Pipeline_image**: Image processing pipeline with a trained UNet model.
- **Screening Test**: Contains screening task code and video recordings.
- **document_embedding_analysis**: Notebooks and scripts for embedding analysis of ArXiv papers, patents, and Wikipedia.
- **model_evaluation_project**: Infrastructure for evaluating machine learning models.
- **mongodb_dashboard**: A Flask-based interactive dashboard for MongoDB data.
- **movie_reviews_analysis**: Sentiment or text analysis on movie reviews.
- **task**, **task2**, **task3**, **task4_Postman**: Various scripting tasks involving CSV processing, SVG generation, and API templates.

## Setup

Many of these projects require Python. It's recommended to navigate to the specific project directory and install the requirements in a virtual environment.

```bash
cd <project_directory>
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

*Note: Environment variables and large binary files (like .h5, .keras, .mp4) are ignored in this repository.*
