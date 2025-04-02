# Movie Gross Predictor

This Flask application predicts the gross revenue of movies based on their features. It uses a machine learning model trained on a dataset of movies to make predictions. The application also provides a plot of predicted versus actual values.

## Features

- **Predict Movie Gross**: Submit movie details to predict its gross revenue.
- **Plot Predicted vs Actual**: Visualize the predicted and actual gross values for a given movie.
- **Fetch Random Data Point**: Retrieve a random movie data point from the dataset.

## Requirements

- Python 3.x
- Flask
- Flask-CORS
- Pandas
- Scikit-Learn
- Matplotlib
- Numpy
- Joblib

## Installation

1. Download the Folder and go to "movie-gross-predictor-be" in the terminal or command prompt

2.Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required packages using pip:

    ```shell
    pip install -r requirements.txt
    ```


## Usage

1. Start the Flask application:

    ```Flask run
    ```

2. The application will run on `http://localhost:8000` by default.

## API Endpoints

The following API endpoints are available:

- **POST /api/predict**: Predict the gross revenue for a given movie.

    - Request: JSON object containing movie features (`year`, `score`, `votes`, `runtime`, `rating`, `genre`, `director`, `writer`, `star`, `country`, `company`).
    - Response: JSON object containing the predicted gross revenue.

- **POST /api/plot-predicted-vs-actual**: Plot the predicted vs actual gross values.

    - Request: Form data containing movie features.
    - Response: Image of the plot.

- **GET /api/random-data-point**: Fetch a random movie data point.

    - Response: JSON object containing the data point.

## Data Preparation

Ensure that the dataset file `cleaned_movies.xlsx` is available in the project's root directory. The file should contain the required columns for the model features.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request with any improvements, bug fixes, or features.

## License

This project is licensed under the [MIT License](LICENSE).
