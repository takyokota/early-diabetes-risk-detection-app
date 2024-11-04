# Early Stage Diabetes Risk Prediction App

<!-- ABOUT THE PROJECT -->
## About The Project
This app predicts the risk of diabetes. On the application page, an actual app is shown. On the dashboard page, a dashboard shows the charts to explain about the dataset used for training a machine learning model and the prediction accuracy.

### Built With
For data preparation
- NumPy
- pandas
- scikit-learn (MinMaxScaler, train_test_split)
- imbalanced-learn (RandomOverSampler)

For a machine learning model
- scikit-learn (SVC)

For visualization
- matplotlib (pyplot)
- scikit-learn (confusion_matrix, ConfusionMatrixDisplay)
- seaborn

For GUI
- Streamlit
- pickle
 

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
Python 3

### Installation
To use this app, follow these steps:
- Clone this repository and unzip it.
- Create a virtual environment with Python 3 and activate it.
- Install the required packages using `pip install -r requirements.txt`
- Run the Streamlit: `streamlit run App/Home.py
- Open http://localhost:8501 in your browser if it does not open automatically

<!-- USAGE EXAMPLES -->
## Usage
- Home Page
![home](/screenshots/home.png) <br>

- Application Page
![application](/screenshots/application.png) <br>
The following shows how the app looks like. After answering all questions and then pressing the 'Submit' button, the app prompts the prediction of the risk of diabetes (positive or negative).
![app](/screenshots/app_sample1.png) <br>
This is how the app shows the prediction. <br>
![prediction](/screenshots/app_sample2.png) <br>

- Dashboard Page (Analysis for the dataset and the prediction accuracy)
![dashboard](/screenshots/dashboard.png) <br>
Histgram <br>
![histgram](/screenshots/histgram.png) <br>
Bar Chart <br>
![bar chart](/screenshots/bar_chart.png) <br>
Correlation Matrix Heatmap <br>
![correlation matrix](/screenshots/correlation_matrix_heatmap.png) <br>
Confusion Matrix <br>
![confusion matrix](/screenshots/confusion_matrix.png) <br>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [Dataset](https://doi.org/10.24432/C5VG8H)