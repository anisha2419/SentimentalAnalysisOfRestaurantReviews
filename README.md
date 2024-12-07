# Sentiment Analysis of Restaurant Reviews

## Overview
This project performs sentiment analysis on a dataset of restaurant reviews to classify them as positive or negative. It employs various machine learning models, including K-Nearest Neighbors (KNN), Recurrent Neural Networks (RNN), and Support Vector Machines (SVM), to predict the sentiment of the reviews.

## Dataset
The dataset used in this project consists of 61,332 bytes of restaurant reviews sourced from online platforms such as Yelp and TripAdvisor. Each review is labeled as either positive (1) or negative (0), making it a binary classification task.

## Project Structure
The project is organized as follows:
- `data/`: Directory containing the dataset file
- `notebooks/`: Jupyter notebooks for data preprocessing, modeling, and evaluation
- `src/`: Python scripts for data loading, preprocessing, and model training
- `README.md`: Project overview and instructions

## Requirements
To run this project, you need the following libraries:
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Matplotlib
- TensorFlow (for RNN)

You can install the required libraries using pip:
```
pip install pandas numpy nltk scikit-learn matplotlib tensorflow
```

## Methodology
The project follows these main steps:
1. Data Collection: The dataset is obtained in a recognized format, and missing fields are removed.
2. Data Preprocessing:
   - Text cleaning and preprocessing
   - Splitting sentences and words from the text body
   - Creating a bag-of-words representation using a sparse matrix
3. Model Training and Evaluation:
   - Splitting the dataset into training and test sets
   - Fitting predictive models (KNN, RNN, SVM)
   - Evaluating model performance using accuracy, precision, recall, and F1 score
4. Results Analysis:
   - Comparing the performance of different models
   - Generating confusion matrices for each model
   - Analyzing the effectiveness of each model in capturing sentiment

## Results
The models achieved the following performance metrics:

| Model      | Accuracy | Precision | Recall | F1 Score |
|------------|----------|-----------|--------|----------|
| KNN        | 82%      | 83%       | 83%    | 83%      |
| RNN        | 79%      | 76%       | 81%    | 81%      |
| SVM        | 85%      | 86%       | 85%    | 85%      |
| SVM + PCA  | 85%      | 86%       | 85%    | 85%      |

The SVM model demonstrated the best overall performance, with high accuracy and balanced precision, recall, and F1 score. The KNN and RNN models also showed competitive results in sentiment classification.

## Usage
1. Ensure you have the required libraries installed.
2. Place the dataset file in the `data/` directory.
3. Run the Jupyter notebooks in the `notebooks/` directory to preprocess the data, train the models, and evaluate their performance.
4. Alternatively, you can run the Python scripts in the `src/` directory to perform the same tasks.

## Contributors
- Sayali Ambre
- Anisha Menezes

## Acknowledgments
We would like to thank Professor Jianmin Chen for his guidance and support throughout this project.
