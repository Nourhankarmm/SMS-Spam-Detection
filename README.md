# SMS-Spam-Detection

## Project Description

This project implements a **SMS-Spam-Detection** system using machine learning techniques. The goal of the system is to classify emails as either **spam** or **ham** (not spam). The system is trained using a dataset from Kaggle that contains labeled emails.

I use various text preprocessing techniques such as tokenization, lemmatization, and vectorization to convert the text data into a format suitable for training machine learning models. I then train and evaluate a model using algorithms **Random Forest** 

## Dataset

The dataset used in this project is the **Spam SMS Dataset** from Kaggle, which contains labeled email and SMS messages classified as "spam" and "ham."

- **Dataset URL**: [Spam SMS Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)

The dataset consists of two main columns:
- `v1`: The label (either "ham" or "spam")
- `v2`: The content of the message

## Installation

To set up the project locally, follow these steps:

## Clone the repository

git clone https://github.com/Nourhankarmm/SMS-Spam-Detection/blob/main/SMS_Spam_Detection.ipynb

cd sms-mail-detection

## Usage
1. Data Preprocessing
The first step in the project is to preprocess the dataset. This includes cleaning the text data (removing special characters, single characters, etc.), tokenizing the text, and converting it to lowercase.

2. Feature Extraction
The text data is transformed into a Bag of Words representation using CountVectorizer from sklearn.

3. Model Training
In this project, we train a Random Forest Classifier on the preprocessed data. We also test Naive Bayes and other algorithms for performance comparison.

4. Running the Script
Once the project is set up, you can run the script to train the model and evaluate its performance.
python spam_detection.py

## dependencies
pandas: Data manipulation
scikit-learn: For machine learning algorithms and tools
matplotlib: For data visualization
seaborn: For statistical data visualization
nltk: Natural Language Toolkit for text processing
You can install all required dependencies using the requirements.txt:
pip install -r requirements.txt

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Kaggle for the dataset.
Scikit-learn for machine learning tools.
NLTK for text processing.
Matplotlib and Seaborn for data visualization.
