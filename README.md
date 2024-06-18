# SMS Spam Classification Project

## Project Overview

This project aims to develop a text classification model to classify SMS messages as either spam or non-spam using data science techniques in Python. The project involves data preprocessing, exploratory data analysis (EDA), feature extraction, and training a machine learning model for text classification.

## Table of Contents

1. [Project Description](#project-description)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Code Explanation](#code-explanation)
   - [Data Preprocessing](#data-preprocessing)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Feature Extraction](#feature-extraction)
   - [Model Training](#model-training)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Project Description

The project involves the following key steps:

1. **Data Preprocessing**: Cleaning and preparing the data for analysis.
2. **Exploratory Data Analysis (EDA)**: Visualizing the data to understand patterns and relationships.
3. **Feature Extraction**: Converting text data into numerical features.
4. **Model Training**: Building and training a predictive model to classify SMS messages.

### Features

- Data cleaning and preprocessing
- Visualization of spam and non-spam messages
- Feature extraction from text data
- Training and evaluating a text classification model

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Seaborn**: Data visualization
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural Language Toolkit for text processing

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Jupyter Notebook or any Python IDE

### Installation Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/username/sms-spam-classification.git
   cd sms-spam-classification
   ```

2. **Install Dependencies**

   ```bash
   pip install pandas matplotlib seaborn scikit-learn nltk
   ```

3. **Download the Dataset**

   Place the SMS dataset in the project directory.

## Usage

1. **Open the Jupyter Notebook**

   ```bash
   jupyter notebook sms_spam_classification.ipynb
   ```

2. **Run the Notebook**

   Execute the cells in the notebook to preprocess the data, perform EDA, extract features, and train the model.

## Project Structure

```
sms-spam-classification/
├── data/
│   └── spam.csv
├── sms classified.ipynb
├── README.md
```

## Code Explanation

### Data Preprocessing

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('data/spam.csv', encoding='latin-1')

# Drop irrelevant columns
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Rename columns for easier access
data.columns = ['label', 'message']

# Map labels to binary values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Check for missing values
data.isnull().sum()
```

### Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Bar graph for label distribution
sns.countplot(x='label', data=data)
plt.title('Distribution of Spam and Non-Spam Messages')
plt.show()

# Heatmap for correlation
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Word count distribution
data['word_count'] = data['message'].apply(lambda x: len(x.split()))
sns.histplot(data=data, x='word_count', hue='label', element='step', stat='density')
plt.title('Word Count Distribution by Label')
plt.show()
```

### Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Define stop words
stop_words = stopwords.words('english')

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words=stop_words, max_features=3000)

# Transform messages into TF-IDF features
X = tfidf.fit_transform(data['message']).toarray()
y = data['label']
```

### Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

-**Saumya Poojari** - [saumya.poojarii7@gmail.com]

-LinkedIn - https://www.linkedin.com/in/ssaumz/

Feel free to reach out with any questions or feedback!

---

Thank you for your interest in my SMS Spam Classification project. I hope you find it informative and useful!
