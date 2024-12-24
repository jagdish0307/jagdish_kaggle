##  Disaster_Tweet

**Project Overview**
   This project focuses on analyzing and classifying tweets into two categories:

**Disaster-related tweets**: Tweets indicating a natural disaster or emergency.
**Non-disaster-related tweets**: Tweets unrelated to disasters.

** Load Dataset and Import Necessary Libraries **

**Exploratory Data Analysis (EDA):**

   Checked the structure, shape, missing values, and duplicate records in the datasets (train_df and test_df).
   Visualized the distribution of disaster and non-disaster tweets using a pie chart.

**Word Cloud Generation:**

   Concatenated all the tweets for disaster and non-disaster categories separately.
   Used the WordCloud library to visualize the most frequent words in disaster and non-disaster tweets.
   Created side-by-side visualizations for easy comparison.



**Text Cleaning Script**

   We are removing unwanted elements such as URLs, HTML tags, character references, non-printable characters, and numeric values. This preprocessing is essential    for ensuring the text data is clean and ready for analysis or modeling.

**Features**

**cleaning steps:**

**Remove URLs**
   Using regular expressions, remove all URLs from the text.

**Remove HTML Tags** 
   Strips HTML tags from the text.

**Remove Character References**
   Removes HTML character references from the text.

**Remove Non-Printable Characters**
   Removes characters not in the printable ASCII range.

**Remove Numeric Values**
   Removes numeric values and mixtures of alphanumeric content.

**Dataframe Operations:**

   The cleaning process is applied to train_df and test_df datasets, creating a new column text_cleaned that stores the cleaned version of the text column.

=========================================================



============================================================


**Text Preprocessing and Visualization**:


**Lemmatize the Text**:
   Apply lemmatization to standardize words to their root forms.

**Convert Text to Lowercase**:
   Transform all text to lowercase for uniformity.

**Remove Repeated Characters in Elongated Words**:
   Use a regular expression to reduce exaggerated repeated characters.

**Remove Mentions**:
   Strip user mentions (e.g., @username) using regular expressions.

**Remove Stopwords**:
   Exclude common stopwords to retain only meaningful words.

**Remove Punctuation**:
   Remove punctuation to simplify tokenization and text analysis.

**Generate Word Clouds**:
   Combine text for each category (raw and preprocessed).
   Create and display word clouds for both disaster and non-disaster tweets, before and after preprocessing.


**Feature Analysis and Visualization (Histograms)**:
   - Ensures that specific features exist in the DataFrame (`train_df`).
   - Compares the distribution of these features for disaster-related (`target = 1`) and non-disaster-related (`target = 0`) tweets using histograms.
   - Creates a figure with multiple subplots and labels for better visualization.

2. **Donut Plot Function**:
   - Defines a reusable function `donutplot` to create a donut-shaped pie chart with a central hole, useful for visualizing proportions.

3. **Barplot for Keyword Analysis**:
   - Groups tweets by keywords and their disaster classification.
   - Calculates the counts of tweets for each keyword in both disaster and non-disaster categories.
   - Uses a bar plot to show the top keywords based on their overall frequency in the dataset.

4. **Word Counter Function**:
   - Defines a helper function `word_counter` to count the occurrences of a specific word in the text column.

5. **Handling Missing Values**:
   - Identifies the number of missing and duplicated entries in `train_df`.
   - Removes the `location` column from the DataFrame and fill rows with missing values.

6. **Visualization of Keyword Importance**:
   - Creates a sorted DataFrame of keyword counts for both disaster and non-disaster tweets.
   - Visualizes the top 200 keywords with their counts using a horizontal bar plot.


**applying  Word2Vec and Machine Learning Models**
- This project implements a machine learning pipeline for text classification. The pipeline includes preprocessing, feature extraction using Word2Vec, data normalization, and training and evaluation of multiple classifiers.


**Text Preprocessing:**

- Tokenizes the cleaned text (text_cleaned) column into individual words.
Creates Word2Vec embeddings to represent text as numerical vectors.
Word2Vec Model:

**Trains a Word2Vec model using tokenized text**.
- Generates sentence embeddings by averaging the Word2Vec vectors of individual words.
Data Preparation:

**Splits the dataset into training and test sets**.
- Normalizes numerical features such as sent_count, word_count, char_count, etc., using StandardScaler.
Encodes categorical features, such as keyword, into numerical form.

- Combines Word2Vec embeddings with other normalized features for training machine learning models.
Model Training and Evaluation:

** Implements a variety of machine learning classifiers**:
 1. Naive Bayes (Multinomial and Gaussian)
 2 .Logistic Regression
 3. Support Vector Classifier (SVC)
 4. Decision Tree Classifier
 5. K-Nearest Neighbors (KNN)
 6. Random Forest Classifier
 7. Gradient Boosting Classifier
 8. XGBoost Classifier

**Comparison of Models**:

 Provides a summary of results for each model, including accuracy, precision, recall, and F1-score.
 Displays a classification report and confusion matrix for detailed insights.
 Performs hyperparameter tuning for KNN, RandomForest and evaluates its performance.
 Uses metrics such as accuracy, precision, recall, F1-score, and a confusion matrix for model evaluation.

**Summary Results**:


| Model                      | Accuracy  | Precision | Recall   | F1 Score |
|----------------------------|-----------|-----------|----------|----------|
| MultinomialNB              | 0.567301  | 0.578062  | 0.567301 | 0.569456 |
| SVC                        | 0.570584  | 0.325567  | 0.570584 | 0.414580 |
| LogisticRegression         | 0.672357  | 0.669011  | 0.672357 | 0.667771 |
| DecisionTreeClassifier     | 0.638214  | 0.638562  | 0.638214 | 0.638381 |
| KNeighborsClassifier       | 0.695338  | 0.694595  | 0.695338 | 0.694907 |
| RandomForestClassifier     | 0.710440  | 0.710385  | 0.710440 | 0.703399 |
| GradientBoostingClassifier | 0.717006  | 0.715241  | 0.717006 | 0.713362 |

**After Hyperparameter Tunning**:

| Model                      | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| KNeighborsClassifier       | 0.722    | 0.72      | 0.72   | 0.72     |
| RandomForestClassifier     | 0.726    | 0.73      | 0.73   | 0.72     |
| GradientBoostingClassifier | 0.733    | 0.73      | 0.73   | 0.73     |
| XGBClassifier              | 0.741    | 0.74      | 0.74   | 0.74     |

**Models Selection**:
   - Selects a best model for predict unseen data
   - Tree Based Models gives best accuracy for this dataset
   - Here we selects XGBClassifier to detect a particular Tweet is disaster or non disaster

** Check Model Performance On Unseen Data(test_df)**:
   - We have aleredy unseen data which is test_df using above RandomForestClassifier or XGBClassifier
   - Predicts test_df and create a submission and chek on kaggale it  gives **Accuracy between 72-74**
     
   






