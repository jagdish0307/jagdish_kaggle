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



**Text Feature Engineering** 
The goal is to derive useful features that can be used in predictive models for various natural language processing (NLP) tasks. 
This script includes different techniques to extract text-based features, such as sentence count, word count, character count, and more.

**Dependencies**
     .nltk
     .spacy
     .pandas

**Features Extracted**
      **Steps**
     **Number of Sentences (sent_count)**
         Tokenizes each text into sentences and counts them.
                [nltk.tokenize.sent_tokenize]
                
   **Number of Words (word_count) **
        Tokenizes each text into words and counts them.
        
  ** Number of Characters (excluding whitespaces) (char_count)**
        Counts the number of characters excluding spaces.
        
  **Number of Hashtags (hash_count)**
       Identifies the number of hashtags (#).
       
  **Number of Mentions (ment_count)**
       Identifies the number of mentions (@)
       
  **Number of Uppercase Words (all_caps_count)**
      Identifies words with consecutive uppercase letters.
      
  **Average Word Length (avg_word_len)**
      Computes the average length of the words in each text.
      
 **Number of Proper Nouns using NLTK (propn_count_nltk)**
       Uses NLTK's POS tagging to count proper nouns
       
  **Number of Proper Nouns using spaCy (propn_count)**
       Uses spaCy to count proper nouns
      
  **Number of Non-Proper Nouns (noun_count)**
       Uses spaCy to count non-proper nouns.
      
  ** Percentage of Punctuation (punc_per)**
      Computes the percentage of punctuation marks in the text.
          



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


| Model                                                       | Accuracy | Precision | Recall  | F1 Score |
|-------------------------------------------------------------|----------|-----------|---------|----------|
| **Bernoulli Naive Bayes (BernoulliNB)**                     | 0.7236   | 0.7256    | 0.7236  | 0.7243   |
| **Support Vector Classifier (SVC)**                         | 0.7643   | 0.7660    | 0.7643  | 0.7598   |
| **Logistic Regression (LogisticRegression)**                | 0.7669   | 0.7662    | 0.7669  | 0.7664   |
| **Decision Tree Classifier (DecisionTreeClassifier)**       | 0.6448   | 0.6455    | 0.6448  | 0.6451   |
| **K-Nearest Neighbors (KNeighborsClassifier)**              | 0.6894   | 0.6874    | 0.6894  | 0.6877   |
| **Random Forest Classifier (RandomForestClassifier)**       | 0.7203   | 0.7230    | 0.7203  | 0.7117   |
| **Gradient Boosting Classifier(GradientBoostingClassifier)**| 0.7223   | 0.7212    | 0.7223  | 0.7177   |
| **XGBClassifier**                                           |  0.7570  | 0.76      |  0.76   | 0.76     |

**After Hyperparameter Tunning**:

| Model                      | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| **Logistic Regression**   |  0.7682  | 0.77      | 0.77   | 0.77     |

- Here **Logistic Regression** gives best accuracy which is on training data **0.7958** and on testing **F1 Score is 0.77**

 **Interpretation of Metrics**:

- **Accuracy**: The proportion of correct predictions made by the model out of all predictions.
- **Precision**: The proportion of positive predictions that were correct (i.e., the model’s ability to avoid false positives).
- **Recall**: The proportion of actual positives that were correctly predicted by the model (i.e., the model’s ability to avoid false negatives).
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.
- **for this perticular problem recall is most important** 

**Models Selection**:
   - Selects a best model for predict unseen data
   - **Logistic Regression and SVC** Models gives best accuracy for this dataset
   - Here we selects **Logistic Regression**  to detect a particular Tweet is disaster or non disaster

**Check Model Performance On Unseen Data(test_df)**:
   - We have aleredy unseen data which is test_df using above **Logistic Regression**
   - Predicts test_df and create a submission and chek on kaggale it  gives **Accuracy 0.7560**
     
   






