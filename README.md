# Disaster_Tweet

**Exploratory Data Analysis (EDA):**

Checked the structure, shape, missing values, and duplicate records in the datasets (train_df and test_df).
Visualized the distribution of disaster and non-disaster tweets using a pie chart.

**Word Cloud Generation:**

Concatenated all the tweets for disaster and non-disaster categories separately.
Used the WordCloud library to visualize the most frequent words in disaster and non-disaster tweets.
Created side-by-side visualizations for easy comparison.

**Text Cleaning Script**

We are removing unwanted elements such as URLs, HTML tags, character references, non-printable characters, and numeric values. This preprocessing is essential for ensuring the text data is clean and ready for analysis or modeling.

**Features**

cleaning steps:

Remove URLs
Using regular expressions, remove all URLs (e.g., https://example.com) from the text.

Remove HTML Tags
Strips HTML tags (e.g., <div>, <br>) from the text.

Remove Character References
Removes HTML character references (e.g., &lt;, &amp;, &nbsp;) from the text.

Remove Non-Printable Characters
Removes characters not in the printable ASCII range (e.g., control characters).

Remove Numeric Values
Removes numeric values and mixtures of alphanumeric content (e.g., 1234, abc123).

**Dataframe Operations:**

The cleaning process is applied to train_df and test_df datasets, creating a new column text_cleaned that stores the cleaned version of the text column.
