# STAT5243_project_2
2025/3/8  Rain Shi's  work:
1. UI Design
File Upload: Supports .csv and .xlsx, allowing users to import local data files.
Variable Selection: Users can check the variables they want to retain, avoiding unnecessary data processing.
Missing Value Handling: Provides multiple strategies for handling missing values.
Data Display:
Variable Types Table (Shows the data structure first).
Data Table (Displays the first 10 rows of processed data).
2. Data Cleaning
Removing Invalid Columns: Deletes columns labeled Unnamed to eliminate empty or index-based columns.
Cleaning Text Data:
Uses BeautifulSoup to remove HTML tags (helpful for web-scraped content).
Keeps only ASCII characters, filtering out corrupted text.
Filtering Corrupted Rows: Detects non-ASCII characters and removes rows with excessive encoding issues.
3. Variable Selection
Users can choose specific variables to retain, optimizing processing efficiency.
Ensures that only valid and selected columns remain in the dataset.
4. Missing Value Handling
Supports four methods:

Convert Common Missing Values to NA – Replaces "", -9, -99 with NaN.
Listwise Deletion – Removes all rows containing missing values.
Mean Imputation – Fills missing values in numeric columns using the mean.
Mode Imputation – Replaces missing values using the most frequent value (mode).
5. Data Display
Variable Types Table: First, displays the data type of each column (int64, float64, object).
Data Table: Next, presents the first 10 rows of the cleaned and processed dataset for preview.
Summary
Removes invalid columns, HTML, and corrupted text.
Allows users to select specific variables for analysis.
Provides intelligent missing value handling with four strategies.
Enhances clarity by showing variable types before the dataset preview.


Qiaoyang Lin's Work, Feature Engineering:

For uploading data, set the button, you can select the specified columns to do feature engineering, such as One-Hot, Normalize and Box-Cox, 
for the 'date' button you can change the format of the date data to generate the corresponding year, month and day, respectively; and generate the time to a specific date.

You can also generate the average of two columns or interaction terms for two columns to generate meaningful features with the 'extra_operation' button.



