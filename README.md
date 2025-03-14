# STAT5243_project_2
Data Cleaning and Preprocessing - Rain Shi's Work (2025/3/11)
Our Shiny for Python application provides an interactive data preprocessing tool that allows users to upload datasets, clean data, handle missing values, and apply advanced preprocessing techniques before analysis.

1. UI Design
File Upload: Supports .csv and .xlsx file imports.
Built-in Datasets: Users can choose from preloaded datasets (iris, wine, breast_cancer, diabetes) for testing.
Variable Selection: Users can select specific variables to retain, optimizing data processing.
Missing Value Handling: Provides multiple strategies to handle missing data.
Additional Cleaning Steps: Supports removing duplicates, standardizing data, normalizing data, and one-hot encoding.
Data Display:
Variable Types Table: Shows data structure before displaying the actual dataset.
Data Table: Displays the first 10 rows of the processed dataset.
2. Data Cleaning
Removing Invalid Columns: Deletes unnecessary index-based columns (e.g., "Unnamed").
Cleaning Text Data:
Uses BeautifulSoup to remove HTML tags from text-based fields.
Removes non-ASCII characters, filtering out corrupted text.
Filtering Corrupted Rows: Detects excessive encoding issues and removes rows with non-ASCII characters.
3. Variable Selection
Users can choose specific variables to retain before processing.
Ensures that only valid and selected columns remain in the dataset.
4. Missing Value Handling
Provides four missing value handling strategies:

Convert Common Missing Values to NA – Replaces placeholders like "", -9, -99 with NaN.
Listwise Deletion – Drops rows containing any missing values.
Mean Imputation – Fills missing values in numeric columns with the column mean.
Mode Imputation – Replaces missing values with the most frequent value in the column.
5. Additional Cleaning & Preprocessing Options
Users can apply real-time feature engineering through these options:

Remove Duplicates: Drops duplicate rows to eliminate redundancy.
Standardize Data: Uses StandardScaler to scale numeric data using Z-score normalization.
Normalize Data: Uses MinMaxScaler to scale values between 0 and 1.
One-Hot Encoding:
Encodes categorical variables into numerical format.
Uses pd.get_dummies() with drop_first=True to prevent multicollinearity.
6. Data Display
Variable Types Table: Shows each column's data type (int64, float64, object).
Data Table: Displays the first 10 rows of the cleaned and processed dataset.

Qiaoyang Lin's Work, Feature Engineering:

For uploading data, set the button, you can select the specified columns to do feature engineering, such as One-Hot, Normalize and Box-Cox, 
for the 'date' button you can change the format of the date data to generate the corresponding year, month and day, respectively; and generate the time to a specific date.

You can also generate the average of two columns or interaction terms for two columns to generate meaningful features with the 'extra_operation' button.

Nan Xiao's Work: Exploration Data Analysis (EDA)
1. Fully Interactive Data Exploration
File Upload & Dataset Selection: Users can upload CSV files
Dynamic Table View: Interactive data table with pagination
2. Filtering
Column-based Filtering: Users can apply numeric range filters or categorical selections to focus on specific subsets of the data
Sliders for Numeric Variables: Range sliders for numeric data
Checkboxes for Categorical Variables: Multi-select for categorical data
3. Custom Visualization Settings
Multiple Graph Types: Histogram, scatter plot, bar chart, correlation heatmap
Dynamic Axis Selection: Users can select which variables to be the x-axis and y-axis of the plot, the bin size of histogram, whether display a regression line
Grouping & Color Options: Color-coding data points by categorical variables
4. Dynamic Statistical Insights
Summary Statistics: Shows descriptive statistics for all numerical columns, including metrics like mean, median, skewness, correlation, and regression analysis.
Correlation Analysis: Correlation heatmap with detailed analysis

Yemin Wang's Work: User Interface(UI) and User Experience and Web Application
Construct a well-structured, feature-rich analytical tool that covers the complete data analysis workflow from loading and cleaning to visualization and exploration.
The application has a modern UI with cards, styling, and responsive layouts.
