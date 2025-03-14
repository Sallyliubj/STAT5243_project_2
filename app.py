from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from bs4 import BeautifulSoup
import re
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import io
import base64
import warnings
import pyreadr
import openpyxl  
warnings.filterwarnings('ignore')


# Define CSS for the modern UI
app_css = """
body {
    font-family: Arial, sans-serif;
    background-color: #f5f7fa;
    color: #333;
}

.btn-primary {
    background-color: #4a6fa5;
    color: white;
}

.checkbox-group {
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 8px;
    margin-bottom: 15px;
}

.card {
    margin-bottom: 20px;
    border: 1px solid #dee2e6;
    border-radius: 5px;
}

.card-header {
    background-color: #4a6fa5;
    color: white;
    padding: 10px 15px;
    font-weight: bold;
}

.card-body {
    padding: 15px;
}

.feature-result {
    background-color: #f0f7ff;
    border-left: 4px solid #4a6fa5;
    padding: 10px;
    font-family: monospace;
}

h3.section-title {
    color: #395682;
    border-bottom: 2px solid #4a6fa5;
    padding-bottom: 8px;
    margin-bottom: 15px;
}

.sidebar {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 5px;
    border: 1px solid #dee2e6;
}

.main-panel {
    padding: 15px;
}
"""

# Built-in dataset loader
def get_builtin_dataset(name):
    """Load built-in dataset"""
    try:
        if name == "iris":
            data = load_iris(as_frame=True)
            df = pd.DataFrame(data.data, columns=data.feature_names)
        elif name == "wine":
            data = load_wine(as_frame=True)
            df = pd.DataFrame(data.data, columns=data.feature_names)
        elif name == "breast_cancer":
            data = load_breast_cancer(as_frame=True)
            df = pd.DataFrame(data.data, columns=data.feature_names)
        elif name == "diabetes":
            data = load_diabetes(as_frame=True)
            df = pd.DataFrame(data.data, columns=data.feature_names)
        else:
            print(f"⚠️ Unknown dataset: {name}")
            return None

        df["target"] = data.target
        print(f"✓ Successfully loaded {name} dataset\n")
        print(f"✓ Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ Error loading {name} dataset: {str(e)}")
        return None

app_ui = ui.page_fluid(
    ui.tags.style(app_css),
    
    ui.tags.h1("Data Analysis", 
              style="color: #4a6fa5; text-align: center; margin: 20px 0;"),
    
    # Tab selector
    ui.tags.div(
        ui.input_radio_buttons(
            "active_tab", 
            "Select Tab:", 
            {"data_cleaning": "Data Cleaning", "feature_engineering": "Feature Engineering", "visualization": "Visualization", "eda": "EDA"},
            selected="data_cleaning",
            inline=True
        ),
        style="text-align: center; margin-bottom: 20px;"
    ),
    
    # Data Cleaning UI
    ui.panel_conditional(
        "input.active_tab === 'data_cleaning'",
        ui.row(
            # Sidebar
            ui.column(4,
                ui.tags.div(
                    ui.tags.h3("Data Loading and Cleaning", class_="section-title"),
                    ui.input_file("file1", "Select Data File", 
                               accept=[".csv", ".xlsx", ".xls", ".json", ".rds"]),
                    ui.input_select("builtinDataset", "Choose Built-in Dataset",
                                  choices=["None", "iris", "wine", "breast_cancer", "diabetes"], 
                                  selected="None"),
                    ui.tags.h4("Select Variables to Keep:"),
                    ui.tags.div(
                        ui.input_checkbox_group("varSelect", "", choices=[]),
                        class_="checkbox-group"
                    ),
                    ui.tags.div(
                        ui.input_action_button("selectAll", "Select All"),
                        ui.input_action_button("deselectAll", "Deselect All"),
                        style="display: flex; gap: 10px; margin-bottom: 15px;"
                    ),
                    ui.input_select("missingDataOption", "Handle Missing Values:",
                                  choices=["None", "Convert Common Missing Values to NA", 
                                         "Listwise Deletion", "Mean Imputation", "Mode Imputation"], 
                                  selected="None"),
                    ui.tags.h4("Additional Cleaning Steps:"),
                    ui.tags.div(
                        ui.input_checkbox_group("dataProcessingOptions", "",
                                           choices=["Remove Duplicates", "Standardize Data", 
                                                  "Normalize Data", "One-Hot Encoding"]),
                        class_="checkbox-group"
                    ),
                    ui.tags.div(
                        ui.input_action_button("processData", "Clean Data", 
                                             class_="btn-primary"),
                        ui.input_action_button("revertCleaningChange", "Revert Last Change",
                                             class_="btn-warning",
                                             style="margin-left: 10px;"),
                        style="margin-bottom: 15px;"
                    ),
                    class_="sidebar"
                )
            ),
            # Main content
            ui.column(8,
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Data Summary"),
                        ui.tags.div(
                            ui.tags.pre(ui.output_text("dataSummary")),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Data Types and Missing Values"),
                        ui.tags.div(
                            ui.output_table("dataTypesTable"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Numerical Columns Summary"),
                        ui.tags.div(
                            ui.output_table("numericalSummary"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Sample Data Preview"),
                        ui.tags.div(
                            ui.output_table("dataTable"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    class_="main-panel"
                )
            )
        )
    ),
    
    # Feature Engineering UI
    ui.panel_conditional(
        "input.active_tab === 'feature_engineering'",
        ui.row(
            # Sidebar
            ui.column(4,
                ui.tags.div(
                    ui.tags.h3("Feature Engineering Operations", class_="section-title"),
                    ui.tags.div(
                        ui.input_select("featureColumn", "Select Column for Feature Engineering", 
                                     choices=[]),
                        ui.input_select("featureOperation", "Select Operation", 
                                     choices=["Normalize", "One-Hot", "Convert Date Format", "Box-Cox"]),
                        ui.panel_conditional(
                            "input.featureOperation === 'Normalize'",
                            ui.p("Scales the selected column to range [0,1]. Useful for features with different scales.", 
                                 style="color: #666; font-style: italic; margin: 10px 0;"),
                            ui.p("Note: Only works with numerical columns.", 
                                 style="color: red; font-style: italic;"),
                        ),
                        ui.panel_conditional(
                            "input.featureOperation === 'One-Hot'",
                            ui.p("Creates binary columns for each unique value in the selected categorical column.", 
                                 style="color: #666; font-style: italic; margin: 10px 0;"),
                            ui.p("Best used for categorical columns with limited unique values.", 
                                 style="color: red; font-style: italic;"),
                        ),
                        ui.panel_conditional(
                            "input.featureOperation === 'Convert Date Format'",
                            ui.p("Specify a reference date to calculate the number of days between your date column and this reference date.", 
                                 style="color: #666; font-style: italic; margin: 10px 0;"),
                            ui.p("Please make sure the selected column is a date column.", 
                                style="color: red; font-style: italic; margin: 10px 0;"),
                            ui.input_numeric("input_year", "Input year (YYYY)", value=2025),
                            ui.input_numeric("input_month", "Input month (MM)", value=3),
                            ui.input_numeric("input_day", "Input day (DD)", value=7),
                        ),
                        ui.panel_conditional(
                            "input.featureOperation === 'Box-Cox'",
                            ui.p("Transforms data to be more normally distributed. Useful for skewed numerical data.", 
                                 style="color: #666; font-style: italic; margin: 10px 0;"),
                            ui.p("Note: Only works with positive numerical values. Negative values will be shifted.", 
                                 style="color: red; font-style: italic;"),
                        ),
                        style="background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px;"
                    ),
                    ui.tags.hr(),
                    ui.tags.h3("Add New Features from Multiple Columns", class_="section-title"),
                    ui.tags.div(
                        ui.input_selectize("multiColumns", "Select Columns for New Feature", 
                                        choices=[], multiple=True),
                        ui.input_select("extraOperation", "Create New Feature", 
                                     choices=["None", "Average", "Interactions"]),
                        ui.panel_conditional(
                            "input.extraOperation === 'Average'",
                            ui.p("Creates a new column with the average value of selected columns. Useful for combining related features.", 
                                 style="color: #666; font-style: italic; margin: 10px 0;"),
                            ui.p("Note: Select at least 2 numerical columns. Non-numerical columns will be ignored.", 
                                 style="color: red; font-style: italic;"),
                        ),
                        ui.panel_conditional(
                            "input.extraOperation === 'Interactions'",
                            ui.p("Creates a new column by multiplying two selected columns. Useful for capturing feature relationships.", 
                                 style="color: #666; font-style: italic; margin: 10px 0;"),
                            ui.p("Note: Select exactly 2 numerical columns. Only the first two selected columns will be used.", 
                                 style="color: red; font-style: italic;"),
                        ),
                        style="background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px;"
                    ),
                    ui.input_action_button("applyFeatureEng", "Apply Feature Engineering",
                                         class_="btn-primary"),
                    ui.input_action_button("revertChange", "Revert Last Change",
                                         class_="btn-warning",
                                         style="margin-left: 10px;"),
                    ui.tags.hr(),
                    ui.download_button(
                        id="downloadData",
                        label="Download Processed Data",
                        class_="btn-primary"
                    ),
                    class_="sidebar"
                )
            ),
            # Main content
            ui.column(8,
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Feature Engineering Results"),
                        ui.tags.div(
                            ui.tags.pre(ui.output_text("featureStatus"), 
                                     class_="feature-result"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Updated Data Preview"),
                        ui.tags.div(
                            ui.output_table("featureDataTable"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    class_="main-panel"
                )
            )
        )
    ),
    
    # Visualization UI
    ui.panel_conditional(
        "input.active_tab === 'visualization'",
        ui.row(
            # Sidebar
            ui.column(4,
                ui.tags.div(
                    ui.tags.h3("Visualization Controls", class_="section-title"),
                    ui.input_date_range(
                        "date_range", 
                        "Select Date Range",
                        start=datetime(2020, 1, 1),
                        end=datetime.now(),
                        format="yyyy-mm-dd"
                    ),
                    ui.input_select(
                        "plot_type", 
                        "Plot Type",
                        choices={
                            "line": "Line Chart", 
                            "bar": "Bar Chart",
                            "scatter": "Scatter Plot",
                            "histogram": "Histogram"
                        },
                        selected="line"
                    ),
                    ui.tags.div(
                        ui.input_select(
                            "x_var", 
                            "X-axis Variable",
                            choices=[]
                        ),
                        style="margin-bottom: 15px;"
                    ),
                    ui.tags.div(
                        ui.input_select(
                            "y_var", 
                            "Y-axis Variable",
                            choices=[]
                        ),
                        style="margin-bottom: 15px;"
                    ),
                    ui.input_action_button("update_plot", "Update Plot"),
                    ui.tags.hr(),
                    ui.tags.h3("Summary Statistics", class_="section-title"),
                    ui.input_select(
                        "summary_var", 
                        "Select Variable for Summary",
                        choices=[]
                    ),
                    class_="sidebar"
                )
            ),
            # Main content
            ui.column(8,
                ui.tags.div(
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Data Visualization"),
                        ui.tags.div(
                            ui.output_ui("main_plot"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Summary Statistics"),
                        ui.tags.div(
                            ui.output_table("summary_stats"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Data Distribution"),
                        ui.tags.div(
                            ui.output_ui("distribution_plot"),
                            class_="card-body"
                        ),
                        class_="card"
                    ),
                    class_="main-panel"
                )
            )
        )
    ),
    
    # EDA UI
    ui.panel_conditional(
        "input.active_tab === 'eda'",
        ui.row(
            ui.column(12,
                ui.tags.div(
                    ui.tags.h3("Exploratory Data Analysis", class_="section-title"),
                    
                    # Visualization Settings
                    ui.tags.div(
                        ui.tags.h4("Visualization Settings", style="margin-top: 20px;"),
                        ui.input_select("eda_plot_type", "Select Plot Type:", {
                            "histogram": "Histogram",
                            "scatter": "Scatter Plot",
                            "bar": "Bar Chart",
                            "heatmap": "Correlation Heatmap"
                        }),
                        ui.output_ui("eda_plot_controls"),
                        style="background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px;"
                    ),
                    
                    # Data Filters
                    ui.tags.div(
                        ui.tags.h4("Data Filters", style="margin-top: 20px;"),
                        ui.output_ui("eda_filter_ui"),
                        style="background-color: #f8f9fa; border-radius: 5px; padding: 15px; margin-bottom: 20px;"
                    ),
                    
                    # Data Preview and Visualization
                    ui.row(
                        ui.column(6,
                            ui.tags.div(
                                ui.tags.div(class_="card-header", children="Data Preview"),
                                ui.tags.div(
                                    ui.output_table("eda_data_preview"),
                                    class_="card-body"
                                ),
                                class_="card"
                            )
                        ),
                        ui.column(6,
                            ui.tags.div(
                                ui.tags.div(class_="card-header", children="Numerical Summary"),
                                ui.tags.div(
                                    ui.output_table("eda_numerical_summary"),
                                    class_="card-body"
                                ),
                                class_="card"
                            )
                        )
                    ),
                    
                    # Plot and Description
                    ui.row(
                        ui.column(8,
                            ui.tags.div(
                                ui.tags.div(class_="card-header", children="Visualization"),
                                ui.tags.div(
                                    ui.output_plot("eda_plot"),
                                    class_="card-body",
                                    style="min-height: 400px;"
                                ),
                                class_="card"
                            )
                        ),
                        ui.column(4,
                            ui.tags.div(
                                ui.tags.div(class_="card-header", children="Plot Description"),
                                ui.tags.div(
                                    ui.output_ui("eda_plot_description"),
                                    class_="card-body"
                                ),
                                class_="card"
                            )
                        )
                    ),
                    
                    # Correlation Analysis
                    ui.tags.div(
                        ui.tags.div(class_="card-header", children="Correlation Analysis"),
                        ui.tags.div(
                            ui.output_plot("eda_correlation_plot"),
                            class_="card-body",
                            style="min-height: 400px;"
                        ),
                        class_="card"
                    ),
                    
                    class_="main-panel"
                )
            )
        )
    )
)

def server(input, output, session):
    data = reactive.Value(None)
    processing_status = reactive.Value("")
    feature_status = reactive.Value("")
    previous_data = reactive.Value(None)
    cleaning_history = reactive.Value(None)
    
    def clean_text(text):
        """ Remove HTML content and keep only ASCII characters """
        if pd.isna(text):
            return text
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()

    def read_dataset(file_path, file_ext):
        """Read dataset from various file formats"""
        try:
            if file_ext == "csv":
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"✓ Successfully read CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        raise Exception(f"Error reading CSV: {str(e)}")
                
            elif file_ext in ["xlsx", "xls"]:
                try:
                    df = pd.read_excel(file_path, engine='openpyxl' if file_ext == 'xlsx' else 'xlrd')
                    print(f"✓ Successfully read {file_ext.upper()} file")
                except Exception as e:
                    raise Exception(f"Error reading Excel file: {str(e)}")
                
            elif file_ext == "json":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.DataFrame([data])
                    print("✓ Successfully read JSON file")
                except:
                    df = pd.read_json(file_path, lines=True)
                    print("✓ Successfully read JSON Lines file")
                
            elif file_ext == "rds":
                result = pyreadr.read_r(file_path)
                df = result[None] if None in result else result[list(result.keys())[0]]
                print("✓ Successfully read RDS file")
                
            else:
                raise Exception(f"Unsupported file format: {file_ext}")

            if df.empty:
                raise Exception("Loaded data is empty")
                
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.astype(str)
            print(f"✓ Loaded data shape: {df.shape}")
            return df

        except Exception as e:
            print(f"❌ Error reading file: {str(e)}")
            raise e

    @reactive.effect
    def update_data():
        """Read the file or load built-in dataset"""
        file_info = input.file1()
        builtin_selected = input.builtinDataset()
        
        if builtin_selected != "None":
            df = get_builtin_dataset(builtin_selected)
            if df is not None:
                update_ui_with_data(df)
                processing_status.set(f"✓ Successfully loaded {builtin_selected} dataset\n")
            return

        if file_info:
            ui.update_select("builtinDataset", selected="None")
            try:
                file_path = file_info[0]["datapath"]
                file_ext = file_info[0]["name"].split(".")[-1].lower()
                df = read_dataset(file_path, file_ext)
                update_ui_with_data(df)
                processing_status.set(f"✓ Successfully loaded {file_ext.upper()} file\n")
            except Exception as e:
                processing_status.set(f"❌ Error: {str(e)}")

    def update_ui_with_data(df):
        """Update UI elements when new data is loaded"""
        ui.update_checkbox_group("varSelect", choices=df.columns.tolist(), 
                               selected=df.columns.tolist())
        ui.update_select("featureColumn", choices=df.columns.tolist())
        ui.update_selectize("multiColumns", choices=df.columns.tolist())
        
        # Update visualization tab dropdowns
        ui.update_select("x_var", choices=df.columns.tolist())
        ui.update_select("y_var", choices=df.columns.tolist())
        ui.update_select("summary_var", choices=df.columns.tolist())
        
        data.set(df)

    @reactive.effect
    @reactive.event(input.applyFeatureEng)
    def apply_feature_engineering():
        """Apply feature engineering operations"""
        df = data.get()
        if df is None:
            feature_status.set("❌ No data available for feature engineering")
            return

        try:
            # Store current state before modification
            previous_data.set(df.copy())
            
            df = df.copy()
            status_messages = []
            
            column = input.featureColumn()
            operation = input.featureOperation()
            
            if column and operation != "None":
                if operation == "Normalize":
                    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
                    status_messages.append(f"✓ Normalized column: {column}")
                    
                elif operation == "One-Hot":
                    df = pd.get_dummies(df, columns=[column])
                    status_messages.append(f"✓ One-hot encoded column: {column}")
                    
                elif operation == "Convert Date Format":
                    df[column] = pd.to_datetime(df[column], errors="coerce")
                    df[f"{column}_year"] = df[column].dt.year
                    df[f"{column}_month"] = df[column].dt.month
                    df[f"{column}_day"] = df[column].dt.day
                    
                    try:
                        input_date = datetime(input.input_year(), 
                                           input.input_month(), 
                                           input.input_day())
                        df[f"{column}_days_since_input_date"] = (input_date - df[column]).dt.days
                        status_messages.append(f"✓ Created date features for: {column}")
                    except ValueError as e:
                        status_messages.append(f"⚠️ Invalid date input: {str(e)}")
                    
                    df.drop(columns=[column], inplace=True)
                    
                elif operation == "Box-Cox":
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                    if df[column].min() <= 0:
                        shift = abs(df[column].min()) + 1
                        df[column] += shift
                        status_messages.append(f"✓ Shifted data by {shift}")
                    df[column], _ = stats.boxcox(df[column])
                    status_messages.append(f"✓ Applied Box-Cox transformation")

            # Multi-column operations
            selected_columns = list(input.multiColumns())
            extra_operation = input.extraOperation()
            
            if len(selected_columns) >= 2 and extra_operation != "None":
                if extra_operation == "Average":
                    df["weighted_avg"] = df[selected_columns].mean(axis=1)
                    status_messages.append(f"✓ Created weighted average of: {', '.join(selected_columns)}")
                    
                elif extra_operation == "Interactions":
                    col1, col2 = selected_columns[:2]
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                    status_messages.append(f"✓ Created interaction: {col1}_x_{col2}")

            # Update data and UI with feature engineering status
            data.set(df)
            update_ui_with_data(df)
            feature_status.set("\n".join(status_messages))
            
        except Exception as e:
            feature_status.set(f"❌ Error in feature engineering: {str(e)} is not a selected column")

    @reactive.effect
    @reactive.event(input.revertChange)
    def revert_last_change():
        """Revert to the previous state before last feature engineering operation"""
        prev_df = previous_data.get()
        if prev_df is None:
            feature_status.set("⚠️ No previous state available to revert to")
            return
        
        # Restore previous state
        data.set(prev_df)
        update_ui_with_data(prev_df)
        feature_status.set("✓ Reverted to previous state")

    @output
    @render.download(
        filename=lambda: f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    def downloadData():
        df = data.get()
        if df is not None:
            try:
                csv_bytes = df.to_csv(index=False).encode('utf-8')
                return {"content-type": "text/csv", "content": csv_bytes}
            except Exception as e:
                print(f"Download error: {str(e)}")
                return {"content-type": "text/plain", "content": b"Error occurred while downloading data"}
        return {"content-type": "text/plain", "content": b"No data available"}


    @output
    @render.text
    def featureStatus():
        return feature_status.get()

    @output
    @render.table
    def featureDataTable():
        df = data.get()
        if df is None:
            return pd.DataFrame({'Message': ['No data available']})
        return df.head(10)

    @reactive.effect
    @reactive.event(input.processData)
    def process_data():
        """Process data: keep selected variables and handle missing values dynamically"""
        df = data.get()
        if df is None:
            processing_status.set("❌ No data loaded")
            return

        # Store current state before modification
        cleaning_history.set(df.copy())
        
        selected_vars = input.varSelect()
        if not selected_vars:
            processing_status.set("⚠️ Error: Please select at least one variable to proceed\n")
            return

        df = df.copy()
        status_messages = []

        # Keep selected variables
        valid_vars = [col for col in selected_vars if col in df.columns]
        if not valid_vars:
            status_messages.append("⚠️ No valid columns selected")
            return
        else:
            df = df.loc[:, valid_vars].copy()
            status_messages.append(f"✓ Selected {len(valid_vars)} variables")

        # Handle missing values
        missing_option = input.missingDataOption()
        if missing_option == "Convert Common Missing Values to NA":
            missing_values = ["", "-9", "-99", "NA", "N/A", "nan", "NaN", "null", "NULL", "None"]
            original_na_count = df.isna().sum().sum()
            df.replace(missing_values, pd.NA, inplace=True)
            
            string_cols = df.select_dtypes(include=['object']).columns
            for col in string_cols:
                df[col] = df[col].apply(lambda x: pd.NA if isinstance(x, str) and x.strip() == "" else x)
            
            new_na_count = df.isna().sum().sum()
            status_messages.append(f"✓ Converted {new_na_count - original_na_count} values to NA")
                
        elif missing_option == "Listwise Deletion":
            original_rows = len(df)
            df = df.dropna()
            rows_dropped = original_rows - len(df)
            status_messages.append(f"✓ Dropped {rows_dropped} rows with missing values")
            
        elif missing_option == "Mean Imputation":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            imputed_count = 0
            for col in numeric_cols:
                if df[col].isna().any():
                    mean_val = df[col].mean()
                    na_count = df[col].isna().sum()
                    df[col].fillna(mean_val, inplace=True)
                    imputed_count += na_count
            if imputed_count > 0:
                status_messages.append(f"✓ Imputed {imputed_count} missing values with mean values")
                    
        elif missing_option == "Mode Imputation":
            imputed_count = 0
            for col in df.columns:
                if df[col].isna().any():
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
                    if mode_val is not None:
                        na_count = df[col].isna().sum()
                        df[col].fillna(mode_val, inplace=True)
                        imputed_count += na_count
            if imputed_count > 0:
                status_messages.append(f"✓ Imputed {imputed_count} missing values with mode values")

        selected_processing_options = input.dataProcessingOptions()
        if "Remove Duplicates" in selected_processing_options:
            original_rows = len(df)
            df = df.drop_duplicates()
            rows_dropped = original_rows - len(df)
            if rows_dropped > 0:
                status_messages.append(f"✓ Removed {rows_dropped} duplicate rows")
            
        if "Standardize Data" in selected_processing_options:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                status_messages.append(f"✓ Standardized {len(numeric_cols)} numeric columns")
                
        if "Normalize Data" in selected_processing_options:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if not numeric_cols.empty:
                scaler = MinMaxScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                status_messages.append(f"✓ Normalized {len(numeric_cols)} numeric columns")
                
        if "One-Hot Encoding" in selected_processing_options:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                original_cols = df.shape[1]
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                new_cols = df.shape[1] - original_cols
                status_messages.append(f"✓ One-hot encoding added {new_cols} new columns")

        status_messages.append(f"\n✅ Processing complete! New shape: {df.shape}\n")
        processing_status.set("\n".join(status_messages))
        data.set(df)

    @output
    @render.text
    def dataSummary():
        df = data.get()
        status = processing_status.get()
        
        if df is None or df.empty:
            return "No data loaded"
        
        summary = f"""{'=' * 40}
DATASET OVERVIEW
{'=' * 40}

Dataset Shape:           {df.shape[0]} rows × {df.shape[1]} columns
Memory Usage:           {df.memory_usage().sum() / 1024**2:.2f} MB
Number of Duplicate Rows: {df.duplicated().sum()}

{'=' * 40}
COLUMN TYPES
{'=' * 40}

Numerical Columns:  {len(df.select_dtypes(include=['int64', 'float64']).columns)}
Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}
DateTime Columns:    {len(df.select_dtypes(include=['datetime64']).columns)}

"""
        columns = df.columns.tolist()
        column_list = '\n'.join(f"{i:3d}. {col}" for i, col in enumerate(columns, 1))
        
        if status:
            summary += f"\n{'=' * 40}\nPROCESSING STATUS\n{'=' * 40}\n\n{status}"
            
        return summary + column_list

    @output
    @render.table
    def dataTypesTable():
        df = data.get()
        if df is None or df.empty:
            return pd.DataFrame()

        dtype_info = []
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_percentage = (missing_count / len(df)) * 100
            unique_count = df[column].nunique()
            
            dtype_info.append({
                "Column Name": column,
                "Data Type": str(df[column].dtype),
                "Missing Values": f"{missing_count} ({missing_percentage:.1f}%)",
                "Unique Values": unique_count
            })
        
        return pd.DataFrame(dtype_info)

    @output
    @render.table
    def numericalSummary():
        df = data.get()
        if df is None or df.empty:
            return pd.DataFrame()

        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) == 0:
            return pd.DataFrame({'Message': ['No numerical columns found']})

        summary_stats = []
        for col in numerical_cols:
            stats = df[col].describe()
            summary_stats.append({
                'Column': col,
                'Mean': f"{stats['mean']:.2f}",
                'Std': f"{stats['std']:.2f}",
                'Min': f"{stats['min']:.2f}",
                'Q1': f"{stats['25%']:.2f}",
                'Median': f"{stats['50%']:.2f}",
                'Q3': f"{stats['75%']:.2f}",
                'Max': f"{stats['max']:.2f}"
            })
        
        return pd.DataFrame(summary_stats)

    @output
    @render.table
    def dataTable():
        df = data.get()
        if df is None or df.empty:
            return pd.DataFrame()
        
        preview_df = df.head(10).copy()
        for col in preview_df.select_dtypes(include=['object']).columns:
            preview_df[col] = preview_df[col].apply(
                lambda x: str(x)[:50] + '...' if len(str(x)) > 50 else str(x)
            )
        return preview_df
        
    @output
    @render.ui
    def main_plot():
        df = data.get()
        if df is None or df.empty or not input.x_var() or not input.y_var():
            fig = go.Figure()
            fig.update_layout(title="No data to display")
            return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            
        try:
            x_col = input.x_var()
            y_col = input.y_var()
            plot_type = input.plot_type()
            
            if plot_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            elif plot_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif plot_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            elif plot_type == "histogram":
                fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
            else:
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                template="plotly_white"
            )
            return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"Error creating plot: {str(e)}")
            return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            
    @output
    @render.table
    def summary_stats():
        df = data.get()
        if df is None or df.empty or not input.summary_var():
            return pd.DataFrame({'Message': ['No data available or variable selected']})
            
        try:
            var = input.summary_var()
            
            if pd.api.types.is_numeric_dtype(df[var]):
                stats_df = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
                    'Value': [
                        df[var].count(),
                        df[var].mean(),
                        df[var].std(),
                        df[var].min(),
                        df[var].quantile(0.25),
                        df[var].median(),
                        df[var].quantile(0.75),
                        df[var].max()
                    ]
                })
            else:
                value_counts = df[var].value_counts().reset_index()
                value_counts.columns = ['Value', 'Count']
                stats_df = value_counts
                
            return stats_df
        except Exception as e:
            return pd.DataFrame({'Error': [str(e)]})
            
    @output
    @render.ui
    def distribution_plot():
        df = data.get()
        if df is None or df.empty or not input.summary_var():
            fig = go.Figure()
            fig.update_layout(title="No data to display")
            return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            
        try:
            var = input.summary_var()
            
            if pd.api.types.is_numeric_dtype(df[var]):
                fig = px.histogram(
                    df, x=var,
                    title=f"Distribution of {var}",
                    marginal="box"
                )
            else:
                value_counts = df[var].value_counts().reset_index()
                fig = px.bar(
                    value_counts, 
                    x='index', 
                    y=var,
                    title=f"Distribution of {var}"
                )
                fig.update_xaxes(title="Value")
                fig.update_yaxes(title="Count")
                
            return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            fig = go.Figure()
            fig.update_layout(title=f"Error creating plot: {str(e)}")
            return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    @reactive.effect
    @reactive.event(input.deselectAll)
    def deselect_all_variables():
        """Deselect all variables in the checkbox group"""
        df = data.get()
        if df is not None:
            ui.update_checkbox_group("varSelect", selected=[])

    @reactive.effect
    @reactive.event(input.selectAll)
    def select_all_variables():
        """Select all variables in the checkbox group"""
        df = data.get()
        if df is not None:
            ui.update_checkbox_group("varSelect", selected=df.columns.tolist())
            

    @output
    @render.ui
    def eda_filter_ui():
        df = data.get()
        if df is None:
            return ui.TagList()
        
        filter_inputs = ui.TagList()
        
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(col_data) == 0:
                        continue
                    
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    
                    if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
                        continue
                    
                    if max_val - min_val > 1e10:
                        continue
                        
                    filter_inputs.append(
                        ui.input_slider(f"eda_filter_{col}", f"Filter {col}:", 
                                      min=min_val, max=max_val, 
                                      value=[min_val, max_val])
                    )
                elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    unique_vals = df[col].dropna().unique().tolist()
                    if len(unique_vals) < 15 and len(unique_vals) > 0:  # Only create filter for categorical with reasonable number of values
                        choices = {str(val): str(val) for val in unique_vals}
                        filter_inputs.append(
                            ui.input_checkbox_group(f"eda_filter_{col}", f"Filter {col}:", 
                                                  choices=choices))
            except Exception as e:
                print(f"Error creating filter for column {col}: {str(e)}")
                continue
        
        return filter_inputs
    
    @reactive.Calc
    def get_eda_filtered_data():
        df = data.get()
        if df is None:
            return None
        
        try:
            for col in df.columns:
                if hasattr(input, f"eda_filter_{col}"):
                    try:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            range_val = getattr(input, f"eda_filter_{col}")()
                            if range_val and len(range_val) == 2:
                                mask = df[col].notna() & (df[col] >= range_val[0]) & (df[col] <= range_val[1])
                                df = df[mask]
                        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                            selected = getattr(input, f"eda_filter_{col}")()
                            if selected and len(selected) > 0:
                                df = df[df[col].isin(selected)]
                    except Exception as e:
                        print(f"Error applying filter for column {col}: {str(e)}")
                        continue
            
            return df
        except Exception as e:
            print(f"Error in get_eda_filtered_data: {str(e)}")
            return df
    
    @output
    @render.ui
    def eda_plot_controls():
        df = data.get()
        if df is None:
            return ui.TagList()
        
        plot_type = input.eda_plot_type()
        controls = ui.TagList()
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numeric_cols) == 0 or (plot_type != "heatmap" and len(numeric_cols) == 0):
            return ui.p("Not enough appropriate columns for the selected plot type.")
        
        all_cols = df.columns.tolist()
        
        if plot_type == "histogram":
            controls.append(ui.input_select("eda_hist_col", "Select Column:", 
                                          {col: col for col in numeric_cols}))
            controls.append(ui.input_slider("eda_hist_bins", "Number of Bins:", 
                                          min=5, max=50, value=20))
            controls.append(ui.input_checkbox("eda_hist_kde", "Show KDE", value=True))
            if len(cat_cols) > 0:
                controls.append(ui.input_select("eda_hist_hue", "Color by (optional):", 
                                              {"": "None", **{col: col for col in cat_cols}}))
                
        elif plot_type == "scatter":
            controls.append(ui.input_select("eda_scatter_x", "X-axis:", 
                                          {col: col for col in numeric_cols}))
            controls.append(ui.input_select("eda_scatter_y", "Y-axis:", 
                                          {col: col for col in numeric_cols}))
            if len(cat_cols) > 0:
                controls.append(ui.input_select("eda_scatter_hue", "Color by (optional):", 
                                              {"": "None", **{col: col for col in cat_cols}}))
            controls.append(ui.input_checkbox("eda_scatter_regression", "Show Regression Line", value=False))
            
        elif plot_type == "bar":
            controls.append(ui.input_select("eda_bar_x", "X-axis:", 
                                          {col: col for col in all_cols}))
            controls.append(ui.input_select("eda_bar_y", "Y-axis (optional):", 
                                          {"": "Count", **{col: col for col in numeric_cols}}))
            if len(cat_cols) > 0:
                controls.append(ui.input_select("eda_bar_hue", "Color by (optional):", 
                                              {"": "None", **{col: col for col in cat_cols}}))
        
        return controls
    
    @output
    @render.table
    def eda_data_preview():
        df = get_eda_filtered_data()
        if df is None:
            return pd.DataFrame({'Message': ['No data available']})
        
        return df.head(10)
    
    @output
    @render.table
    def eda_numerical_summary():
        df = get_eda_filtered_data()
        if df is None:
            return pd.DataFrame({'Message': ['No data available']})
        
        numeric_data = df.select_dtypes(include=['number'])
        if numeric_data.shape[1] == 0:
            return pd.DataFrame({'Message': ['No numeric columns in dataset']})
        
        summary = numeric_data.describe().transpose()
        
        summary['skew'] = numeric_data.skew()
        summary['kurtosis'] = numeric_data.kurtosis()
        
        summary = summary.round(4)
        
        summary = summary.reset_index()
        summary.rename(columns={'index': 'Column'}, inplace=True)
        
        return summary
    
    @output
    @render.plot
    def eda_plot():
        df = get_eda_filtered_data()
        if df is None:
            return plt.figure()

        plot_type = input.eda_plot_type()
        fig = plt.figure(figsize=(10, 6))
        
        try:
            if plot_type == "histogram":
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if not num_cols:
                    plt.text(0.5, 0.5, "No numeric columns available for histogram", 
                           ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    return fig
                
                col = input.eda_hist_col() if hasattr(input, "eda_hist_col") else num_cols[0]
                bins = input.eda_hist_bins() if hasattr(input, "eda_hist_bins") else 20
                kde = input.eda_hist_kde() if hasattr(input, "eda_hist_kde") else True
                
                hue = None
                if hasattr(input, "eda_hist_hue") and input.eda_hist_hue():
                    hue = input.eda_hist_hue()
                
                sns.histplot(data=df, x=col, bins=bins, kde=kde, hue=hue)
                plt.title(f'Histogram of {col}')
                plt.tight_layout()
                
            elif plot_type == "scatter":
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) < 2:
                    plt.text(0.5, 0.5, "Need at least 2 numeric columns for scatter plot", 
                           ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    return fig
                
                x_col = input.eda_scatter_x() if hasattr(input, "eda_scatter_x") else num_cols[0]
                y_col = input.eda_scatter_y() if hasattr(input, "eda_scatter_y") else num_cols[1]
                
                regression = input.eda_scatter_regression() if hasattr(input, "eda_scatter_regression") else False
                
                hue = None
                if hasattr(input, "eda_scatter_hue") and input.eda_scatter_hue():
                    hue = input.eda_scatter_hue()
                
                scatter = sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue)
                
                if regression:
                    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, ax=scatter.axes)
                
                plt.title(f'Scatter Plot of {y_col} vs {x_col}')
                plt.tight_layout()
                
            elif plot_type == "bar":
                x_col = input.eda_bar_x() if hasattr(input, "eda_bar_x") else df.columns[0]
                y_col = input.eda_bar_y() if hasattr(input, "eda_bar_y") and input.eda_bar_y() else None
                
                hue = None
                if hasattr(input, "eda_bar_hue") and input.eda_bar_hue():
                    hue = input.eda_bar_hue()
                
                if y_col:
                    sns.barplot(data=df, x=x_col, y=y_col, hue=hue)
                    plt.title(f'Bar Plot of {y_col} by {x_col}')
                else:
                    sns.countplot(data=df, x=x_col, hue=hue)
                    plt.title(f'Count of Records by {x_col}')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
            elif plot_type == "heatmap":
                corr_data = df.select_dtypes(include=['number']).corr()
                if corr_data.shape[0] < 2:
                    plt.text(0.5, 0.5, "Need at least 2 numeric columns for correlation heatmap", 
                           ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    return fig
                
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title('Correlation Matrix')
                plt.tight_layout()
        
        except Exception as e:
            plt.clf()
            plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            
        return fig
    
    @output
    @render.ui
    def eda_plot_description():
        df = get_eda_filtered_data()
        if df is None:
            return ui.TagList()
        
        plot_type = input.eda_plot_type()
        plot_info = ui.TagList()
        
        try:
            if plot_type == "histogram":
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if not num_cols:
                    return ui.p("No numeric columns available for analysis")
                
                col = input.eda_hist_col() if hasattr(input, "eda_hist_col") else num_cols[0]
                
                description = [
                    ui.h4(f"Histogram Analysis: {col}"),
                    ui.p(f"Mean: {df[col].mean():.4f}"),
                    ui.p(f"Median: {df[col].median():.4f}"),
                    ui.p(f"Standard Deviation: {df[col].std():.4f}"),
                    ui.p(f"Skewness: {df[col].skew():.4f}"),
                    ui.p(f"Kurtosis: {df[col].kurtosis():.4f}")
                ]
                plot_info.extend(description)
                
            elif plot_type == "scatter":
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) < 2:
                    return ui.p("Need at least 2 numeric columns for analysis")
                
                x_col = input.eda_scatter_x() if hasattr(input, "eda_scatter_x") else num_cols[0]
                y_col = input.eda_scatter_y() if hasattr(input, "eda_scatter_y") else num_cols[1]
                
                correlation = df[[x_col, y_col]].corr().iloc[0, 1]
                
                if hasattr(input, "eda_scatter_regression") and input.eda_scatter_regression():
                    X = sm.add_constant(df[x_col])
                    model = sm.OLS(df[y_col], X).fit()
                    description = [
                        ui.h4(f"Scatter Plot Analysis: {y_col} vs {x_col}"),
                        ui.p(f"Correlation: {correlation:.4f}"),
                        ui.h5("Regression Summary:"),
                        ui.p(f"Intercept: {model.params[0]:.4f}"),
                        ui.p(f"Slope: {model.params[1]:.4f}"),
                        ui.p(f"R-squared: {model.rsquared:.4f}"),
                        ui.p(f"P-value: {model.f_pvalue:.4f}")
                    ]
                else:
                    description = [
                        ui.h4(f"Scatter Plot Analysis: {y_col} vs {x_col}"),
                        ui.p(f"Correlation: {correlation:.4f}")
                    ]
                plot_info.extend(description)
                
            elif plot_type == "bar":
                x_col = input.eda_bar_x() if hasattr(input, "eda_bar_x") else df.columns[0]
                y_col = input.eda_bar_y() if hasattr(input, "eda_bar_y") and input.eda_bar_y() else None
                
                if y_col:
                    grouped = df.groupby(x_col)[y_col].agg(['mean', 'count']).reset_index()
                    if not grouped.empty:
                        description = [
                            ui.h4(f"Bar Plot Analysis: {y_col} by {x_col}"),
                            ui.p(f"Number of groups: {grouped.shape[0]}"),
                            ui.p(f"Highest average: {grouped.loc[grouped['mean'].idxmax(), x_col]} ({grouped['mean'].max():.4f})"),
                            ui.p(f"Lowest average: {grouped.loc[grouped['mean'].idxmin(), x_col]} ({grouped['mean'].min():.4f})")
                        ]
                    else:
                        description = [ui.p("No data available for analysis after grouping")]
                else:
                    value_counts = df[x_col].value_counts()
                    if not value_counts.empty:
                        description = [
                            ui.h4(f"Count Plot Analysis: {x_col}"),
                            ui.p(f"Number of unique values: {value_counts.shape[0]}"),
                            ui.p(f"Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)"),
                            ui.p(f"Least common: {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)")
                        ]
                    else:
                        description = [ui.p("No data available for analysis")]
                plot_info.extend(description)
                
            elif plot_type == "heatmap":
                corr_data = df.select_dtypes(include=['number']).corr()
                if corr_data.shape[0] < 2:
                    return ui.p("Need at least 2 numeric columns for correlation analysis")
                
                high_corr = corr_data.unstack().sort_values(ascending=False)
                high_corr = high_corr[(high_corr < 1.0) & (high_corr > 0.5)]
                
                description = [
                    ui.h4("Correlation Heatmap Analysis:"),
                    ui.p(f"Number of numeric features: {corr_data.shape[0]}")
                ]
                
                if len(high_corr) > 0:
                    description.append(ui.h5("Strong Positive Correlations (>0.5):"))
                    for idx, corr_val in high_corr.items():
                        if idx[0] != idx[1]:  # Skip self-correlations
                            description.append(ui.p(f"{idx[0]} & {idx[1]}: {corr_val:.4f}"))
                plot_info.extend(description)
        
        except Exception as e:
            plot_info.append(ui.p(f"Error generating plot description: {str(e)}"))
            
        return plot_info
    
    @output
    @render.plot
    def eda_correlation_plot():
        df = get_eda_filtered_data()
        if df is None:
            return plt.figure()
        
        numeric_data = df.select_dtypes(include=['number'])
        if numeric_data.shape[1] < 2:
            fig = plt.figure()
            plt.text(0.5, 0.5, "Need at least 2 numeric columns for correlation analysis", 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
            return fig
        
        fig = plt.figure(figsize=(10, 8))
        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix (Upper Triangle Hidden)')
        plt.tight_layout()
        return fig

    @reactive.effect
    @reactive.event(input.revertCleaningChange)
    def revert_cleaning_change():
        """Revert to the previous state before last cleaning operation"""
        prev_df = cleaning_history.get()
        if prev_df is None:
            processing_status.set("⚠️ No previous state available to revert to\n")
            return
        
        data.set(prev_df)
        update_ui_with_data(prev_df)
        processing_status.set("✓ Reverted to previous state\n")

app = App(app_ui, server)