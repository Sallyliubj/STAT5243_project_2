from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from bs4 import BeautifulSoup
import re
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import pyreadr
import openpyxl


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

# Create the navbar page
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Data Cleaning",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Data Loading and Cleaning"),
                ui.input_file("file1", "Select Data File", 
                             accept=[".csv", ".xlsx", ".xls", ".json", ".rds"]),
                ui.input_select("builtinDataset", "Choose Built-in Dataset",
                              choices=["None", "iris", "wine", "breast_cancer", "diabetes"], 
                              selected="None"),
                ui.div(
                    ui.input_checkbox_group("varSelect", "Select Variables to Keep:", choices=[]),
                    ui.div(
                        ui.input_action_button("selectAll", "Select All", 
                                             class_="btn-secondary btn-sm",
                                             style="margin-right: 10px;"),
                        ui.input_action_button("deselectAll", "Deselect All", 
                                             class_="btn-secondary btn-sm"),
                        style="margin-top: 5px;"
                    ),
                    style="margin-bottom: 15px;"
                ),
                ui.input_select("missingDataOption", "Handle Missing Values:",
                              choices=["None", "Convert Common Missing Values to NA", 
                                     "Listwise Deletion", "Mean Imputation", "Mode Imputation"], 
                              selected="None"),
                ui.input_checkbox_group("dataProcessingOptions", "Additional Cleaning Steps:",
                                      choices=["Remove Duplicates", "Standardize Data", 
                                             "Normalize Data", "One-Hot Encoding"]),
                ui.div(
                    ui.input_action_button("processData", "Clean Data", 
                                         class_="btn-primary"),
                    ui.input_action_button("revertCleaningChange", "Revert Last Change",
                                         class_="btn-warning",
                                         style="margin-left: 10px;"),
                ),
            ),
            ui.card(
                ui.h4("Data Summary"),
                ui.tags.pre(ui.output_text("dataSummary")),
                ui.h4("Data Types and Missing Values"),
                ui.output_table("dataTypesTable"),
                ui.h4("Numerical Columns Summary"),
                ui.output_table("numericalSummary"),
                ui.h4("Sample Data Preview"),
                ui.output_table("dataTable")
            )
        )
    ),
    ui.nav_panel(
        "Feature Engineering",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Feature Engineering Operations"),
                ui.input_select("featureColumn", "Select Column for Feature Engineering", choices=[]),
                ui.input_select("featureOperation", "Select Operation", choices=[
                    "Normalize", "One-Hot", "Convert Date Format", "Box-Cox"
                ]),
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
                    "input.featureOperation === 'Box-Cox'",
                    ui.p("Transforms data to be more normally distributed. Useful for skewed numerical data.", 
                         style="color: #666; font-style: italic; margin: 10px 0;"),
                    ui.p("Note: Only works with positive numerical values. Negative values will be shifted.", 
                         style="color: red; font-style: italic;"),
                ),
                ui.hr(),
                ui.h4("Add a New Feature from Multiple Columns"),
                ui.input_selectize("multiColumns", "Select Columns for New Feature", 
                                 choices=[], multiple=True),
                ui.input_select("extraOperation", "Create New Feature", choices=[
                    "None", "Average", "Interactions"
                ]),
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
                ui.div(
                    ui.input_action_button("applyFeatureEng", "Apply Feature Engineering", 
                                         class_="btn-primary"),
                    ui.input_action_button("revertChange", "Revert Last Change",
                                         class_="btn-warning",
                                         style="margin-left: 10px;"),
                ),
                ui.hr(),
                ui.download_button("downloadData", "Download Processed Data")
            ),
            ui.card(
                ui.h4("Feature Engineering Results"),
                ui.tags.pre(ui.output_text("featureStatus")),
                ui.h4("Updated Data Preview"),
                ui.output_table("featureDataTable")
            )
        )
    ),
    title="Data Preprocessing Pipeline",
    bg="#f8f9fa",
    inverse=True,
    selected="Data Cleaning"
)

def server(input, output, session):
    data = reactive.Value(None)
    original_data = reactive.Value(None)
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

    def update_ui_with_data(df, is_original=False):
        """Update UI elements when new data is loaded"""
        if is_original:
            # Update variable selection with all original columns
            ui.update_checkbox_group("varSelect", 
                                   choices=df.columns.tolist(),
                                   selected=df.columns.tolist())
            original_data.set(df)
        
        # Update feature engineering dropdowns
        ui.update_select("featureColumn", choices=[""] + df.columns.tolist())
        ui.update_selectize("multiColumns", choices=df.columns.tolist())
        data.set(df)

   

    @reactive.effect
    def update_data():
        """Read the file or load built-in dataset"""
        file_info = input.file1()
        builtin_selected = input.builtinDataset()
        
        if builtin_selected != "None":
            df = get_builtin_dataset(builtin_selected)
            if df is not None:
                update_ui_with_data(df, is_original=True)
                processing_status.set(f"✓ Successfully loaded {builtin_selected} dataset\n")
            return

        if file_info:
            ui.update_select("builtinDataset", selected="None")
            try:
                file_path = file_info[0]["datapath"]
                file_ext = file_info[0]["name"].split(".")[-1].lower()
                df = read_dataset(file_path, file_ext)
                update_ui_with_data(df, is_original=True)
                processing_status.set(f"✓ Successfully loaded {file_ext.upper()} file\n")
            except Exception as e:
                processing_status.set(f"❌ Error: {str(e)}")

    @reactive.effect
    @reactive.event(input.applyFeatureEng)
    def apply_feature_engineering():
        """Apply feature engineering operations"""
        df = data.get()
        if df is None:
            feature_status.set("❌ No data available for feature engineering")
            return

        try:
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

    @output
    @render.download
    def downloadData():
        """Download the processed dataset"""
        def download():
            df = data.get()
            if df is not None:
                return df.to_csv(index=False)
            return ""
        
        return download

    # Add separate outputs for feature engineering page
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
        orig_df = original_data.get()
        
        if df is None:
            processing_status.set("❌ No data loaded")
            return

        # Check if any variables are selected
        selected_vars = input.varSelect()
        if not selected_vars:
            processing_status.set("⚠️ Error: Please select at least one variable to proceed\n")
            return

        # Store current state before modification
        cleaning_history.set(df.copy())
        
        # Start with original data and select variables
        df = orig_df.copy()
        status_messages = []

        valid_vars = [col for col in selected_vars if col in df.columns]
        if not valid_vars:
            status_messages.append("⚠️ No valid columns selected")
            return
        else:
            df = df.loc[:, valid_vars].copy()
            status_messages.append(f"✓ Selected {len(valid_vars)} variables")

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
        df = original_data.get()
        if df is not None:
            ui.update_checkbox_group("varSelect", selected=df.columns.tolist())

    # Add new reactive effect for reverting changes
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

    @reactive.effect
    @reactive.event(input.revertCleaningChange)
    def revert_cleaning_change():
        """Revert to the previous state before last cleaning operation"""
        prev_df = cleaning_history.get()
        if prev_df is None:
            processing_status.set("⚠️ No previous state available to revert to\n")
            return
        
        data.set(prev_df)
        # Don't update the variable selection list when reverting
        ui.update_select("featureColumn", choices=[""] + prev_df.columns.tolist())
        ui.update_selectize("multiColumns", choices=prev_df.columns.tolist())
        processing_status.set("✓ Reverted to previous state\n")

app = App(app_ui, server) 