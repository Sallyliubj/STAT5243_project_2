from shiny import App, ui, render, reactive
import pandas as pd
import io
import re
from bs4 import BeautifulSoup
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import numpy as np
import json


try:
    import pyreadr  # reading RDS files
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False
    print("Note: pyreadr not installed. RDS files will not be supported.")

try:
    import openpyxl  #  Excel files
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("Note: openpyxl not installed. Modern Excel files will not be supported.")


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

        # Add target column
        df["target"] = data.target
        print(f"✓ Successfully loaded {name} dataset\n")
        print(f"✓ Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"❌ Error loading {name} dataset: {str(e)}")
        return None

# --- UI Design ---
app_ui = ui.page_fluid(
    ui.panel_title("Data Cleaning and Feature Selection Tool - Python Shiny"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("file1", "Select Data File", 
                         accept=[".csv", ".xlsx", ".xls", ".json", ".rds"]),
            ui.input_select("builtinDataset", "Choose Built-in Dataset",
                            choices=["None", "iris", "wine", "breast_cancer", "diabetes"], selected="None"),
            ui.input_checkbox_group("varSelect", "Select Variables to Keep:", choices=[]),
            ui.input_select("missingDataOption", "Handle Missing Values:",
                            choices=["None", "Convert Common Missing Values to NA", "Listwise Deletion",
                                     "Mean Imputation", "Mode Imputation"], selected="None"),
            ui.input_checkbox_group("dataProcessingOptions", "Additional Cleaning Steps:",
                                    choices=["Remove Duplicates", "Standardize Data", "Normalize Data", "One-Hot Encoding"]),
            ui.input_action_button("processData", "Process Data", class_="btn-primary"),
        ),
        ui.card(
            ui.h4("Data Summary"),
            ui.tags.pre(ui.output_text("dataSummary")),
            ui.h4("Data Types and Missing Values"),
            ui.output_table("dataTypesTable"),
            ui.h4("Numerical Columns Summary"),
            ui.output_table("numericalSummary"),
            ui.h4("Sample Data Preview"),
            ui.output_table("dataTable"),
        )
    )
)

# --- Server Logic ---
def server(input, output, session):
    data = reactive.Value(None)
    processing_status = reactive.Value("")

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
                if not HAS_OPENPYXL and file_ext == "xlsx":
                    raise Exception("openpyxl not installed. Please install it to read modern Excel files.")
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
                if not HAS_PYREADR:
                    raise Exception("pyreadr not installed. Please install it to read RDS files.")
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
                ui.update_checkbox_group("varSelect", 
                                      choices=df.columns.tolist(), 
                                      selected=df.columns.tolist())
                data.set(df)
                processing_status.set(f"✓ Successfully loaded {builtin_selected} dataset\n")
            return
            

        if file_info:
            ui.update_select("builtinDataset", selected="None")
            
            file_path = file_info[0]["datapath"]
            file_ext = file_info[0]["name"].split(".")[-1].lower()

            try:
                df = read_dataset(file_path, file_ext)
                
         
                ui.update_checkbox_group("varSelect", 
                                      choices=df.columns.tolist(), 
                                      selected=df.columns.tolist())
                
                data.set(df)
                
                processing_status.set(f"✓ Successfully loaded {file_ext.upper()} file\n")
                
            except Exception as e:
                processing_status.set(f"❌ Error: {str(e)}")
                return

    @reactive.effect
    @reactive.event(input.processData)
    def process_data():
        """ Process data: keep selected variables and handle missing values dynamically """
        df = data.get()
        if df is None or df.empty:
            processing_status.set("❌ Data not loaded properly")
            return

        status_messages = [] 
        
        selected_vars = input.varSelect()
        valid_vars = [col for col in selected_vars if col in df.columns]
        if not valid_vars:
            status_messages.append("⚠️ No valid variables selected, keeping all variables")
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

        # Additional Data Processing Steps
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

{'=' * 40}
AVAILABLE COLUMNS
{'=' * 40}

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

# Run Shiny App
app = App(app_ui, server)

print("✅ Shiny App started, visit: http://127.0.0.1:8000")
