from shiny import App, ui, render, reactive
import pandas as pd
import io
import re
from bs4 import BeautifulSoup
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# --- Function to Get Built-in Dataset ---
def get_builtin_dataset(name):
    if name == "iris":
        data = load_iris(as_frame=True)
    elif name == "wine":
        data = load_wine(as_frame=True)
    elif name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
    elif name == "diabetes":
        data = load_diabetes(as_frame=True)
    else:
        return None

    df = data.data
    df["target"] = data.target
    return df

# --- UI Design ---
app_ui = ui.page_fluid(
    ui.panel_title("Data Cleaning and Feature Selection Tool - Python Shiny"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("file1", "Select Data File", accept=[".csv", ".xlsx"]),
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
            ui.output_table("dataTypesTable"),
            ui.output_table("dataTable")
        )
    )
)

# --- Server Logic ---
def server(input, output, session):
    data = reactive.Value(None)

    def clean_text(text):
        """ Remove HTML content and keep only ASCII characters """
        if pd.isna(text):
            return text
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()

    @reactive.effect
    def update_data():
        """ Read the file or load built-in dataset """
        if input.builtinDataset() != "None":
            df = get_builtin_dataset(input.builtinDataset())
        else:
            file_info = input.file1()
            if not file_info:
                print("⚠️ No file selected")
                return

            file_path = file_info[0]["datapath"]
            file_ext = file_info[0]["name"].split(".")[-1]

            try:
                if file_ext == "csv":
                    df = pd.read_csv(file_path)
                elif file_ext in ["xls", "xlsx"]:
                    df = pd.read_excel(file_path, sheet_name=0, engine="openpyxl")
                else:
                    print("❌ Unsupported file format:", file_ext)
                    return

                if df.empty:
                    print("❌ Loaded data is empty")
                    return

                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            except Exception as e:
                print(f"❌ Error reading file: {e}")

        ui.update_checkbox_group("varSelect", choices=df.columns.tolist(), selected=df.columns.tolist())
        data.set(df)

    @reactive.effect
    @reactive.event(input.processData)
    def process_data():
        """ Process data: keep selected variables and handle missing values dynamically """
        df = data.get()
        if df is None or df.empty:
            print("❌ df is empty, data not loaded properly")
            return

        selected_vars = input.varSelect()
        valid_vars = [col for col in selected_vars if col in df.columns]

        if not valid_vars:
            print("⚠️ Selected variables are invalid, keeping all variables")
        else:
            df = df.loc[:, valid_vars].copy()

        # Handle missing values
        missing_option = input.missingDataOption()
        if missing_option == "Convert Common Missing Values to NA":
            df.replace(["", "-9", "-99"], pd.NA, inplace=True)
        elif missing_option == "Listwise Deletion":
            df = df.dropna()
        elif missing_option == "Mean Imputation":
            for col in df.select_dtypes(include=["number"]).columns:
                df.loc[:, col] = df[col].fillna(df[col].mean())
        elif missing_option == "Mode Imputation":
            for col in df.columns:
                if df[col].isna().sum() > 0:
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df.loc[:, col] = df[col].fillna(mode_value[0])

        # Additional Data Processing Steps
        selected_processing_options = input.dataProcessingOptions()
        if "Remove Duplicates" in selected_processing_options:
            df = df.drop_duplicates()
        if "Standardize Data" in selected_processing_options:
            scaler = StandardScaler()
            df[df.select_dtypes(include=["number"]).columns] = scaler.fit_transform(
                df[df.select_dtypes(include=["number"]).columns])
        if "Normalize Data" in selected_processing_options:
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=["number"]).columns] = scaler.fit_transform(
                df[df.select_dtypes(include=["number"]).columns])
        if "One-Hot Encoding" in selected_processing_options:
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"✅ Data processing complete, final shape: {df.shape}")
        data.set(df)

    @output
    @render.table
    def dataTable():
        df = data.get()
        if df is None or df.empty:
            print("⚠️ dataTable df is empty")
            return pd.DataFrame()
        return df.head(10)

    @output
    @render.table
    def dataTypesTable():
        df = data.get()
        if df is None or df.empty:
            print("⚠️ dataTypesTable df is empty")
            return pd.DataFrame()

        dtype_df = pd.DataFrame({
            "Variable Name": df.columns,
            "Data Type": df.dtypes.astype(str)
        })
        return dtype_df

# Run Shiny App
app = App(app_ui, server)

print("✅ Shiny App started, visit: http://127.0.0.1:8000")
