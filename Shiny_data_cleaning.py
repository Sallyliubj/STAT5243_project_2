from shiny import App, ui, render, reactive
import pandas as pd
import io
import re
from bs4 import BeautifulSoup

# --- UI Design ---
app_ui = ui.page_fluid(
    ui.panel_title("Data Cleaning and Feature Selection Tool - Python Shiny"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("file1", "Select Data File", accept=[".csv", ".xlsx"]),
            ui.input_checkbox_group("varSelect", "Select Variables to Keep:", choices=[]),
            ui.input_select("missingDataOption", "Handle Missing Values:", 
                            choices=["None", "Convert Common Missing Values to NA", "Listwise Deletion", "Mean Imputation", "Mode Imputation"], 
                            selected="None"),
            ui.input_action_button("processData", "Process Data", class_="btn-primary"),
        ),
        ui.card(
            ui.output_table("dataTypesTable"),  # **New: Display Variable Types**
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
        text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Keep only ASCII characters
        return text.strip()

    @reactive.effect
    def update_data():
        """ Read the file, update variable selection, and remove invalid columns """
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
            
            # **Remove invalid columns (Unnamed)**
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # **Clean text data**
            df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)

            # **Remove corrupted rows (detect excessive non-ASCII characters)**
            df = df[~df.applymap(lambda x: isinstance(x, str) and re.search(r'[^\x00-\x7F]', x)).any(axis=1)]

            print("✅ Data successfully loaded:", df.shape)
            print("📌 Variable List:", df.columns.tolist())
    
            # **Update variable selection box**
            ui.update_checkbox_group("varSelect", choices=df.columns.tolist(), selected=df.columns.tolist())
    
            # **Store data**
            data.set(df)
    
        except Exception as e:
            print(f"❌ Error reading file: {e}")

    @reactive.effect
    @reactive.event(input.processData)  
    def process_data():
        """ Process data: keep selected variables and correctly handle missing values """
        df = data.get()
        if df is None or df.empty:
            print("❌ df is empty, data not loaded properly")
            return

        # **Get user-selected variables**
        selected_vars = input.varSelect()

        # **Ensure all selected variables exist in df.columns**
        valid_vars = [col for col in selected_vars if col in df.columns]
        
        if not valid_vars:
            print("⚠️ Selected variables are invalid, keeping all variables")
        else:
            df = df[valid_vars]  # **Keep only selected columns**
        
        # **Handle missing values**
        missing_option = input.missingDataOption()
        if missing_option == "Convert Common Missing Values to NA":
            df.replace(["", "-9", "-99"], pd.NA, inplace=True)
        elif missing_option == "Listwise Deletion":
            df.dropna(inplace=True)
        elif missing_option == "Mean Imputation":
            for col in df.select_dtypes(include=["number"]).columns:
                df[col].fillna(df[col].mean(), inplace=True)
        elif missing_option == "Mode Imputation":  
            for col in df.columns:
                if df[col].isna().sum() > 0:  # **Only fill if there are missing values**
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df[col].fillna(mode_value[0], inplace=True)

        print(f"✅ Data processing complete, final shape: {df.shape}")
        data.set(df)  

    @output
    @render.table
    def dataTable():
        """ Display the data table, showing only selected variables """
        df = data.get()
        if df is None or df.empty:
            print("⚠️ dataTable df is empty")
            return pd.DataFrame()
        return df.head(10)

    @output
    @render.table
    def dataTypesTable():
        """ Display the type of each variable """
        df = data.get()
        if df is None or df.empty:
            print("⚠️ dataTypesTable df is empty")
            return pd.DataFrame()

        # Get variable type information
        dtype_df = pd.DataFrame({
            "Variable Name": df.columns,
            "Data Type": df.dtypes.astype(str)
        })
        return dtype_df


# Run Shiny App
app = App(app_ui, server)

print("✅ Shiny App started, visit: http://127.0.0.1:8000")
#########################################################################
########df  is  the   csv  type  data  we  used  in  following  step.