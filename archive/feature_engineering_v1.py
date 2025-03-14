import pandas as pd
import io
import numpy as np
from datetime import datetime
from scipy import stats
from shiny import App, ui, render, reactive

# DEFINE UI
app_ui = ui.page_fluid(
    ui.input_file("file", "upload", multiple=False, accept=[".csv"]),
    ui.input_action_button("load_columns", "Getting Column Names"),
    ui.input_select("column", "Column Select", choices=[]),
    ui.output_table("preview_column"),
    ui.input_select("operation", "Selecting Feature Engineering Operations", choices=[
        "Normalize", "One-Hot", "Date", "Box-Cox"
    ]),
    ui.input_action_button("apply_transform", "Apply Feature Engineering"),
    ui.input_numeric("input_year", "Input year（YYYY）", value=2025),
    ui.input_numeric("input_month", "Input month（MM）", value=3),
    ui.input_numeric("input_day", "Input day（DD）", value=7),
    ui.input_selectize("multi_columns", "Select Two Column for Create meaningful New Feature", choices=[], multiple=True),
    ui.input_select("extra_operation", "Meaningful New Feature", choices=[
        "None", "Average", "Interactions"
    ]),
    ui.input_action_button("apply_extra_transform", "Apply New Feature"),
    ui.output_table("table")
)

def server(input, output, session):
    # LOAD DATA
    @reactive.Calc
    def get_data():
        file = input.file()
        if not file:
            return None
        file_path = file[0]["datapath"]
        return pd.read_csv(file_path)

    # SELECTING COLUMN
    @reactive.Effect
    @reactive.event(input.load_columns)
    def update_column_choices():
        df = get_data()
        if df is not None:
            ui.update_select("column", choices=df.columns.tolist())
            ui.update_selectize("multi_columns", choices=df.columns.tolist())

    # DISPLAY DATA
    @output
    @render.table
    def preview_column():
        df = get_data()
        if df is None:
            return pd.DataFrame()
        column = input.column()
        if column not in df.columns:
            return pd.DataFrame()
        return df[[column]].head(10)

    
    @reactive.Calc
    @reactive.event(input.apply_transform)
    def transformed_data():
        df = get_data()
        if df is None:
            return None

        column = input.column()
        operation = input.operation()

        if column not in df.columns:
            return df

        if operation == "Normalize":
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

        elif operation == "One-Hot":
            df = pd.get_dummies(df, columns=[column])

        elif operation == "Date":
            df[column] = pd.to_datetime(df[column], errors="coerce")
            df[f"{column}_year"] = df[column].dt.year
            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_day"] = df[column].dt.day

            input_year = input.input_year()
            input_month = input.input_month()
            input_day = input.input_day()

            try:
                input_date = datetime(input_year, input_month, input_day)
            except ValueError:
                input_date = datetime.today()

            df[f"{column}_days_since_input_date"] = (input_date - df[column]).dt.days
            df.drop(columns=[column], inplace=True)

        elif operation == "Box-Cox":
            df[column] = pd.to_numeric(df[column], errors="coerce")
            if df[column].min() <= 0:
                df[column] += abs(df[column].min()) + 1  

            df[column], _ = stats.boxcox(df[column])

        return df

    # NEW FEATURE
    @reactive.Calc
    @reactive.event(input.apply_extra_transform)
    def extra_transformed_data():
        df = transformed_data()
        if df is None:
            return None
            
        selected_columns = list(input.multi_columns()) 
        extra_operation = input.extra_operation()

        if len(selected_columns) < 2:
            return df  

        if extra_operation == "Average":
            df["weighted_avg"] = df[selected_columns].mean(axis=1)

        elif extra_operation == "Interactions":
            col1, col2 = selected_columns[:2]
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        return df

    # DISPLAY AGAIN
    @output
    @render.table
    def table():
        df = extra_transformed_data()
        if df is None:
            return pd.DataFrame()
        return df.head()

# Shiny App
app = App(app_ui, server)





