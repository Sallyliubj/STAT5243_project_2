import pandas as pd
import io
import numpy as np
from datetime import datetime
from scipy import stats
from shiny import App, ui, render, reactive
import seaborn as sns
import matplotlib.pyplot as plt

# DEFINE UI
app_ui = ui.page_fluid(
    ui.input_file("file", "upload", multiple=False, accept=[".csv"]),
    ui.input_action_button("load_columns", "Getting Column Names"),
    ui.input_select("column", "Column Select", choices=[]),
    ui.output_table("preview_column"),
    ui.input_action_button("apply_transform", "Apply EDA"),
    ui.output_table("descriptive_stats"),
    ui.output_plot('plot_histogram'),
    ui.output_plot('plot_correlation_matrix')
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
        
    @output
    @render.table
    def descriptive_stats():
        df = get_data()
        if df is None:
            return pd.DataFrame()
        return df.describe()

    @output
    @render.plot
    def plot_histogram():
        df = get_data()
        column = input.column()
        p = sns.histplot(df, x=column)
        return p

    @output
    @render.plot
    def plot_correlation_matrix():
        df = get_data()

        corr = df.corr()
        plt.figure(figsize=(10, 8))
        p = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        return p
               
# Shiny App
app = App(app_ui, server)