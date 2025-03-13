from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

app_ui = ui.page_fluid(
    ui.h2("Interactive EDA Dashboard"),
    ui.h3("Data Loading"),
    ui.input_file("file1", "Upload CSV File", multiple=False, accept=[".csv"]),
            
    ui.output_ui("filter_ui"),
    ui.hr(),
            
    ui.h3("Visualization Settings"),
    ui.input_select("plot_type", "Select Plot Type:", {
        "histogram": "Histogram",
        "scatter": "Scatter Plot",
        "bar": "Bar Chart",
        "heatmap": "Correlation Heatmap"
    }),
    ui.output_ui("plot_controls"),
    ui.hr(),        
            
    ui.h3("Data Preview"),
    ui.output_data_frame("data_preview"),
            
    ui.output_plot("plot"),
    ui.output_ui("plot_description"),
 
    ui.h3("Numerical Summary"),
    ui.output_data_frame("numerical_summary"),
    ui.h3("Correlation Analysis"),
    ui.output_plot("correlation_plot"),
)


# Define server logic
def server(input, output, session):
    # Initialize reactive values
    current_dataset = reactive.Value(None)
    filtered_dataset = reactive.Value(None)
    
    @reactive.Effect
    def _():
        # Handle file upload
        file_infos = input.file1()
        if file_infos and len(file_infos) > 0:
            file_info = file_infos[0]
            data = pd.read_csv(file_info['datapath'])
            current_dataset.set(data)
            # Removed the problematic line that tried to update sample_data
    
    @reactive.Calc
    def get_data():
        data = current_dataset.get()
        if data is None:
            return None
        return data
    
    @reactive.Calc
    def get_filtered_data():
        data = current_dataset.get()
        if data is None:
            return None
        
        # Apply filters if they exist
        for col in data.columns:
            if hasattr(input, f"filter_{col}"):
                if pd.api.types.is_numeric_dtype(data[col]):
                    range_val = getattr(input, f"filter_{col}")()
                    if range_val and len(range_val) == 2:
                        data = data[(data[col] >= range_val[0]) & (data[col] <= range_val[1])]
                elif pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                    selected = getattr(input, f"filter_{col}")()
                    if selected and len(selected) > 0:
                        data = data[data[col].isin(selected)]
        
        filtered_dataset.set(data)
        return data
    
    @output
    @render.ui
    def filter_ui():
        data = get_data()
        if data is None:
            return ui.TagList()
        
        filter_inputs = ui.TagList(ui.h3("Data Filters"))
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                filter_inputs.append(
                    ui.input_slider(f"filter_{col}", f"Filter {col}:", 
                                   min=min_val, max=max_val, 
                                   value=[min_val, max_val])
                )
            elif pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                unique_vals = data[col].dropna().unique().tolist()
                if len(unique_vals) < 10:  # Only create filter for categorical with reasonable number of values
                    choices = {str(val): str(val) for val in unique_vals}
                    filter_inputs.append(
                        ui.input_checkbox_group(f"filter_{col}", f"Filter {col}:", 
                                              choices=choices)
                    )
        
        return filter_inputs
    
    @output
    @render.ui
    def plot_controls():
        data = get_data()
        if data is None:
            return ui.TagList()
        
        plot_type = input.plot_type()
        controls = ui.TagList()
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = data.columns.tolist()
        
        if plot_type == "histogram":
            controls.append(ui.input_select("hist_col", "Select Column:", 
                                          {col: col for col in numeric_cols}))
            controls.append(ui.input_slider("hist_bins", "Number of Bins:", 
                                          min=5, max=50, value=20))
            controls.append(ui.input_checkbox("hist_kde", "Show KDE", value=True))
            if len(cat_cols) > 0:
                controls.append(ui.input_select("hist_hue", "Color by (optional):", 
                                              {"": "None", **{col: col for col in cat_cols}}))
                
        elif plot_type == "scatter":
            controls.append(ui.input_select("scatter_x", "X-axis:", 
                                          {col: col for col in numeric_cols}))
            controls.append(ui.input_select("scatter_y", "Y-axis:", 
                                          {col: col for col in numeric_cols}))
            if len(cat_cols) > 0:
                controls.append(ui.input_select("scatter_hue", "Color by (optional):", 
                                              {"": "None", **{col: col for col in cat_cols}}))
            controls.append(ui.input_checkbox("scatter_regression", "Show Regression Line", value=False))
            
        elif plot_type == "bar":
            controls.append(ui.input_select("bar_x", "X-axis:", 
                                          {col: col for col in all_cols}))
            controls.append(ui.input_select("bar_y", "Y-axis (optional):", 
                                          {"": "Count", **{col: col for col in numeric_cols}}))
            if len(cat_cols) > 0:
                controls.append(ui.input_select("bar_hue", "Color by (optional):", 
                                              {"": "None", **{col: col for col in cat_cols}}))
                
        elif plot_type == "heatmap":
            # No additional controls needed for correlation heatmap
            pass
            
        return controls
    
    @output
    @render.data_frame
    def data_preview():
        data = get_filtered_data()
        if data is None:
            return pd.DataFrame()
        
        return render.DataGrid(data, width="100%", height="400px")
    
    @output
    @render.plot
    def plot():
        data = get_filtered_data()
        if data is None:
            return plt.figure()

        plot_type = input.plot_type()
        fig = plt.figure(figsize=(10, 6))
        
        try:
            if plot_type == "histogram":
                col = input.hist_col() if hasattr(input, "hist_col") else data.select_dtypes(include=['number']).columns[0]
                bins = input.hist_bins() if hasattr(input, "hist_bins") else 20
                kde = input.hist_kde() if hasattr(input, "hist_kde") else True
                
                sns.histplot(data=data, x=col, bins=bins, kde=kde)

                plt.title(f'Histogram of {col}')
                plt.tight_layout()
                
            elif plot_type == "scatter":
                x_col = input.scatter_x() if hasattr(input, "scatter_x") else data.select_dtypes(include=['number']).columns[0]
                y_col = input.scatter_y() if hasattr(input, "scatter_y") else data.select_dtypes(include=['number']).columns[1] if len(data.select_dtypes(include=['number']).columns) > 1 else data.select_dtypes(include=['number']).columns[0]
                
                regression = input.scatter_regression() if hasattr(input, "scatter_regression") else False
                
                scatter = sns.scatterplot(data=data, x=x_col, y=y_col)
                
                if regression:
                    sns.regplot(data=data, x=x_col, y=y_col, scatter=False, ax=scatter.axes)
                
                plt.title(f'Scatter Plot of {y_col} vs {x_col}')
                plt.tight_layout()
                
            elif plot_type == "bar":
                x_col = input.bar_x() if hasattr(input, "bar_x") else data.select_dtypes(include=['object', 'category']).columns[0] if len(data.select_dtypes(include=['object', 'category']).columns) > 0 else data.columns[0]
                y_col = input.bar_y() if hasattr(input, "bar_y") and input.bar_y() else None
                
                if y_col:
                    sns.barplot(data=data, x=x_col, y=y_col)
                    plt.title(f'Bar Plot of {y_col} by {x_col}')
                else:
                    sns.countplot(data=data, x=x_col)
                    plt.title(f'Count of Records by {x_col}')
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
            elif plot_type == "heatmap":
                corr_data = data.select_dtypes(include=['number']).corr()
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
    def plot_description():
        data = get_filtered_data()
        if data is None:
            return ui.TagList()
        
        plot_type = input.plot_type()
        plot_info = ui.TagList()
        
        try:
            if plot_type == "histogram":
                col = input.hist_col() if hasattr(input, "hist_col") else data.select_dtypes(include=['number']).columns[0]
                description = [
                    ui.h4(f"Histogram Analysis: {col}"),
                    ui.p(f"Mean: {data[col].mean():.4f}"),
                    ui.p(f"Median: {data[col].median():.4f}"),
                    ui.p(f"Standard Deviation: {data[col].std():.4f}"),
                    ui.p(f"Skewness: {data[col].skew():.4f}"),
                    ui.p(f"Kurtosis: {data[col].kurtosis():.4f}")
                ]
                plot_info.extend(description)
                
            elif plot_type == "scatter":
                x_col = input.scatter_x() if hasattr(input, "scatter_x") else data.select_dtypes(include=['number']).columns[0]
                y_col = input.scatter_y() if hasattr(input, "scatter_y") else data.select_dtypes(include=['number']).columns[1] if len(data.select_dtypes(include=['number']).columns) > 1 else data.select_dtypes(include=['number']).columns[0]
                
                correlation = data[[x_col, y_col]].corr().iloc[0, 1]
                
                if hasattr(input, "scatter_regression") and input.scatter_regression():
                    X = sm.add_constant(data[x_col])
                    model = sm.OLS(data[y_col], X).fit()
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
                x_col = input.bar_x() if hasattr(input, "bar_x") else data.select_dtypes(include=['object', 'category']).columns[0] if len(data.select_dtypes(include=['object', 'category']).columns) > 0 else data.columns[0]
                y_col = input.bar_y() if hasattr(input, "bar_y") and input.bar_y() else None
                
                if y_col:
                    grouped = data.groupby(x_col)[y_col].agg(['mean', 'count']).reset_index()
                    description = [
                        ui.h4(f"Bar Plot Analysis: {y_col} by {x_col}"),
                        ui.p(f"Number of groups: {grouped.shape[0]}"),
                        ui.p(f"Highest average: {grouped.loc[grouped['mean'].idxmax(), x_col]} ({grouped['mean'].max():.4f})"),
                        ui.p(f"Lowest average: {grouped.loc[grouped['mean'].idxmin(), x_col]} ({grouped['mean'].min():.4f})")
                    ]
                else:
                    value_counts = data[x_col].value_counts()
                    description = [
                        ui.h4(f"Count Plot Analysis: {x_col}"),
                        ui.p(f"Number of unique values: {value_counts.shape[0]}"),
                        ui.p(f"Most common: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)"),
                        ui.p(f"Least common: {value_counts.index[-1]} ({value_counts.iloc[-1]} occurrences)")
                    ]
                plot_info.extend(description)
                
            elif plot_type == "heatmap":
                corr_data = data.select_dtypes(include=['number']).corr()
                high_corr = corr_data.unstack().sort_values(ascending=False)
                high_corr = high_corr[(high_corr < 1.0) & (high_corr > 0.5)]
                
                description = [
                    ui.h4("Correlation Heatmap Analysis:"),
                    ui.p(f"Number of numeric features: {corr_data.shape[0]}")
                ]
                
                if len(high_corr) > 0:
                    description.append(ui.h5("Strong Positive Correlations (>0.5):"))
                    for idx, corr_val in high_corr.items():
                        description.append(ui.p(f"{idx[0]} & {idx[1]}: {corr_val:.4f}"))
                        
                plot_info.extend(description)
        
        except Exception as e:
            plot_info.append(ui.p(f"Error generating plot description: {str(e)}"))
            
        return plot_info
    
 
    @output
    @render.data_frame
    def numerical_summary():
        data = get_filtered_data()
        if data is None:
            return pd.DataFrame()
        
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] == 0:
            return pd.DataFrame({'Message': ['No numeric columns in dataset']})
        
        summary = numeric_data.describe().transpose()
        summary.insert(0, 'Column Name', data.columns)
        
        return render.DataGrid(summary, width="100%")
    
    @output
    @render.plot
    def correlation_plot():
        data = get_filtered_data()
        if data is None:
            return plt.figure()
        
        numeric_data = data.select_dtypes(include=['number'])
        if numeric_data.shape[1] < 2:
            fig = plt.figure()
            plt.text(0.5, 0.5, "Not enough numeric columns for correlation analysis", 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            return fig
        
        fig = plt.figure(figsize=(10, 8))
        corr = numeric_data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        return fig
       
# Create Shiny app
app = App(app_ui, server)
