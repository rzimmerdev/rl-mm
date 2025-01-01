import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import re

# JSON mapping for tag labels
TAG_LABELS = {
    "Data": {"simulated": "Simulated Data", "historical": "Historical Data"},
    "StateSpace": {"lob": "Limit Order Book"},
    "ActionSpace": {"bid_ask_levels": "Bid-Ask Levels"},
    "Rewards": {"running_pnl": "Running PnL", "inventory_penalty": "Inventory Penalty"},
}

# Columns for tag processing
data_columns = ["Data", "StateSpace", "ActionSpace", "Rewards", "Algorithm", "MultiAgent", "ModelFree", "Metrics",
                "Benchmark"]


# Helper function to normalize text
def normalize_text(text):
    if pd.isna(text):  # Handle missing values
        return []
    text = str(text).split(",")
    text = [re.sub(r"[^a-zA-Z0-9]+", " ", t) for t in text]
    return [t.strip().lower().replace(" ", "_") for t in text]


# Load and parse the CSV
def parse_csv(file_path):
    df = pd.read_csv(file_path)
    for col in data_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
    # Convert lists to strings for DataTable compatibility
    for col in data_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df


# Initialize Dash app
app = dash.Dash(__name__)

# Load and parse the example CSV
file_path = "references.csv"  # Replace with your file path
data = parse_csv(file_path)

# Layout for the Dash app
app.layout = html.Div([
    html.H1("Research Papers Tagging and Visualization"),

    # Tag Filtering Section
    html.Div([
        dcc.Dropdown(
            id='tag-filter',
            options=[{'label': col, 'value': col} for col in data_columns],
            multi=True,
            placeholder="Filter by columns"
        ),
        dash_table.DataTable(
            id='papers-table',
            columns=[{"name": i, "id": i} for i in data.columns],
            data=data.to_dict('records'),
            style_table={'overflowX': 'auto'},
            page_size=10,
        ),
    ]),

    # Graph and Tag Count Section
    html.Div([
        html.H2("Tag Visualization"),
        dcc.Dropdown(
            id='column-dropdown',
            options=[{'label': col, 'value': col} for col in data_columns],
            placeholder="Select a column for visualization"
        ),
        dcc.RadioItems(
            id='graph-type',
            options=[
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Pie Chart', 'value': 'pie'}
            ],
            value='bar',
            inline=True,
        ),
        dcc.Graph(id='tag-graph'),
        dash_table.DataTable(
            id='tag-summary-table',
            style_table={'overflowX': 'auto'},
        ),
    ])
])


# Callback for filtering the table based on columns
@app.callback(
    Output('papers-table', 'data'),
    Input('tag-filter', 'value')
)
def filter_table(selected_columns):
    if not selected_columns:
        return data.to_dict('records')
    filtered_data = data.copy()
    for col in selected_columns:
        filtered_data = filtered_data[filtered_data[col].apply(lambda x: bool(x))]
    return filtered_data.to_dict('records')


# Callback for generating tag graph and table
@app.callback(
    [Output('tag-graph', 'figure'),
     Output('tag-summary-table', 'data'),
     Output('tag-summary-table', 'columns')],
    [Input('column-dropdown', 'value'),
     Input('graph-type', 'value')]
)
def update_graph_and_table(selected_column, graph_type):
    if not selected_column:
        return {}, [], []

    # Calculate tag counts
    tag_counts = {}
    for tags in data[selected_column]:
        if tags:  # Ensure tags is not empty
            for tag in tags.split(", "):  # Split the string into tags
                tag_label = TAG_LABELS.get(selected_column, {}).get(tag, tag)  # Use label if available
                tag_counts[tag_label] = tag_counts.get(tag_label, 0) + 1

    # Create graph
    df_counts = pd.DataFrame(list(tag_counts.items()), columns=['Tag', 'Count'])
    if graph_type == 'bar':
        fig = px.bar(df_counts, x='Tag', y='Count', title=f"{selected_column} Tag Counts")
    else:  # Pie chart
        fig = px.pie(df_counts, names='Tag', values='Count', title=f"{selected_column} Tag Distribution")

    # Create summary table
    table_data = [{"Tag": tag, selected_column: count} for tag, count in tag_counts.items()]
    table_columns = [{"name": "Tag", "id": "Tag"}, {"name": selected_column, "id": selected_column}]

    return fig, table_data, table_columns


if __name__ == '__main__':
    app.run_server(debug=True)
