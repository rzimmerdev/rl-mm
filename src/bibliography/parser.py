import json
import dash
from dash import dcc, html, Input, Output, dash_table, State
import pandas as pd
import plotly.express as px
import re

# read labels.json
TAG_LABELS = json.load(open("labels.json"))

data_columns = ["Data", "StateSpace", "ActionSpace", "Rewards", "Algorithm", "MultiAgent", "ModelFree", "Metrics",
                "Benchmark"]

templates = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]


def normalize_text(text):
    if pd.isna(text):
        return []
    text = str(text).split(",")
    text = [re.sub(r"[^a-zA-Z0-9]+", " ", t) for t in text]
    return [t.strip().lower().replace(" ", "_") for t in text]


def parse_csv(file_path):
    df = pd.read_csv(file_path)
    for col in data_columns:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)
    for col in data_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    return df


app = dash.Dash(__name__)

file_path = "references.csv"
data = parse_csv(file_path)
app.layout = html.Div([
    html.H1("Research Papers Tagging and Visualization"),

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

    html.Div([
        html.H2("Tag Visualization"),
        dcc.Input(
            id='chart-title',
            type='text',
            placeholder="Enter Chart Title",
            style={'marginBottom': '10px', 'width': '50%'}
        ),
        dcc.Dropdown(
            id='chart-template',
            options=[{"label": name, "value": name} for name in templates],
            placeholder="Select Chart Template",
            style={'marginBottom': '10px', 'width': '50%'}
        ),
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
        html.Div(
            id='axis-inputs',
            children=[
                dcc.Input(id='x-axis-label', type='text', placeholder="Enter X-axis Label",
                          style={'marginRight': '10px'}),
                dcc.Input(id='y-axis-label', type='text', placeholder="Enter Y-axis Label")
            ],
            style={'marginBottom': '10px', 'display': 'none'}
        ),
        html.Div([
            dcc.Input(
                id='graph-width',
                type='number',
                placeholder="Graph Width (px)",
                style={'marginRight': '10px', 'width': '20%'}
            ),
            dcc.Input(
                id='graph-height',
                type='number',
                placeholder="Graph Height (px)",
                style={'marginRight': '10px', 'width': '20%'}
            ),
            dcc.Input(
                id='font-size',
                type='number',
                placeholder="Font Size (px)",
                style={'width': '20%'}
            ),
            dcc.Input(
                id='bar-label-angle',
                type='number',
                placeholder="Bar Label Angle (degrees)",
                style={'width': '20%'}
            )
        ], style={'marginBottom': '10px'}),
        dcc.Graph(id='tag-graph'),
        dash_table.DataTable(
            id='tag-summary-table',
            style_table={'overflowX': 'auto'},
        ),
    ])
])


@app.callback(
    Output('axis-inputs', 'style'),
    Input('graph-type', 'value')
)
def toggle_axis_inputs(graph_type):
    """Show or hide axis inputs based on the selected graph type."""
    if graph_type == 'bar':
        return {'marginBottom': '10px', 'display': 'block'}
    return {'marginBottom': '10px', 'display': 'none'}


@app.callback(
    [Output('tag-graph', 'figure'),
     Output('tag-summary-table', 'data'),
     Output('tag-summary-table', 'columns')],
    [Input('column-dropdown', 'value'),
     Input('graph-type', 'value'),
     Input('chart-title', 'value'),
     Input('chart-template', 'value'),
     Input('x-axis-label', 'value'),
     Input('y-axis-label', 'value'),
     Input('graph-width', 'value'),
     Input('graph-height', 'value'),
     Input('font-size', 'value'),
     Input('bar-label-angle', 'value')]
)
def update_graph_and_table(selected_column, graph_type, chart_title, chart_template,
                           x_axis_label, y_axis_label, graph_width, graph_height, font_size, bar_label_angle):
    if not selected_column:
        return {}, [], []

    tag_counts = {}
    for tags in data[selected_column]:
        if tags:
            for tag in tags.split(", "):
                tag_label = TAG_LABELS.get(selected_column, {}).get(tag, tag)
                tag_counts[tag_label] = tag_counts.get(tag_label, 0) + 1
    df_counts = pd.DataFrame(list(tag_counts.items()), columns=['Tag', 'Count'])
    # remove empty
    df_counts = df_counts[df_counts['Tag'] != '']
    df_counts = df_counts.sort_values(by='Count', ascending=False)

    # Apply the selected chart type and additional customization
    if graph_type == 'bar':
        fig = px.bar(
            df_counts,
            x='Tag',
            y='Count',
            title=chart_title or f"{selected_column} Tag Counts",
            template=chart_template or "plotly",
            labels={'Tag': x_axis_label or 'Tag', 'Count': y_axis_label or 'Count'}
        )
    else:
        fig = px.pie(
            df_counts,
            names='Tag',
            values='Count',
            title=chart_title or f"{selected_column} Tag Distribution",
            template=chart_template or "plotly"
        )

    # Resize the graph and update font size if provided
    fig.update_layout(
        width=graph_width if graph_width else None,
        height=graph_height if graph_height else None,
        font=dict(size=font_size if font_size else 12)  # Default font size is 12
    )

    # If bar chart, update x-axis label rotation (angle)
    if graph_type == 'bar' and bar_label_angle is not None:
        fig.update_layout(
            xaxis_tickangle=bar_label_angle  # Apply the angle to the bar labels
        )

    table_data = [{"Tag": tag, selected_column: count} for tag, count in tag_counts.items()]
    table_columns = [{"name": "Tag", "id": "Tag"}, {"name": selected_column, "id": selected_column}]

    return fig, table_data, table_columns


if __name__ == '__main__':
    app.run_server(debug=True)
