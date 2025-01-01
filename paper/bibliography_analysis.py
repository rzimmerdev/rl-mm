import re

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load the cleaned data (replace with your actual path)
data_path = "../src/bibliography/references.csv"  # Save the earlier output to this path
data = pd.read_csv(data_path)

# Select custom fields
custom_fields = [
    "StateSpace", "ActionSpace", "Rewards", "Algorithm",
    "MultiAgent", "ModelFree", "Metrics", "Benchmark"
]

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Bibliography Tag Analysis"),
    html.Label("Select Field:"),
    dcc.Dropdown(
        id="field-dropdown",
        options=[{"label": field, "value": field} for field in custom_fields],
        value="StateSpace"  # Default field
    ),
    html.Label("Select Chart Type:"),
    dcc.RadioItems(
        id="chart-type",
        options=[
            {"label": "Pie Chart", "value": "pie"},
            {"label": "Vertical Bar Chart", "value": "bar_v"},
            {"label": "Horizontal Bar Chart", "value": "bar_h"}
        ],
        value="pie"
    ),
    html.Label("Stacked (only for bar charts):"),
    dcc.Checklist(
        id="stacked-checklist",
        options=[{"label": "Stacked", "value": "stacked"}],
        value=[]
    ),
    html.Label("Group Others (for Pie Chart) Percentage:"),
    dcc.Slider(
        id="group-slider",
        min=0,
        max=100,
        step=1,
        value=10,  # Default value for the slider
        marks={i: f"{i}%" for i in range(0, 101, 10)},
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    html.Label("Graph Width:"),
    dcc.Slider(
        id="width-slider",
        min=400,
        max=1200,
        step=10,
        value=800,  # Default width value
        marks={i: f"{i}" for i in range(400, 1201, 200)},
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    html.Label("Graph Height:"),
    dcc.Slider(
        id="height-slider",
        min=400,
        max=800,
        step=10,
        value=600,  # Default height value
        marks={i: f"{i}" for i in range(400, 801, 200)},
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    html.Label("Custom Title:"),
    dcc.Input(id="custom-title", type="text", placeholder="Enter chart title", value="Bibliography Tag Analysis"),
    html.Label("X-Axis Label:"),
    dcc.Input(id="x-axis-label", type="text", placeholder="Enter X-axis label", value="Tags"),
    html.Label("Y-Axis Label:"),
    dcc.Input(id="y-axis-label", type="text", placeholder="Enter Y-axis label", value="Count"),
    html.Label("Color Palette:"),
    dcc.Dropdown(
        id="color-palette",
        options=[
            {"label": "Plotly", "value": "plotly"},
            *[{"label": k, "value": k} for k in dir(px.colors.qualitative) if not k.startswith("_")]
        ],
        value="plotly"  # Default palette
    ),
    html.Label("Custom Labels (JSON):"),
    dcc.Textarea(
        id="custom-labels",
        value='{}',  # Default is empty JSON
        style={"width": "100%", "height": "100px"}
    ),
    html.Label("Enable 3D Tilt for Pie Chart:"),
    dcc.Checklist(
        id="pie-3d-checklist",
        options=[{"label": "3D Tilt", "value": "3d"}],
        value=[]
    ),
    html.Label("Chart Theme:"),
    dcc.Dropdown(
        id="theme-dropdown",
        options=[
            {"label": "Plotly", "value": "plotly"},
            {"label": "ggplot2", "value": "ggplot2"},
            {"label": "seaborn", "value": "seaborn"},
            {"label": "simple_white", "value": "simple_white"},
            {"label": "none", "value": "none"},
        ],
        value="plotly"  # Default theme
    ),
    dcc.Graph(id="tag-chart"),
])

from plotly.colors import qualitative

import json


@app.callback(
    Output("tag-chart", "figure"),
    [
        Input("field-dropdown", "value"),
        Input("chart-type", "value"),
        Input("stacked-checklist", "value"),
        Input("group-slider", "value"),
        Input("width-slider", "value"),
        Input("height-slider", "value"),
        Input("custom-labels", "value"),
        Input("custom-title", "value"),
        Input("x-axis-label", "value"),
        Input("y-axis-label", "value"),
        Input("pie-3d-checklist", "value"),
        Input("color-palette", "value"),
        Input("theme-dropdown", "value"),
    ]
)
def update_chart(selected_field, chart_type, stacked_value, slider_value, graph_width, graph_height,
                 custom_labels_json, custom_title, x_label, y_label, pie_3d_value, color_palette, theme):
    # Parse the JSON input for custom labels
    try:
        custom_labels = json.loads(custom_labels_json)
    except json.JSONDecodeError:
        custom_labels = {}

    # Count occurrences for each tag value
    exploded = (
        data[selected_field]
        .dropna()  # Remove NaN values
        .str.split(",")  # Split by commas
        .explode()  # Flatten the lists
        .str.strip()  # Remove leading/trailing spaces
        .str.lower()  # Convert to lowercase
        .apply(lambda x: re.sub(r"[^a-z0-9\s]", "", x))  # Remove special characters
    )
    exploded = exploded[exploded != ""]
    value_counts = exploded.value_counts().reset_index()
    value_counts.columns = [selected_field, "Count"]

    # Apply custom labels1
    value_counts[selected_field] = value_counts[selected_field].apply(
        lambda x: custom_labels.get(x, x)
    )

    # Color palette selection
    palettes = {
        "plotly": qualitative.Plotly,
    }
    palettes.update({k: getattr(px.colors.qualitative, k) for k in dir(px.colors.qualitative) if not k.startswith("_")})

    color_sequence = palettes.get(color_palette, qualitative.Plotly)

    # Chart generation
    if chart_type == "pie":
        total_count = value_counts["Count"].sum()
        threshold = total_count * (slider_value / 100)
        value_counts = value_counts.sort_values(by="Count", ascending=True)
        others_count = 0
        grouped = []
        for index, row in value_counts.iterrows():
            others_count += row["Count"]
            if others_count <= threshold:
                grouped.append([row[selected_field], row["Count"]])
            else:
                grouped.append(["Others", others_count])
                break

        if others_count > 0:
            grouped_df = pd.DataFrame(grouped, columns=[selected_field, "Count"])
            data_source = grouped_df
        else:
            data_source = value_counts

        # Create pie chart
        fig = px.pie(
            data_source,
            names=selected_field,
            values="Count",
            title=custom_title,
            color_discrete_sequence=color_sequence,
            template=theme
        )

        # Apply 3D effect if enabled
        if "3d" in pie_3d_value:
            fig.update_traces(
                pull=[0.1] * len(data_source),  # Separate slices slightly
                rotation=45,  # Rotate for visual effect
                textinfo="label+percent",  # Include labels and percentages
            )
        else:
            fig.update_traces(textinfo="label+percent")

        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            width=graph_width,
            height=graph_height
        )
    elif chart_type == "bar_v":
        fig = px.bar(
            value_counts,
            x=selected_field,  # This must match the column name for the x-axis
            y="Count",  # This must match the column name for the y-axis
            text="Count",
            title=custom_title,
            labels={selected_field: x_label, "Count": y_label},  # Correctly map DataFrame columns
            barmode="stack" if "stacked" in stacked_value else "group",
            color_discrete_sequence=color_sequence,
            template=theme
        )
    else:
        fig = px.bar(
            value_counts,
            x="Count",
            y=selected_field,
            text="Count",
            title=custom_title,
            labels={"x": y_label, "y": x_label},
            barmode="stack" if "stacked" in stacked_value else "group",
            color_discrete_sequence=color_sequence,
            template=theme
        )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        margin=dict(l=40, r=40, t=40, b=40),
        width=graph_width,
        height=graph_height
    )
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
