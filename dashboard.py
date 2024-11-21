import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load dataset using pandas
df = pd.read_csv("supply_chain_data.csv")

# Function to integrate Dash into Flask
def add_dashboard(server):
    # Create a Dash app attached to the Flask app
    dash_app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')

    # Layout of the dashboard
    dash_app.layout = html.Div([
        html.H1("Supply Chain Costs Analysis", style={'textAlign': 'center', 'color': '#4B0082', 'fontSize': 40}),
        
        html.Div([
            dcc.Dropdown(
                id='product-filter',
                options=[{'label': product, 'value': product} for product in df['Product type'].unique()],
                value=None,
                placeholder="Select a Product Type",
                multi=True
            )
        ], style={'width': '50%', 'margin': '0 auto'}),

        html.Div([
            dcc.Graph(id='cost-by-product', style={'display': 'inline-block', 'width': '48%'}),
            dcc.Graph(id='revenue-vs-cost', style={'display': 'inline-block', 'width': '48%'}),
        ]),

        html.Div([
            dcc.Graph(id='cost-by-defects', style={'display': 'inline-block', 'width': '48%'}),
            dcc.Graph(id='cost-by-lead-times', style={'display': 'inline-block', 'width': '48%'}),
        ]),

        html.Div([
            dcc.Graph(id='stock-vs-cost', style={'display': 'inline-block', 'width': '48%'}),
            dcc.Graph(id='cost-distribution-pie', style={'display': 'inline-block', 'width': '48%'}),
        ]),

        # Add the histogram for costs
        html.Div([
            dcc.Graph(id='cost-histogram', style={'display': 'inline-block', 'width': '100%'}),
        ]),
    ], style={'backgroundColor': '#f0f0f0', 'padding': '20px'})

    # Callbacks to update charts dynamically
    @dash_app.callback(
        [
            Output('cost-by-product', 'figure'),
            Output('revenue-vs-cost', 'figure'),
            Output('cost-by-defects', 'figure'),
            Output('cost-by-lead-times', 'figure'),
            Output('stock-vs-cost', 'figure'),
            Output('cost-distribution-pie', 'figure'),
            Output('cost-histogram', 'figure')
        ],
        [Input('product-filter', 'value')]
    )
    def update_charts(selected_products):
        # Handle case where no products are selected
        if selected_products:
            filtered_df = df[df['Product type'].apply(lambda x: x in selected_products)]
        else:
            filtered_df = df  # Use entire DataFrame if no filter is applied

        # Create the various figures
        cost_by_product_fig = px.bar(filtered_df, x='Product type', y='Costs', title="Avg. Costs by Product Type", color='Product type')
        revenue_vs_cost_fig = px.scatter(filtered_df, x='Revenue generated', y='Costs', title="Revenue vs Costs")
        cost_by_defects_fig = px.line(filtered_df, x='Defect rates', y='Costs', title="Costs by Defect Rates")
        cost_by_lead_times_fig = px.bar(filtered_df, x='Lead times', y='Costs', title="Costs by Lead Times")
        stock_vs_cost_fig = px.scatter(filtered_df, x='Stock levels', y='Costs', title="Stock Levels vs Costs")
        cost_distribution_pie_fig = px.pie(filtered_df, names='Product type', values='Costs', title="Cost Distribution by Product Type")
        cost_histogram_fig = px.histogram(filtered_df, x='Costs', title="Distribution of Costs", nbins=30)

        # Return updated figures
        return (cost_by_product_fig, revenue_vs_cost_fig, cost_by_defects_fig, cost_by_lead_times_fig,
                stock_vs_cost_fig, cost_distribution_pie_fig, cost_histogram_fig)

    return dash_app
