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
            dcc.Graph(id='cost-distribution-pie', style={'display': 'inline-block', 'width': '48%'}),  # Moved pie chart up
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
            Output('cost-histogram', 'figure')  # Add output for histogram
        ],
        [Input('product-filter', 'value')]
    )
    def update_charts(selected_products):
        # Handle case where no products are selected
        if selected_products:
            filtered_df = df[df['Product type'].apply(lambda x: x in selected_products)]
        else:
            filtered_df = df  # Use entire DataFrame if no filter is applied

        # Bar chart for Avg. Cost by Product Type
        cost_by_product_fig = px.bar(
            filtered_df, x='Product type', y='Costs', title="Avg. Costs by Product Type",
            labels={'Costs': 'Avg. Costs'}, color='Product type'
        )
        cost_by_product_fig.update_layout(template='plotly_dark')

        # Scatter plot for Revenue vs. Costs
        revenue_vs_cost_fig = px.scatter(
            filtered_df, x='Revenue generated', y='Costs', color='Product type', title="Revenue vs Costs",
            labels={'Revenue generated': 'Revenue', 'Costs': 'Costs'}, size='Costs', hover_name='Product type'
        )
        revenue_vs_cost_fig.update_layout(template='plotly_dark')

        # Line chart for Avg. Costs by Defect Rates
        cost_by_defects_fig = px.line(
            filtered_df, x='Defect rates', y='Costs', title="Costs by Defect Rates",
            labels={'Defect rates': 'Defect Rates (%)', 'Costs': 'Costs'}, markers=True
        )
        cost_by_defects_fig.update_layout(template='plotly_dark')

        # Bar chart for Costs by Lead Times
        cost_by_lead_times_fig = px.bar(
            filtered_df, x='Lead times', y='Costs', title="Costs by Lead Times",
            labels={'Lead times': 'Lead Times (days)', 'Costs': 'Costs'}, color='Product type'
        )
        cost_by_lead_times_fig.update_layout(template='plotly_dark')

        # Scatter plot for Stock Levels vs Costs
        stock_vs_cost_fig = px.scatter(
            filtered_df, x='Stock levels', y='Costs', title="Stock Levels vs Costs",
            labels={'Stock levels': 'Stock Levels', 'Costs': 'Costs'}, size='Costs', color='Product type'
        )
        stock_vs_cost_fig.update_layout(template='plotly_dark')

        # Pie chart for Cost Distribution by Product Type
        cost_distribution_pie_fig = px.pie(
            filtered_df, names='Product type', values='Costs', title="Cost Distribution by Product Type"
        )
        cost_distribution_pie_fig.update_layout(template='plotly_dark')

        # Histogram for Costs
        cost_histogram_fig = px.histogram(
            filtered_df, x='Costs', title="Distribution of Costs",
            labels={'Costs': 'Costs'}, 
            nbins=30  # You can adjust the number of bins
        )
        cost_histogram_fig.update_layout(template='plotly_dark')

        return (cost_by_product_fig, revenue_vs_cost_fig, cost_by_defects_fig, cost_by_lead_times_fig,
                stock_vs_cost_fig, cost_distribution_pie_fig, cost_histogram_fig)

    return dash_app
