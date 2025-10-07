"""
Legal Research AI - Dash Frontend Application

A modern web interface for legal document search using Dash.
Features:
- Text query search
- Document upload and search
- Clean, modern UI design
- Real-time search results
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import requests
import base64
import io
import json
from datetime import datetime
import pandas as pd

# Initialize Dash app with Bootstrap theme for modern styling
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Define the layout
app.layout = dbc.Container([
    # Header Section
    dbc.Row([
        dbc.Col([
            html.H1(
                "Vakalat AI",
                className="text-center mb-3",
                style={
                    "fontSize": "3rem",
                    "fontWeight": "bold",
                    "color": "#2c3e50",
                    "marginTop": "2rem"
                }
            ),
            html.P(
                "Get relevant court cases for any legal query",
                className="text-center mb-4",
                style={
                    "fontSize": "1.1rem",
                    "color": "#34495e",
                    "maxWidth": "600px",
                    "margin": "0 auto"
                }
            )
        ], width=12)
    ]),
    
    # Search Input Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # Search Input
                    dbc.InputGroup([
                        dbc.Input(
                            id="search-input",
                            placeholder="Ask a question",
                            type="text",
                            style={
                                "borderRadius": "8px",
                                "border": "1px solid #e0e0e0",
                                "padding": "12px 16px",
                                "fontSize": "1rem"
                            }
                        ),
                        dbc.Button(
                            [html.I(className="fas fa-paper-plane")],
                            id="search-button",
                            color="primary",
                            style={
                                "borderRadius": "8px",
                                "marginLeft": "8px",
                                "padding": "12px 16px"
                            }
                        )
                    ], className="mb-3"),
                    
                    # File Upload Section
                    dbc.Row([
                        dbc.Col([
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    html.I(className="fas fa-paperclip", style={"marginRight": "8px"}),
                                    'Attach Files'
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '40px',
                                    'lineHeight': '40px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '8px',
                                    'textAlign': 'center',
                                    'cursor': 'pointer',
                                    'color': '#666',
                                    'borderColor': '#e0e0e0'
                                },
                                multiple=False
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                "Search Document",
                                id="search-document-button",
                                color="secondary",
                                outline=True,
                                style={"width": "100%", "height": "40px"}
                            )
                        ], width=6)
                    ])
                ])
            ], style={"border": "none", "boxShadow": "0 2px 10px rgba(0,0,0,0.1)"})
        ], width=10, className="mx-auto")
    ], className="mb-4"),
    
    # Suggested Queries
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button(
                    "Cases relevant to Aadhar Card?",
                    id="suggestion-1",
                    color="light",
                    outline=True,
                    style={
                        "width": "100%",
                        "margin": "5px",
                        "borderRadius": "8px",
                        "border": "1px solid #e0e0e0",
                        "backgroundColor": "white",
                        "color": "#333"
                    }
                ),
                dbc.Button(
                    "When did right to privacy become a fundamental right?",
                    id="suggestion-2",
                    color="light",
                    outline=True,
                    style={
                        "width": "100%",
                        "margin": "5px",
                        "borderRadius": "8px",
                        "border": "1px solid #e0e0e0",
                        "backgroundColor": "white",
                        "color": "#333"
                    }
                )
            ], style={"display": "flex", "gap": "10px", "justifyContent": "center"})
        ], width=10, className="mx-auto")
    ], className="mb-4"),
    
    # Loading indicator
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                children=[html.Div(id="loading-output")],
                type="default"
            )
        ], width=12)
    ]),
    
    # Results Section
    dbc.Row([
        dbc.Col([
            html.Div(id="search-results")
        ], width=12)
    ]),
    
    # Store for uploaded file data
    dcc.Store(id='uploaded-file-data'),
    
    # Store for search type
    dcc.Store(id='search-type', data='query'),
    
    # Powered by Kanoon logo - bottom right
    html.Div([
        html.Img(
            src="/assets/ikanoon6_powered_transparent.png",
            style={
                "height": "40px",
                "width": "auto"
            },
            alt="Powered by Kanoon"
        )
    ], style={
        "position": "fixed",
        "top": "20px",
        "right": "20px",
        "backgroundColor": "white",
        "padding": "8px 12px",
        "borderRadius": "6px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
        "zIndex": "1000"
    })
    
], fluid=True, style={"minHeight": "100vh", "backgroundColor": "#f8f9fa"})

# Callback for search button
@app.callback(
    [Output('search-results', 'children'),
     Output('loading-output', 'children'),
     Output('search-type', 'data')],
    [Input('search-button', 'n_clicks'),
     Input('suggestion-1', 'n_clicks'),
     Input('suggestion-2', 'n_clicks')],
    [State('search-input', 'value')],
    prevent_initial_call=True
)
def perform_search(search_clicks, sugg1_clicks, sugg2_clicks, query_value):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return no_update, no_update, no_update
    
    # Determine which button was clicked and get the query
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'suggestion-1':
        query = "Cases relevant to Aadhar Card?"
    elif button_id == 'suggestion-2':
        query = "When did right to privacy become a fundamental right?"
    else:
        query = query_value
    
    if not query:
        return no_update, no_update, no_update
    
    # Update the search input with the query
    if button_id in ['suggestion-1', 'suggestion-2']:
        return perform_query_search(query), "", "query"
    else:
        return perform_query_search(query), "", "query"

# Callback for document search
@app.callback(
    [Output('search-results', 'children', allow_duplicate=True),
     Output('loading-output', 'children', allow_duplicate=True),
     Output('search-type', 'data', allow_duplicate=True)],
    [Input('search-document-button', 'n_clicks')],
    [State('uploaded-file-data', 'data')],
    prevent_initial_call=True
)
def perform_document_search(n_clicks, file_data):
    if not n_clicks or not file_data:
        return no_update, no_update, no_update
    
    return perform_document_search_api(file_data), "", "document"

# Callback for file upload
@app.callback(
    Output('uploaded-file-data', 'data'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def store_uploaded_file(contents, filename):
    if contents is not None:
        return {
            'contents': contents,
            'filename': filename
        }
    return no_update

def perform_query_search(query):
    """Perform text query search using the backend API"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/search/query",
            params={"query": query, "top_k": 5},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return create_results_display(data, "query")
        else:
            return create_error_display(f"Search failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        return create_error_display(f"Connection error: {str(e)}")

def perform_document_search_api(file_data):
    """Perform document search using the backend API"""
    try:
        # Decode the file content
        content_type, content_string = file_data['contents'].split(',')
        file_content = base64.b64decode(content_string)
        
        # Prepare files for upload
        files = {
            'file': (file_data['filename'], io.BytesIO(file_content))
        }
        
        response = requests.post(
            f"{BACKEND_URL}/search/document",
            files=files,
            data={'top_k': 5},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return create_results_display(data, "document")
        else:
            return create_error_display(f"Document search failed: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        return create_error_display(f"Connection error: {str(e)}")

def create_results_display(data, search_type):
    """Create the results display component"""
    if not data or 'results' not in data:
        return create_error_display("No results found")
    
    results = data['results']
    if not results:
        return create_error_display("No matching documents found")
    
    # Create result cards
    result_cards = []
    for i, result in enumerate(results):
        card = dbc.Card([
            dbc.CardBody([
                html.H5(
                    result.get('case_title', 'Unknown Title'),
                    className="card-title",
                    style={"color": "#2c3e50", "marginBottom": "10px"}
                ),
                html.P([
                    html.Strong("Court: "), result.get('court', 'Unknown Court'), html.Br(),
                    html.Strong("Date: "), result.get('case_date', 'Unknown Date'), html.Br(),
                    html.Strong("Relevance Score: "), 
                    html.Span(
                        f"{result.get('relevance_score', 0):.3f}",
                        style={"color": "#27ae60", "fontWeight": "bold"}
                    ), html.Br(),
                    html.Strong("Match Reason: "), result.get('reason_for_match', 'No reason provided')
                ], style={"marginBottom": "15px"}),
                html.Hr(),
                html.P([
                    html.Strong("Document Preview: "), html.Br(),
                    result.get('document_preview', 'No preview available')
                ], style={"fontSize": "0.9rem", "color": "#666"})
            ])
        ], style={
            "marginBottom": "20px",
            "border": "1px solid #e0e0e0",
            "borderRadius": "8px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
        })
        result_cards.append(card)
    
    # Create summary
    summary_text = f"Found {data.get('total_results', len(results))} results"
    if search_type == "query":
        summary_text += f" for query: '{data.get('query', 'Unknown')}'"
    else:
        summary_text += f" for document: '{data.get('uploaded_file', 'Unknown file')}'"
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4(
                    summary_text,
                    style={"color": "#2c3e50", "marginBottom": "20px", "textAlign": "center"}
                )
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col(result_cards, width=12)
        ])
    ], style={"marginTop": "30px"})

def create_error_display(message):
    """Create an error display component"""
    return dbc.Alert(
        [
            html.I(className="fas fa-exclamation-triangle", style={"marginRight": "10px"}),
            message
        ],
        color="danger",
        style={"marginTop": "20px", "textAlign": "center"}
    )

# Update search input when suggestions are clicked
@app.callback(
    Output('search-input', 'value'),
    [Input('suggestion-1', 'n_clicks'),
     Input('suggestion-2', 'n_clicks')],
    prevent_initial_call=True
)
def update_search_input(sugg1_clicks, sugg2_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'suggestion-1':
        return "Cases relevant to Aadhar Card?"
    elif button_id == 'suggestion-2':
        return "When did right to privacy become a fundamental right?"    
    return no_update

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
