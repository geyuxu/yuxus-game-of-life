#!/usr/bin/env python3
"""
Web interface for the neuroevolution simulation using Plotly Dash.

Features:
- Real-time environment visualization
- Chemical field heatmaps (4 chemical types)
- Population dynamics charts
- Species statistics panel
- Interactive controls
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from threading import Lock
import colorsys

# Import simulation from main.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import GPULifeGame, SPECIES_CONFIG, NUM_CHEMICALS

# =============================================================================
# GLOBAL STATE
# =============================================================================
game = None
game_lock = Lock()
simulation_running = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def rgb_to_hex(rgb):
    """Convert RGB tuple (0-1 range) to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

def create_environment_figure(game):
    """Create the main environment visualization."""
    render = game.render()

    fig = go.Figure(data=go.Heatmap(
        z=render,
        colorscale='Viridis',
        showscale=False,
        hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Environment - Generation {game.generation}',
        xaxis={'visible': False},
        yaxis={'visible': False},
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font=dict(color='white')
    )

    return fig

def create_chemical_heatmaps(game):
    """Create 4 chemical field heatmaps."""
    chemicals_cpu = game.chemicals.cpu().numpy()

    # Create 2x2 subplot grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'Chemical {i}' for i in range(NUM_CHEMICALS)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    for i in range(NUM_CHEMICALS):
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(
            go.Heatmap(
                z=chemicals_cpu[i],
                colorscale='Hot',
                showscale=True if i == 3 else False,
                colorbar=dict(x=1.05) if i == 3 else None,
                hovertemplate=f'Chemical {i}<br>x: %{{x}}<br>y: %{{y}}<br>concentration: %{{z:.3f}}<extra></extra>'
            ),
            row=row, col=col
        )

        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        title='Chemical Field Distribution',
        height=500,
        showlegend=False,
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#1e1e1e',
        font=dict(color='white')
    )

    return fig

def create_population_chart(game):
    """Create population dynamics chart."""
    fig = go.Figure()

    # Add traces for each species
    history_len = len(game.history['population'])
    gens = list(range(history_len))

    for sp_id in range(len(SPECIES_CONFIG)):
        if SPECIES_CONFIG[sp_id].get('extinct', False):
            continue

        if sp_id < len(game.history['species']):
            color = SPECIES_CONFIG[sp_id]['color']
            name = SPECIES_CONFIG[sp_id]['name']

            fig.add_trace(go.Scatter(
                x=gens,
                y=game.history['species'][sp_id],
                mode='lines',
                name=name,
                line=dict(color=rgb_to_hex(color), width=2),
                hovertemplate=f'{name}<br>Gen: %{{x}}<br>Pop: %{{y}}<extra></extra>'
            ))

    # Add total population
    fig.add_trace(go.Scatter(
        x=gens,
        y=game.history['population'],
        mode='lines',
        name='Total',
        line=dict(color='white', width=3, dash='dash'),
        hovertemplate='Total<br>Gen: %{x}<br>Pop: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title='Population Dynamics',
        xaxis_title='Generation',
        yaxis_title='Population',
        height=400,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(30, 30, 30, 0.8)"
        ),
        paper_bgcolor='#1e1e1e',
        plot_bgcolor='#2e2e2e',
        font=dict(color='white'),
        xaxis=dict(gridcolor='#444444'),
        yaxis=dict(gridcolor='#444444')
    )

    return fig

def create_species_stats_table(game):
    """Create species statistics table."""
    total = game.history['population'][-1] if game.history['population'] else 0

    species_data = []
    for sp_id in range(len(SPECIES_CONFIG)):
        if SPECIES_CONFIG[sp_id].get('extinct', False):
            continue

        if sp_id < len(game.history['species']) and game.history['species'][sp_id]:
            count = game.history['species'][sp_id][-1]
        else:
            count = 0

        if count == 0:
            continue

        name = SPECIES_CONFIG[sp_id]['name']
        pct = (count / total * 100) if total > 0 else 0
        color = rgb_to_hex(SPECIES_CONFIG[sp_id]['color'])
        hidden_size = SPECIES_CONFIG[sp_id]['hidden_size']

        species_data.append({
            'name': name,
            'count': count,
            'pct': pct,
            'color': color,
            'hidden': hidden_size
        })

    # Sort by population
    species_data.sort(key=lambda x: x['count'], reverse=True)

    # Create table rows
    rows = []
    for data in species_data:
        rows.append(html.Tr([
            html.Td(html.Div(style={
                'backgroundColor': data['color'],
                'width': '20px',
                'height': '20px',
                'border': '1px solid white'
            })),
            html.Td(data['name'], style={'font-family': 'monospace'}),
            html.Td(f"{data['count']:,}", style={'text-align': 'right'}),
            html.Td(f"{data['pct']:.1f}%", style={'text-align': 'right'}),
            html.Td(f"{data['hidden']}", style={'text-align': 'center'})
        ], style={'borderBottom': '1px solid #444'}))

    return rows

# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title='Neuroevolution Arena'
)

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1('🧬 Neuroevolution Arena', className='text-center text-light mb-3 mt-3'),
            html.P('GPU-Accelerated Artificial Life Simulation with Chemical Signaling',
                   className='text-center text-secondary mb-4')
        ])
    ]),

    # Stats bar
    dbc.Row([
        dbc.Col([
            html.Div(id='stats-bar', className='text-center text-light mb-3',
                    style={'fontSize': '18px', 'fontFamily': 'monospace'})
        ])
    ]),

    # Main content - 3 columns
    dbc.Row([
        # Left column: Environment + Population chart
        dbc.Col([
            dcc.Graph(id='environment-graph', style={'marginBottom': '20px'}),
            dcc.Graph(id='population-graph')
        ], width=5),

        # Middle column: Chemical heatmaps
        dbc.Col([
            dcc.Graph(id='chemical-graphs')
        ], width=4),

        # Right column: Species statistics
        dbc.Col([
            html.Div([
                html.H5('Species Statistics', className='text-light mb-3'),
                html.Div(id='species-table', style={
                    'overflowY': 'scroll',
                    'height': '900px',
                    'backgroundColor': '#2e2e2e',
                    'padding': '10px',
                    'borderRadius': '5px'
                })
            ])
        ], width=3)
    ]),

    # Control buttons
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button('Pause', id='pause-button', color='warning', className='me-2'),
                dbc.Button('Reset', id='reset-button', color='danger', className='me-2'),
                dbc.Button('Save Snapshot', id='save-button', color='success')
            ], className='d-flex justify-content-center mt-3 mb-3')
        ])
    ]),

    # Update interval
    dcc.Interval(
        id='interval-component',
        interval=100,  # milliseconds
        n_intervals=0
    ),

    # Store for simulation state
    dcc.Store(id='simulation-state', data={'running': True})

], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh'})

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    [Output('environment-graph', 'figure'),
     Output('chemical-graphs', 'figure'),
     Output('population-graph', 'figure'),
     Output('species-table', 'children'),
     Output('stats-bar', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('simulation-state', 'data')]
)
def update_graphs(n, state):
    """Update all graphs and statistics."""
    global game

    if game is None:
        # Initialize game on first call
        with game_lock:
            game = GPULifeGame()

    # Run simulation step if running
    if state.get('running', True):
        with game_lock:
            game.step()

    # Create figures
    with game_lock:
        env_fig = create_environment_figure(game)
        chem_fig = create_chemical_heatmaps(game)
        pop_fig = create_population_chart(game)
        species_rows = create_species_stats_table(game)

        # Stats bar
        total_pop = game.history['population'][-1] if game.history['population'] else 0
        alive_species = sum(1 for sp in SPECIES_CONFIG if not sp.get('extinct', False))

        stats = f"Gen: {game.generation:,} | Species: {alive_species} | Population: {total_pop:,}"

    # Create species table
    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th('', style={'width': '30px'}),
            html.Th('Name'),
            html.Th('Count', style={'text-align': 'right'}),
            html.Th('%', style={'text-align': 'right'}),
            html.Th('NN', style={'text-align': 'center', 'width': '50px'})
        ])),
        html.Tbody(species_rows)
    ], dark=True, striped=True, bordered=True, hover=True, size='sm',
       style={'fontSize': '12px'})

    return env_fig, chem_fig, pop_fig, table, stats

@app.callback(
    Output('simulation-state', 'data'),
    [Input('pause-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('simulation-state', 'data')],
    prevent_initial_call=True
)
def control_simulation(pause_clicks, reset_clicks, state):
    """Handle control button clicks."""
    global game

    ctx = dash.callback_context
    if not ctx.triggered:
        return state

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'pause-button':
        state['running'] = not state.get('running', True)
    elif button_id == 'reset-button':
        with game_lock:
            game = GPULifeGame()
        state['running'] = True

    return state

@app.callback(
    Output('pause-button', 'children'),
    [Input('simulation-state', 'data')]
)
def update_pause_button(state):
    """Update pause button label."""
    return 'Resume' if not state.get('running', True) else 'Pause'

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Neuroevolution Arena Web Interface")
    print("=" * 60)
    print("\nOpen your browser and navigate to:")
    print("    http://localhost:8050")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    app.run_server(debug=False, host='0.0.0.0', port=8050)
