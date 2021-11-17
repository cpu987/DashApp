
# Import pandas library. Reference: https://pandas.pydata.org/docs/
import pandas as pd
# Dash apps. Reference https://dash.gallery/Portal/
import dash
from dash import html
# Dash Core Components: components for interactive user interfaces
# https://dash.plotly.com/dash-core-components
from dash import dcc

# Plotly docs: https://plotly.com/python/
import plotly.graph_objects as go
# Plotly express docs https://plotly.com/python/plotly-express/
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
# import statsmodels as sm  # for Lowess to work, import statsmodels.api instead. Reference: https://stackoverflow.com/questions/50607465/python-3-6-attributeerror-module-statsmodels-has-no-attribute-compat
import statsmodels.api as sm
# from statsmodels import lowess
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import statsmodels.formula.api as smf
import plotly.graph_objs as go
from statsmodels.nonparametric.smoothers_lowess import lowess
from dash.dependencies import Input, Output, State
from datetime import datetime
import matplotlib
import matplotlib.dates as dates

# Load data from indexed csv file
df = pd.read_csv('data/HPI_master_indexed_dparsed.csv', index_col=0, parse_dates=True)

# convert 'date' column to datetime format
# Reference https://github.com/codebasics/py/blob/master/pandas/17_ts_to_date_time/pandas_ts_to_date_time.ipynb
# Documentation https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
df['date'] = pd.to_datetime(df['date'])

# set index column
# df.index = pd.to_datetime(df['date'])

# add moving averages with periods 12 and 24 to data frame
# RefL https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
df['MA12'] = df.index_sa.rolling(12).mean()
df['MA24'] = df.index_sa.rolling(24).mean()

# Implementing MACD reference: https://towardsdatascience.com/implementing-macd-in-python-cc9b2280126a
df['MACD'] = df['MA12'] - df['MA24']
df['signal'] = df.MACD.rolling(9).mean()

# compute lowess regression forecast on the HPI seasonally adjusted dataseries
# First construct numerical arrays from columns "date" and "index_sa"
# set x1 axis as numpy array
x1 = df['date'].to_numpy()
# use matplotlib.dates to convert date column to numpy array of numbers
x1_dates = dates.date2num(x1)
# set y1 axis as numpy array of index_sa values
y1 = df['index_sa'].to_numpy()
# Compute a lowess forecast of the data
# https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
# lowess_forecast1 = lowess(exog=x1_dates, endog=y1, frac=0.2)  # creates a series not a dataframe
lowess_forecast1 = sm.nonparametric.lowess(exog=x1_dates, endog=y1, frac=0.7)
print(lowess_forecast1)

# Insert lowess_forecast1 in dataframe
df.insert(0, "forecast", lowess_forecast1[:, 1], allow_duplicates=True)

# Display code for Pandas Dataframe on console
# print(lowess_forecast1)
#df.info()
#print(df.head(10))
#print(df.tail(10))
#######

# https://pandas.pydata.org/docs/reference/api/pandas.Series.to_frame.html
# filtered_frame = pd.Series(filtered) # To do: covert to timeseries dataframe

# ARIMA model
#How to differencing data if there is a trend in original data?
#from statsmodels.tsa.statespace.tools import diff
#diffprices=diff(prices, k_diff=1, k_seasonal_diff=None, seasonal_periods=1)
# reference
#######


# Drop Rows with NaN Values in Pandas DataFrame. Reference: https://datatofish.com/dropna/
df.dropna()


# Create the modal window
# Reference: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/modal/
modal = html.Div(
    [
        dbc.Button("INFO", id="open", style={'margin-left': '20px'}),
        dbc.Modal(
            [
                dbc.ModalHeader("Header"),
                dbc.ModalBody("BODY OF MODAL"),
                html.Div("Add instructions:", style={'margin-left': '20px'}),
                dbc.ModalFooter(
                    dbc.Button("CLOSE", id="close", className="ml-auto")
                ),
            ],
            id="modal",
            is_open=False,  # True, False
            size="xl",  # "sm", "lg", "xl"
            backdrop=True,  # True, False use Static for modal to close by clicking on backdrop
            scrollable=True,  # False or True for scrolling text
            centered=True,  # True, False
            fade=True,  # True, False
            style={"color": "#333333"}  # set text color style inside the modal
        ),
    ]
)
# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # Reference: https://bootswatch.com/default/
app.config.suppress_callback_exceptions = True  # suppress callback exception output

# Build the app layout
# Reference: https://www.statworx.com/en/blog/how-to-build-a-dashboard-in-python-plotly-dash-step-by-step-tutorial/
app.layout = html.Div(
    children=[
        # Add html to the app
        html.Br(),

        modal,  # Display the Info modal Button

        html.Br(),

        html.Div(className='row',
                 children=[
                     html.Div(className='four columns div-user-controls',
                              children=[
                                  html.H2('Housing Price Index Dashboard'),
                                  html.P('Housing Price Index for the USA'),
                                  html.P('Select place or region from the dropdown list below.'),
                                  html.Div(
                                      className='div-for-dropdown',
                                      children=[
                                          dcc.Dropdown(id='geo-dropdown',
                                                       options=[{'label': i, 'value': i}
                                                                for i in df['place_name'].unique()],
                                                       value='East North Central Division', # Default selection
                                                       multi=False,
                                                       clearable=False,
                                                       searchable=False,
                                                       style={'backgroundColor': '#1E1E1E'},
                                                       className='geo-dropdown'
                                                       ),

                                      ],
                                      style={'color': '#1E1E1E'})
                              ]
                              ),
                     html.Div(className='eight columns div-for-charts bg-grey',
                              children=[
                                  dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True)
                              ])
                 ])
    ]

)


# Set up the callback function for the INFO modal
# References:
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/modal/
# https://python.plainenglish.io/how-to-create-a-model-window-in-dash-4ab1c8e234d3
# https://www.youtube.com/watch?v=X3OuhqS8ueM
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# Set up the callback function for the graphs
@app.callback(  # Callback header
    Output(component_id='timeseries', component_property='figure'),
    Input(component_id='geo-dropdown', component_property='value'),

)
# function updates graph
def update_graph(selected_geography):
    filtered_hpi = df[df['place_name'] == selected_geography]  # filtered dataframe
    # Create subplots. Reference: https://plotly.com/python/subplots/#customizing-subplot-axes
    # Update subplots. Reference: https://plotly.com/python/creating-and-updating-figures/
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"Housing Price Index for: {selected_geography}", "MACD"))
    # f string contains embedded expression

    # Add new traces. Reference: https://www.kite.com/python/docs/plotly.graph_objs.Figure.add_scatter
    fig.add_scatter(y=filtered_hpi['index_sa'], x=filtered_hpi['date'],
                    line=dict(color='Blue'),
                    name="INDEX SA", row=1, col=1)

    fig.add_scatter(y=filtered_hpi['forecast'], x=filtered_hpi['date'],
                    line=dict(color='Orange'),
                    name="forecast", row=1, col=1)

    fig.add_scatter(y=filtered_hpi['MACD'], x=filtered_hpi['date'],
                    line=dict(color='Purple'),
                    name="MACD", row=2, col=1)

    fig.add_scatter(y=filtered_hpi['signal'], x=filtered_hpi['date'],
                    line=dict(color='Black'),
                    name="signal", row=2, col=1)

    fig.update_layout(plot_bgcolor="Gainsboro")

    return fig


if __name__ == '__main__':
 app.run_server(debug=False)
 # to run the app in a production evironment use gunicorn
