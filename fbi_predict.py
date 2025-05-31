import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import joblib

# Load the trained model
with open("xgboost_model.joblib", "rb") as f:
    model = joblib.load(f)

# Load the predictions dataset
df = pd.read_csv("test_predictions.csv")

# Create a simple visualization
fig = px.line(df, x="Date", y="Predicted_Incident_Counts", title="Crime Prediction Over Time")

# Initialize Dash App
app = dash.Dash(__name__)

# Layout of the Dashboard
app.layout = html.Div(children=[
    html.H1("Crime Prediction Dashboard"),
    dcc.Graph(figure=fig)
])

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
