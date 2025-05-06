# app.py
import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import plotly.express as px

# Load model components
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/scoring_features.pkl")

# Session score file
LOG_FILE = "data/session_scores.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["PatientID", "Timestamp", "Score", "Risk", "Source"]).to_csv(LOG_FILE, index=False)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Hospital Risk Intelligence"

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='page-content')
])

navbar = html.Div([
    html.Div([
        html.H2("🏥 Hospital Risk Intelligence Dashboard", 
                style={"textAlign": "center", "color": "#004080", "marginBottom": "5px"})
    ], style={"padding": "20px", "backgroundColor": "#dbeeff", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

    html.Div([
        dcc.Link("🏠 Home", href="/", style={"marginRight": "20px", "textDecoration": "none", "color": "#004080", "fontWeight": "bold"}),
        dcc.Link("🧮 Risk Calculator", href="/calculator", style={"marginRight": "20px", "textDecoration": "none", "color": "#004080", "fontWeight": "bold"}),
        dcc.Link("🩺 In person Data", href="/InPerson", style={"marginRight": "20px", "textDecoration": "none", "color": "#004080", "fontWeight": "bold"}),
        dcc.Link("📡 Remote Monitoring", href="/remote", style={"marginRight": "20px", "textDecoration": "none", "color": "#004080", "fontWeight": "bold"})
    ], style={"textAlign": "center", "padding": "15px", "backgroundColor": "#eaf4ff", "borderBottom": "2px solid #aaccee"})
])

def home_page():
    return html.Div([
        navbar,
        html.Div([
            html.H3("📊 Welcome to the Health Risk Scoring System", style={
                "color": "#003366",
                "marginTop": "20px",
                "textAlign": "center",
                "fontWeight": "bold"
            }),
            html.P("This dashboard helps hospitals triage and monitor patients in real-time based on vitals.",
                   style={
                       "fontSize": "18px",
                       "lineHeight": "1.6",
                       "textAlign": "center",
                       "marginBottom": "20px"
                   }),
            html.Div([
                html.Ul([
                    html.Li("🩺 ER Triage Automation"),
                    html.Li("📊 Inpatient Risk Trends"),
                    html.Li("📡 Remote Patient Monitoring"),
                    html.Li("📋 Post-Op Follow-up Tracking")
                ], style={
                    "fontSize": "16px",
                    "lineHeight": "1.8",
                    "maxWidth": "600px",
                    "margin": "0 auto"
                }),
            ], style={"marginBottom": "30px"}),
            html.Hr(style={
                "border": "none",
                "height": "1px",
                "backgroundColor": "#aaccee",
                "margin": "20px 0"
            }),
            html.H4("📈 Today's Risk Distribution", style={
                "marginTop": "30px",
                "color": "#004080",
                "textAlign": "center"
            }),
            dcc.Graph(id="risk-summary-chart", style={"padding": "0 40px"})
        ], style={
            "padding": "30px",
            "margin": "20px auto",
            "maxWidth": "950px",
            "backgroundColor": "#ffffff",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.05)",
            "borderRadius": "10px"
        })
    ])

# def calculator_page():
#     return html.Div([
#         navbar,
#         html.H3("🧮 Risk Score Calculator", style={"color": "#003366", "textAlign": "center"}),
#         html.Div([
#             html.Label("Patient ID"), dcc.Input(id="pid", type="text", style={"width": "50%"}),
#             html.Label("Height (cm)"), dcc.Input(id="height", type="number", value="", style={"width": "50%"}),
#             html.Label("Weight (kg)"), dcc.Input(id="weight", type="number", style={"width": "50%"}),
#             html.Label("Diastolic BP"), dcc.Input(id="dbp", type="number", style={"width": "50%"}),
#             html.Label("Systolic BP"), dcc.Input(id="sbp", type="number", style={"width": "50%"}),
#             html.Label("Heart Rate"), dcc.Input(id="hr", type="number", style={"width": "50%"}),
#             html.Label("Respiratory Rate"), dcc.Input(id="rr", type="number", style={"width": "50%"}),
#             html.Label("Smoking Status"),
#             dcc.Dropdown(id="smoke", options=[
#                 {"label": "Never", "value": "NO"},
#                 {"label": "Ex-Smoker", "value": "EX"},
#                 {"label": "Smoker", "value": "YES"}
#             ], style={"width": "50%"}),
#             html.Label("Data Source"),
#             dcc.Dropdown(id="source", options=[
#                 {"label": "InPerson", "value": "InPerson"},
#                 {"label": "Remote", "value": "Remote"}
#             ], style={"width": "50%"}),
#             html.Br(),
#             html.Button("Calculate Risk", id="predict", n_clicks=0, style={"marginTop": "10px"})
#         ], style={"display": "flex", "flexDirection": "column", "gap": "10px", "padding": "20px"}),
#         html.Div(id="prediction-output"),
#         dcc.Graph(id="score-trend")
#     ])

def calculator_page():
    return html.Div([
        navbar,
        html.H3("🧮 Risk Score Calculator", style={
            "color": "#003366", 
            "textAlign": "center",
            "marginTop": "20px"
        }),

        html.Div([
            html.Div([
                html.Label("Patient ID"), dcc.Input(id="pid", type="text", style={"width": "100%"}),
                html.Label("Height (cm)"), dcc.Input(id="height", type="number", style={"width": "100%"}),
                html.Label("Weight (kg)"), dcc.Input(id="weight", type="number", style={"width": "100%"}),
                html.Label("Diastolic BP"), dcc.Input(id="dbp", type="number", style={"width": "100%"}),
                html.Label("Systolic BP"), dcc.Input(id="sbp", type="number", style={"width": "100%"}),
                html.Label("Heart Rate"), dcc.Input(id="hr", type="number", style={"width": "100%"}),
                html.Label("Respiratory Rate"), dcc.Input(id="rr", type="number", style={"width": "100%"}),
                html.Label("Smoking Status"),
                dcc.Dropdown(id="smoke", options=[
                    {"label": "Never", "value": "NO"},
                    {"label": "Ex-Smoker", "value": "EX"},
                    {"label": "Smoker", "value": "YES"}
                ], style={"width": "100%"}),
                html.Label("Data Source"),
                dcc.Dropdown(id="source", options=[
                    {"label": "InPerson", "value": "InPerson"},
                    {"label": "Remote", "value": "Remote"}
                ], style={"width": "100%"}),
                html.Br(),
                html.Button("🧮 Calculate Risk", id="predict", n_clicks=0, style={
                    "marginTop": "10px", 
                    "width": "100%", 
                    "backgroundColor": "#003366",
                    "color": "white",
                    "fontWeight": "bold",
                    "padding": "10px"
                })
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",
                "padding": "25px",
                "backgroundColor": "#f5faff",
                "border": "1px solid #cce0f5",
                "borderRadius": "10px",
                "boxShadow": "0 2px 10px rgba(0, 0, 0, 0.1)"
            }),
        ], style={
            "width": "50%",
            "margin": "auto",
            "marginTop": "30px",
            "marginBottom": "30px"
        }),

        html.Div(id="prediction-output", style={
            "textAlign": "center",
            "marginTop": "20px",
            "fontWeight": "bold",
            "fontSize": "20px"
        }),

        dcc.Graph(id="score-trend", style={"padding": "30px"})
    ])

def table_page(title, source_filter):
    return html.Div([
        navbar,
        html.H3(title),
        html.P(f"Displaying patients with source = {source_filter}"),
        dcc.Loading(html.Div(id=f"{source_filter.lower()}-table")),
        dcc.Graph(id=f"{source_filter.lower()}-trend")
    ])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def route(path):
    if path == "/calculator": return calculator_page()
    elif path == "/InPerson": return table_page("🩺 In Person Data", "InPerson")
    elif path == "/remote": return table_page("📡 Remote Monitoring", "Remote")
    return home_page()

@app.callback(
    Output("prediction-output", "children"),
    Output("score-trend", "figure"),
    Input("predict", "n_clicks"),
    State("pid", "value"), State("height", "value"), State("weight", "value"),
    State("dbp", "value"), State("sbp", "value"),
    State("hr", "value"), State("rr", "value"),
    State("smoke", "value"), State("source", "value")
)
def calculate_score(n, pid, h, w, dbp, sbp, hr, rr, smoke, source):
    if n == 0 or None in [pid, h, w, dbp, sbp, hr, rr, smoke, source]:
        return "⚠️ Please enter all values.", {}

    # ✅ Updated logic with physiological safeguards
    bmi = w / ((h / 100) ** 2)
    pp = abs(sbp - dbp)
    phr = max(pp * hr, 0)
    ratio = max(sbp / dbp, 0.01)
    hrxrr = hr * rr
    hs = hr / bmi
    sc = {"NO": 0, "EX": 1, "YES": 2}[smoke]
    sbmi = sc * bmi
    bps = ratio * sc
    bcat = 1 if bmi >= 25 else 0

    df = pd.DataFrame([{
        "Body Height": h, "Body Weight": w, "Diastolic Blood Pressure": dbp,
        "Systolic Blood Pressure": sbp, "Heart rate": hr, "Respiratory rate": rr,
        "BMI": bmi, "Pulse_Pressure": pp, "BP_Ratio": ratio, "HRxRR": hrxrr,
        "HeartStress": hs, "SmokingBMI": sbmi, "Pulse_HR": phr,
        "BP_Smoking": bps
    }])
    scaled = scaler.transform(df)
    v = dict(zip(df.columns, scaled[0]))
    v['Smoking_encoded'] = sc
    v['BMI_Category_Overweight'] = bcat

    score = round(sum([
        0.141 * v['SmokingBMI'], 0.138 * v['Smoking_encoded'], 0.112 * v['BP_Smoking'],
        0.110 * v['Body Height'], 0.099 * v['Body Weight'], 0.084 * v['BMI'],
        0.083 * v['Systolic Blood Pressure'], 0.079 * v['HeartStress'],
        0.077 * v['Respiratory rate'], 0.077 * v['BMI_Category_Overweight']
    ]), 2)
    risk = "Low" if score <= 2.5 else "Medium" if score <= 4.0 else "High"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = pd.read_csv(LOG_FILE)
    new_row = pd.DataFrame([{
        "PatientID": pid, "Timestamp": now,
        "Score": score, "Risk": risk, "Source": source
    }])
    log = pd.concat([log, new_row], ignore_index=True)
    log.to_csv(LOG_FILE, index=False)

    patient_log = log[log.PatientID == pid]
    fig = px.line(patient_log, x="Timestamp", y="Score", title=f"Trend for {pid}", markers=True)

    return [
        html.H4(f"🧮 Calculated Score: {score}"),
        html.H4(f"⚠️ Health Risk Level: {risk}")
    ], fig

@app.callback(Output("inperson-table", "children"), Output("inperson-trend", "figure"), Input("url", "pathname"))
def traige_scores(path):
    if path != "/InPerson": return dash.no_update, dash.no_update
    df = pd.read_csv(LOG_FILE)
    df = df[df.Source == "InPerson"]
    recent = df[df["Timestamp"] >= (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")]
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=recent.to_dict("records"),
        style_cell={"textAlign": "center"},
        style_data_conditional=[
            {"if": {"filter_query": '{Risk} = "High"'}, "backgroundColor": "#f8d7da"},
            {"if": {"filter_query": '{Risk} = "Medium"'}, "backgroundColor": "#fff3cd"},
            {"if": {"filter_query": '{Risk} = "Low"'}, "backgroundColor": "#d4edda"}
        ]
    )
    # return table, px.histogram(recent, x="Risk", color="Risk", title="")
    return table, px.histogram(
    recent, 
    x="Risk", 
    color="Risk", 
    color_discrete_map={
        "Low": "#28a745",     # ✅ Green
        "Medium": "#ffc107",  # ⚠️ Yellow
        "High": "#dc3545"     # 🔴 Red
    },
    title="InPerson data Risk Summary")

@app.callback(Output("remote-table", "children"), Output("remote-trend", "figure"), Input("url", "pathname"))
def remote_scores(path):
    if path != "/remote": return dash.no_update, dash.no_update
    df = pd.read_csv(LOG_FILE)
    df = df[df.Source == "Remote"]
    recent = df[df["Timestamp"] >= (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M")]
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in df.columns],
        data=recent.to_dict("records"),
        style_cell={"textAlign": "center"},
        style_data_conditional=[
            {"if": {"filter_query": '{Risk} = "High"'}, "backgroundColor": "#f8d7da"},
            {"if": {"filter_query": '{Risk} = "Medium"'}, "backgroundColor": "#fff3cd"},
            {"if": {"filter_query": '{Risk} = "Low"'}, "backgroundColor": "#d4edda"}
        ]
    )
    # return table, px.histogram(, x="Risk", color="Risk", title="")
    return table, px.histogram(
        recent, 
        x="Risk", 
        color="Risk", 
        color_discrete_map={
            "Low": "#28a745",     
            "Medium": "#ffc107",  
            "High": "#dc3545"  
        },
        title="Remote Monitoring Risk Summary"
    )

@app.callback(Output("risk-summary-chart", "figure"), Input("url", "pathname"))
def summary_graph(path):
    df = pd.read_csv(LOG_FILE)
    today = df[df["Timestamp"].str.startswith(datetime.now().strftime("%Y-%m-%d"))]
    # return px.histogram(today, x="Risk", color="Risk", title="Today's Risk Summary")
    return px.histogram(
        today, 
        x="Risk", 
        color="Risk", 
        color_discrete_map={
            "Low": "#28a745",     
            "Medium": "#ffc107",  
            "High": "#dc3545"     
        },
        title="Today's Risk Summary"
    )

if __name__ == '__main__':
    app.run(debug=True)
