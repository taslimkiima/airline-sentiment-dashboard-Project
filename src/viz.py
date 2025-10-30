# src/viz.py
import plotly.express as px
import plotly.io as pio
import pandas as pd

PALETTE = px.colors.qualitative.Safe  # bisa ganti: Set2, Prism, Pastel, dll.

def apply_plotly_theme():
    """Daftarkan template Plotly custom & set default."""
    pio.templates["uasku"] = dict(
        layout=dict(
            font=dict(family="Inter, Segoe UI, Arial", size=13),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(l=40, r=20, t=50, b=40),
            xaxis=dict(gridcolor="#eeeeee", zerolinecolor="#eeeeee"),
            yaxis=dict(gridcolor="#eeeeee", zerolinecolor="#eeeeee"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
    )
    pio.templates.default = "uasku"

def fig_sentiment_bar(sent_count: pd.DataFrame):
    fig = px.bar(
        sent_count, x="sentiment", y="tweets", text="tweets",
        height=420, color="sentiment", color_discrete_sequence=PALETTE, template="uasku"
    )
    fig.update_traces(textposition="outside")
    return fig

def fig_sentiment_pie(sent_count: pd.DataFrame):
    return px.pie(
        sent_count, names="sentiment", values="tweets",
        height=420, hole=0.35, color="sentiment", color_discrete_sequence=PALETTE, template="uasku"
    )

def fig_hour_trend(trend: pd.DataFrame):
    return px.line(
        trend, x="hour", y="tweets", color="airline_sentiment",
        markers=True, height=460, color_discrete_sequence=PALETTE, template="uasku"
    )

def fig_topic_stack(df_topic: pd.DataFrame):
    return px.bar(
        df_topic, x="topic", y="tweets", color="airline_sentiment",
        barmode="stack", height=420, color_discrete_sequence=PALETTE, template="uasku"
    )
