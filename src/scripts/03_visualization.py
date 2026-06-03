"""
03_visualization.py
-------------------
Geospatial visualisation script for Perth Crime Hotspot Analysis.

Generates:
  1. A Folium heatmap of total crime intensity across Perth suburbs.
  2. A Folium marker-cluster map colour-coded by risk tier.
  3. A Plotly bar chart of the top 15 suburbs by total offences.
  4. A Plotly stacked-bar chart of offence types by suburb (top 10).

All outputs are saved to outputs/maps/ and outputs/figures/.

Libraries used: Folium, Plotly, Pandas
"""

import os
import json
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CLUSTERED_PATH = os.path.join(PROCESSED_DIR, "suburb_clusters.csv")
MAPS_DIR = os.path.join(BASE_DIR, "outputs", "maps")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

# Perth city centre coordinates
PERTH_LAT = -31.9505
PERTH_LON = 115.8605

# Risk-tier colour palette
TIER_COLORS = {
    "High Crime": "#d73027",
    "Moderate-High Crime": "#fc8d59",
    "Moderate Crime": "#fee08b",
    "Low Crime": "#91cf60",
}


def load_data(path: str) -> pd.DataFrame:
    """Load the clustered suburb data."""
    df = pd.read_csv(path)
    print(f"Loaded clustered data: {len(df)} suburbs.")
    return df


# ---------------------------------------------------------------------------
# 1. Heatmap
# ---------------------------------------------------------------------------
def create_heatmap(df: pd.DataFrame, maps_dir: str) -> str:
    """
    Build a Folium HeatMap layer weighted by total offences.
    Returns the path to the saved HTML file.
    """
    fmap = folium.Map(location=[PERTH_LAT, PERTH_LON], zoom_start=11,
                      tiles="CartoDB positron")

    heat_data = [
        [row["latitude"], row["longitude"], row["total_offences"]]
        for _, row in df.iterrows()
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
    ]

    HeatMap(
        heat_data,
        min_opacity=0.4,
        radius=30,
        blur=20,
        gradient={0.2: "blue", 0.5: "lime", 0.8: "yellow", 1.0: "red"},
    ).add_to(fmap)

    folium.LayerControl().add_to(fmap)

    # Add title
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                background:white;padding:8px 16px;border-radius:6px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3);font-family:sans-serif;
                font-size:15px;font-weight:bold;z-index:9999;">
        Perth Crime Heatmap — 2022/23
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))

    os.makedirs(maps_dir, exist_ok=True)
    out_path = os.path.join(maps_dir, "perth_crime_heatmap.html")
    fmap.save(out_path)
    print(f"Heatmap saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. Marker-cluster map coloured by risk tier
# ---------------------------------------------------------------------------
def create_cluster_map(df: pd.DataFrame, maps_dir: str) -> str:
    """
    Build a Folium map with circle markers colour-coded by cluster risk tier.
    Returns the path to the saved HTML file.
    """
    fmap = folium.Map(location=[PERTH_LAT, PERTH_LON], zoom_start=11,
                      tiles="CartoDB positron")

    marker_cluster = MarkerCluster(name="Suburbs").add_to(fmap)

    for _, row in df.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        tier = row.get("risk_tier", "Unknown")
        color = TIER_COLORS.get(tier, "#999999")

        popup_html = f"""
        <b>{row['suburb']}</b><br>
        District: {row['district']}<br>
        LGA: {row['local_government_area']}<br>
        Risk Tier: <span style="color:{color};font-weight:bold;">{tier}</span><br>
        Total Offences: {int(row['total_offences'])}
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(5, min(20, row["total_offences"] / 50)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{row['suburb']} — {tier}",
        ).add_to(marker_cluster)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;background:white;
                padding:10px 14px;border-radius:6px;
                box-shadow:2px 2px 6px rgba(0,0,0,.3);font-family:sans-serif;
                font-size:13px;z-index:9999;">
        <b>Risk Tier</b><br>
        <span style="color:#d73027;">&#9679;</span> High Crime<br>
        <span style="color:#fc8d59;">&#9679;</span> Moderate-High Crime<br>
        <span style="color:#fee08b;">&#9679;</span> Moderate Crime<br>
        <span style="color:#91cf60;">&#9679;</span> Low Crime
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(fmap)

    out_path = os.path.join(maps_dir, "perth_cluster_map.html")
    fmap.save(out_path)
    print(f"Cluster map saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 3. Top suburbs bar chart (Plotly)
# ---------------------------------------------------------------------------
def create_top_suburbs_chart(df: pd.DataFrame, figures_dir: str, top_n: int = 15) -> str:
    """Bar chart of the top N suburbs by total offences."""
    top = df.nlargest(top_n, "total_offences").sort_values("total_offences")

    fig = px.bar(
        top,
        x="total_offences",
        y="suburb",
        orientation="h",
        color="risk_tier",
        color_discrete_map=TIER_COLORS,
        title=f"Top {top_n} Perth Suburbs by Total Offences (2022/23)",
        labels={"total_offences": "Total Offences", "suburb": "Suburb",
                "risk_tier": "Risk Tier"},
        template="plotly_white",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, legend_title="Risk Tier")

    os.makedirs(figures_dir, exist_ok=True)
    out_path = os.path.join(figures_dir, "top_suburbs_bar.html")
    fig.write_html(out_path)
    print(f"Top suburbs chart saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 4. Offence-type stacked bar (Plotly)
# ---------------------------------------------------------------------------
def create_offence_type_chart(df: pd.DataFrame, figures_dir: str, top_n: int = 10) -> str:
    """Stacked bar chart showing offence-type breakdown for the top N suburbs."""
    offence_cols = [
        c for c in df.columns
        if c not in {"suburb", "district", "local_government_area", "latitude", "longitude",
                     "total_offences", "cluster", "risk_tier"}
    ]
    top = df.nlargest(top_n, "total_offences")[["suburb"] + offence_cols].set_index("suburb")

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, col in enumerate(offence_cols):
        fig.add_trace(go.Bar(
            name=col,
            x=top.index,
            y=top[col],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="stack",
        title=f"Offence Type Breakdown — Top {top_n} Suburbs (2022/23)",
        xaxis_title="Suburb",
        yaxis_title="Total Offences",
        template="plotly_white",
        legend_title="Offence Division",
        xaxis={"tickangle": -30},
    )

    out_path = os.path.join(figures_dir, "offence_type_stacked.html")
    fig.write_html(out_path)
    print(f"Offence type chart saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data(CLUSTERED_PATH)

    create_heatmap(df, MAPS_DIR)
    create_cluster_map(df, MAPS_DIR)
    create_top_suburbs_chart(df, FIGURES_DIR)
    create_offence_type_chart(df, FIGURES_DIR)

    print("\nAll visualisations generated successfully.")


if __name__ == "__main__":
    main()
