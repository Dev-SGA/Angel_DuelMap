import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Duel Map Analysis")

st.title("Duel Map Analysis - Multiple Matches")
st.caption("Click on the icons on the pitch to play the corresponding video analysis.")

# ==========================
# Data Setup (All English labels with new data)
# ==========================
matches_data = {
    "vs IMG": [
        ("DUEL OFENSIVO LOST", 35.40, 65.43, "videos/Duel Of Lost 1.mp4"),
        ("DUEL DEFENSIVO LOST", 24.43, 73.08, "videos/Duel Lost 1.mp4"),
        ("DUEL DEFENSIVO LOST", 65.49, 56.29, "videos/Duel Lost 2.mp4"),
        ("DUEL DEFENSIVO LOST", 53.35, 55.29, "videos/Duel Lost 3.mp4"),
        ("DUEL OFENSIVO LOST", 80.78, 66.93, "videos/Duel Of Lost 2.mp4"),
    ],
    "vs Orlando": [
        ("DUEL OFENSIVO WON", 50.86, 55.79, "videos/Duel Of Won 1.mp4"),
        ("DUEL DEFENSIVO WON", 56.68, 26.37, "videos/Duel Def Won 1.mp4"),
        ("BLOQUEIO", 17.28, 49.64, None),
        ("FOULED", 80.45, 49.64, None),
        ("DUEL OFENSIVO LOST", 100.73, 41.00, "videos/Duel Of Lost 3.mp4"),
        ("INTERCEPTACAO", 82.61, 64.77, "videos/Intercept 1.mp4"),
        ("DUEL DEFENSIVO WON", 70.47, 55.96, "videos/Duel Def Won 2.mp4"),
    ],
    "vs Weston": [
        ("DUEL DEFENSIVO WON", 41.05, 27.20, "videos/Duel Def Won 3.mp4"),
        ("INTERCEPT", 72.30, 10.08, "videos/Intercept 2.mp4"),
        ("DUEL DEFENSIVO WON", 33.90, 37.01, "videos/Duel Def Won 4.mp4"),
        ("DUEL OFENSIVO LOST", 94.24, 36.51, "videos/Duel Of Lost 4.mp4"),
        ("DUEL OFENSIVO LOST", 33.90, 46.82, "videos/Duel Of Lost 5.mp4"),
        ("INTERCEPT", 33.57, 43.99, "videos/Intercept 3.mp4"),
        ("DUEL DEFENSIVO WON", 41.88, 12.74, "videos/Duel Def Won 5.mp4"),
        ("DUEL DEFENSIVO WON", 38.22, 23.21, "videos/Duel Def Won 6.mp4"),
        ("INTERCEPT", 69.14, 13.24, "videos/Intercept 4.mp4"),
        ("DUEL DEFENSIVO WON", 48.03, 14.73, "videos/Duel Def Won 7.mp4"),
        ("DUEL DEFENSIVO LOST", 69.14, 41.83, "videos/Duel Lost 4.mp4"),
        ("INTERCEPT", 65.15, 52.47, "videos/Intercept 5.mp4"),
        ("FOULED", 40.22, 24.21, None),
        ("DUEL DEFENSIVO LOST", 35.90, 53.13, "videos/Duel Lost 5.mp4"),
        ("DUEL DEFENSIVO LOST", 57.51, 75.57, "videos/Duel Lost 6.mp4"),
    ],
    "vs South Florida": [
        ("DUEL DEFENSIVO LOST", 20.79, 55.45, "videos/Duel Lost 7.mp4"),
        ("BLOQUEIO", 20.07, 47.68, None),
        ("DUEL DEFENSIVO WON", 27.11, 3.23, "videos/Duel Def Won 8.mp4"),
        ("DUEL DEFENSIVO WON", 33.98, 14.43, "videos/Duel Def Won 9.mp4"),
    ],
}

dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(events, columns=["type", "x", "y", "video"])

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All games": df_all}
full_data.update(dfs_by_match)

def get_style(event_type, has_video):
    event_type = event_type.upper()

    if "OFENSIVO" in event_type:
        if "WON" in event_type:
            return 'o', (0.1, 0.95, 0.1, 0.95), 110, 0.5
        if "LOST" in event_type:
            alpha = 0.95 if has_video else 0.75
            return 'x', (0.95, 0.1, 0.1, alpha), 120, 3.0

    if "DEFENSIVO" in event_type:
        if "WON" in event_type:
            return 's', (0.0, 0.6, 0.0, 0.9), 110, 0.5
        if "LOST" in event_type:
            alpha = 0.9 if has_video else 0.65
            return 'D', (0.7, 0.0, 0.0, alpha), 110, 2.5

    if "BLOQUEIO" in event_type:
        return '*', (0.7, 0.3, 0.9, 0.8), 140, 0.5

    if "INTERCEPT" in event_type or "INTERCEPTACAO" in event_type:
        return '*', (0.2, 0.6, 0.95, 0.9), 160, 0.8

    if "FOULED" in event_type:
        return 'P', (1, 0.5, 0, 0.8), 110, 0.5

    return 'o', (0.5, 0.5, 0.5, 0.8), 90, 0.5

def translate_event_type(event_type):
    t = event_type.strip().upper()
    if "DUEL OFENSIVO" in t:
        if "WON" in t:
            return "Offensive Duel Won"
        if "LOST" in t:
            return "Offensive Duel Lost"
    if "DUEL DEFENSIVO" in t:
        if "WON" in t:
            return "Defensive Duel Won"
        if "LOST" in t:
            return "Defensive Duel Lost"
    if "BLOQUEIO" in t:
        return "Block"
    if "INTERCEPT" in t or "INTERCEPTACAO" in t:
        return "Interception"
    if "FOULED" in t:
        return "Fouled"
    return event_type  # fallback literal

def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)

    is_duel = df['type'].str.contains('DUEL|AERIAL', case=False)
    is_won = df['type'].str.contains('WON', case=False)
    is_offensive = df['type'].str.contains('OFENSIVO|OFFENSIVE', case=False)
    is_defensive = df['type'].str.contains('DEFENSIVO|DEFENSIVE', case=False)

    all_duels = df[is_duel]
    total_duels = len(all_duels)
    won_duels = all_duels[is_won].shape[0]
    duel_rate = (won_duels / total_duels * 100) if total_duels > 0 else 0

    off_duels = df[is_offensive & is_duel]
    off_total = len(off_duels)
    off_wins = off_duels[is_won].shape[0]
    off_rate = (off_wins / off_total * 100) if off_total > 0 else 0

    def_duels = df[is_defensive & is_duel]
    def_total = len(def_duels)
    def_wins = def_duels[is_won].shape[0]
    def_rate = (def_wins / def_total * 100) if def_total > 0 else 0

    aerial_duels = df[df['type'].str.contains('AERIAL', case=False)]
    aerial_total = len(aerial_duels)
    aerial_wins = aerial_duels[is_won].shape[0]
    aerial_rate = (aerial_wins / aerial_total * 100) if aerial_total > 0 else 0

    left_mask = df['y'] < 26.6
    left_duels = df[left_mask & is_duel]
    left_total = len(left_duels)
    left_wins = left_duels[is_won].shape[0]
    left_rate = (left_wins / left_total * 100) if left_total > 0 else 0

    central_mask = (df['y'] >= 26.6) & (df['y'] <= 53.3)
    central_duels = df[central_mask & is_duel]
    c_total = len(central_duels)
    c_wins = central_duels[is_won].shape[0]
    c_rate = (c_wins / c_total * 100) if c_total > 0 else 0

    right_mask = df['y'] > 53.3
    right_duels = df[right_mask & is_duel]
    right_total = len(right_duels)
    right_wins = right_duels[is_won].shape[0]
    right_rate = (right_wins / right_total * 100) if right_total > 0 else 0

    blocks = len(df[df['type'].str.contains('BLOQUEIO', case=False)])
    intercepts = len(df[df['type'].str.contains('INTERCEPT', case=False)])
    fouls = len(df[df['type'].str.contains('FOULED', case=False)])

    return {
        "total": total,
        "duel_total": total_duels,
        "duel_wins": won_duels,
        "duel_rate": duel_rate,
        "off_total": off_total,
        "off_wins": off_wins,
        "off_rate": off_rate,
        "def_total": def_total,
        "def_wins": def_wins,
        "def_rate": def_rate,
        "aerial_total": aerial_total,
        "aerial_wins": aerial_wins,
        "aerial_rate": aerial_rate,
        "central_total": c_total,
        "central_wins": c_wins,
        "central_rate": c_rate,
        "left_total": left_total,
        "left_wins": left_wins,
        "left_rate": left_rate,
        "right_total": right_total,
        "right_wins": right_wins,
        "right_rate": right_rate,
        "blocks": blocks,
        "intercepts": intercepts,
        "fouls": fouls,
    }

# ==========================
# Sidebar Configuration
# ==========================
st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

filter_duel_type = st.sidebar.multiselect(
    "Duel Type",
    ["Offensive", "Defensive", "Aerial", "Other"],
    default=["Offensive", "Defensive", "Aerial", "Other"]
)

st.sidebar.divider()
st.sidebar.caption("Match filtered by selected options above")

df = full_data[selected_match].copy()

if not all(x in filter_duel_type for x in ["Offensive", "Defensive", "Aerial", "Other"]):
    mask = pd.Series([False] * len(df))
    if "Offensive" in filter_duel_type:
        mask |= df['type'].str.contains('OFENSIVO', case=False)
    if "Defensive" in filter_duel_type:
        mask |= df['type'].str.contains('DEFENSIVO', case=False)
    if "Aerial" in filter_duel_type:
        mask |= df['type'].str.contains('AERIAL', case=False)
    if "Other" in filter_duel_type:
        mask |= ~df['type'].str.contains('OFENSIVO|DEFENSIVO|AERIAL', case=False)
    df = df[mask]

stats = compute_stats(full_data[selected_match])

# ==========================
# Main Layout
# ==========================
col_map, col_vid = st.columns([1, 1])

with col_map:
    st.subheader("Interactive Pitch Map")
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#f8f8f8', line_color='#4a4a4a')
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in df.iterrows():
        has_vid = row["video"] is not None
        marker, color, size, lw = get_style(row["type"], has_vid)
        ec = 'black' if has_vid else color
        pitch.scatter(row.x, row.y, marker=marker, s=size, color=color,
                      edgecolors=ec, linewidths=lw, ax=ax, zorder=3)

    ax.annotate('', xy=(70, 83), xytext=(50, 83),
        arrowprops=dict(arrowstyle='->', color='#4a4a4a', lw=1.5))
    ax.text(60, 86, "Attack Direction", ha='center', va='center',
        fontsize=9, color='#4a4a4a', fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Offensive Duel Won',
               markerfacecolor=(0.1, 0.95, 0.1, 0.95), markersize=10, linestyle='None'),
        Line2D([0], [0], marker='x', color='w', label='Offensive Duel Lost',
               markeredgecolor=(0.95, 0.1, 0.1, 0.95), markersize=10, markeredgewidth=2.5, linestyle='None'),
        Line2D([0], [0], marker='s', color='w', label='Defensive Duel Won',
               markerfacecolor=(0.0, 0.6, 0.0, 0.9), markersize=10, linestyle='None'),
        Line2D([0], [0], marker='D', color='w', label='Defensive Duel Lost',
               markerfacecolor=(0.7, 0.0, 0.0, 0.9), markersize=10, linestyle='None'),
        Line2D([0], [0], marker='*', color='w', label='Shot Block',
               markerfacecolor=(0.7, 0.3, 0.9, 0.8), markersize=12, linestyle='None'),
        Line2D([0], [0], marker='*', color='w', label='Interception',
               markerfacecolor=(0.2, 0.6, 0.95, 0.9), markersize=14, linestyle='None'),
        Line2D([0], [0], marker='P', color='w', label='Fouled',
               markerfacecolor=(1, 0.5, 0, 0.8), markersize=10, linestyle='None'),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor='white',
        edgecolor='#333333',
        fontsize='small',
        title="Match Events",
        title_fontsize='medium',
        labelspacing=1.2,
        borderpad=1.0,
        framealpha=0.95
    )
    legend.get_title().set_fontweight('bold')

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_obj = Image.open(buf)
    click = streamlit_image_coordinates(img_obj, width=700)

selected_event = None

if click is not None:
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)
    mpl_pixel_y = real_h - pixel_y
    coords = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
    field_x, field_y = coords[0], coords[1]

    df["dist"] = np.sqrt((df["x"] - field_x)**2 + (df["y"] - field_y)**2)
    RADIUS = 5
    candidates = df[df["dist"] < RADIUS]

    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

with col_vid:
    st.subheader("Event Details")
    if selected_event is not None:
        st.success(f"**Selected Event:** {translate_event_type(selected_event['type'])}")
        if selected_event["video"]:
            try:
                st.video(selected_event["video"])
            except Exception:
                st.error(f"Video file not found: {selected_event['video']}")
        else:
            st.warning("No video footage available for this specific event.")
    else:
        st.info("Select a marker on the pitch to view event details.")

    st.divider()
    st.subheader("Performance Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Duels", f"{stats['duel_wins']}/{stats['duel_total']}", f"{stats['duel_rate']:.1f}% Success")
    col2.metric("Offensive Duels", f"{stats['off_wins']}/{stats['off_total']}", f"{stats['off_rate']:.1f}% Success")
    col3.metric("Defensive Duels", f"{stats['def_wins']}/{stats['def_total']}", f"{stats['def_rate']:.1f}% Success")

    st.divider()
    st.subheader("Zone Performance")

    zc1, zc2, zc3 = st.columns(3)
    zc1.metric("Left Corridor", f"{stats['left_wins']}/{stats['left_total']}", f"{stats['left_rate']:.1f}% Success")
    zc2.metric("Central Corridor", f"{stats['central_wins']}/{stats['central_total']}", f"{stats['central_rate']:.1f}% Success")
    zc3.metric("Right Corridor", f"{stats['right_wins']}/{stats['right_total']}", f"{stats['right_rate']:.1f}% Success")

    st.divider()
    st.subheader("Organization / Transitions Duel Summary")

    # Organization / Set Pieces
    org_won = 6
    org_lost = 2
    org_pct = f"{(org_won / (org_won + org_lost) * 100):.1f}%"

    col_org1, col_org2, col_org3 = st.columns(3)
    col_org1.metric("Organization Duels Won", f"{org_won}")
    col_org2.metric("Organization Duels Lost", f"{org_lost}")
    col_org3.metric("Organization %", org_pct)

    # Transitions
    trans_won = 3
    trans_lost = 5
    trans_pct = f"{(trans_won / (trans_won + trans_lost) * 100):.1f}%"

    col_tr1, col_tr2, col_tr3 = st.columns(3)
    col_tr1.metric("Transitions Duels Won", f"{trans_won}")
    col_tr2.metric("Transitions Duels Lost", f"{trans_lost}")
    col_tr3.metric("Transitions %", trans_pct)
    
    st.divider()
    st.subheader("Other Events")

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Blocks", stats["blocks"])
    ec2.metric("Interceptions", stats["intercepts"])
    ec3.metric("Fouls", stats["fouls"])
