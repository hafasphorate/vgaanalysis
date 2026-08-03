import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Multi-VGA Analysis Dashboard", layout="wide")
st.title("📊 Multi-VGA Analysis & Spatial Metric Dashboard")

# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def parse_filename(filename):
    """Parses 'Northshore-L1.csv' -> Mall: 'Northshore', Level: 'L1'"""
    base = os.path.splitext(filename)[0]
    parts = base.split("-")
    if len(parts) >= 2:
        mall = parts[0].strip()
        level = "-".join(parts[1:]).strip()
        return mall, level, base
    return "Default", base, base

def draw_histogram(metric_data, metric_name):
    """Generates histogram figure with all 4 value-range thresholds."""
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.hist(metric_data, bins=25, edgecolor="black", alpha=0.7)
    
    val_min = metric_data.min()
    val_max = metric_data.max()
    val_range = val_max - val_min
    
    r10_low = val_min + 0.10 * val_range
    r25_low = val_min + 0.25 * val_range
    r75_high = val_max - 0.25 * val_range
    r90_high = val_max - 0.10 * val_range
    
    ax.axvline(r10_low, color='darkblue', linestyle='--', linewidth=1.2, label='Low 10%')
    ax.axvline(r25_low, color='cyan', linestyle='--', linewidth=1.2, label='Low 25%')
    ax.axvline(r75_high, color='orange', linestyle='--', linewidth=1.2, label='High 25%')
    ax.axvline(r90_high, color='red', linestyle='--', linewidth=1.2, label='High 10%')
    
    ax.set_title(metric_name, fontsize=10)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    return fig

def draw_overlay_chart(filtered_dict, comp_metric, chart_style="Line (Normalized Histogram)"):
    """Plots normalized relative proportions of selected floorplans overlaid on a single chart."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    all_vals = []
    for (m_name, l_name), item in filtered_dict.items():
        if comp_metric in item["df"].columns:
            all_vals.extend(item["df"][comp_metric].dropna().tolist())
            
    if not all_vals:
        return fig
        
    global_min, global_max = min(all_vals), max(all_vals)
    bins = np.linspace(global_min, global_max, 30)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    for (m_name, l_name), item in filtered_dict.items():
        m_df = item["df"]
        if comp_metric in m_df.columns:
            m_series = m_df[comp_metric].dropna()
            label_str = f"{m_name} - {l_name}"
            
            if chart_style == "Line (Normalized Histogram)":
                counts, _ = np.histogram(m_series, bins=bins)
                proportions = counts / len(m_series)
                ax.plot(bin_centers, proportions, marker='o', markersize=3, linewidth=2, label=label_str)
            else:
                m_series.plot(kind='kde', ax=ax, label=label_str, linewidth=2)

    ax.set_title(f"Cross-Mall Distribution Overlay: {comp_metric}", fontsize=12, fontweight='bold')
    ax.set_xlabel(f"{comp_metric} Value", fontsize=10)
    
    if chart_style == "Line (Normalized Histogram)":
        ax.set_ylabel("Proportion of Total Points (Count / Total N)", fontsize=10)
    else:
        ax.set_ylabel("Density", fontsize=10)
        
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    return fig

def process_vga_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) <= 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=r'\s+|,|\t', engine='python')
    except Exception as e:
        return None, None, f"Error reading file: {e}"
        
    filename = uploaded_file.name
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(num_cols) == 0:
        return None, df, "No numerical columns detected in CSV."

    mall, level, base_name = parse_filename(filename)

    stats = {
        "Mall": mall,
        "Level": level,
        "File": filename,
        "Total_Grid_Points": len(df)
    }
    
    for col in num_cols:
        series = df[col].dropna()
        n_total = len(series)
        
        val_min = series.min()
        val_max = series.max()
        val_range = val_max - val_min
        
        r10_low = val_min + 0.10 * val_range
        r25_low = val_min + 0.25 * val_range
        r75_high = val_max - 0.25 * val_range
        r90_high = val_max - 0.10 * val_range
        
        stats[f"{col}_Mean"] = series.mean()
        stats[f"{col}_Std"] = series.std()
        stats[f"{col}_Min"] = val_min
        stats[f"{col}_Max"] = val_max
        stats[f"{col}_Median"] = series.median()
        
        stats[f"{col}_ValRange_Low10_Threshold"] = r10_low
        stats[f"{col}_ValRange_Low25_Threshold"] = r25_low
        stats[f"{col}_ValRange_High75_Threshold"] = r75_high
        stats[f"{col}_ValRange_High90_Threshold"] = r90_high
        
        stats[f"{col}_GridPoints_In_Lower_10pct_Range"] = (series <= r10_low).sum()
        stats[f"{col}_GridPoints_In_Lower_25pct_Range"] = (series <= r25_low).sum()
        stats[f"{col}_GridPoints_In_Upper_25pct_Range"] = (series >= r75_high).sum()
        stats[f"{col}_GridPoints_In_Upper_10pct_Range"] = (series >= r90_high).sum()
        
        stats[f"{col}_PctPoints_In_Lower_10pct_Range"] = ((series <= r10_low).sum() / n_total) * 100 if n_total > 0 else 0
        stats[f"{col}_PctPoints_In_Lower_25pct_Range"] = ((series <= r25_low).sum() / n_total) * 100 if n_total > 0 else 0
        stats[f"{col}_PctPoints_In_Upper_25pct_Range"] = ((series >= r75_high).sum() / n_total) * 100 if n_total > 0 else 0
        stats[f"{col}_PctPoints_In_Upper_10pct_Range"] = ((series >= r90_high).sum() / n_total) * 100 if n_total > 0 else 0
        
    return stats, df, None

# -----------------------------------------------------------------------------
# 2. SIDEBAR FILE UPLOADERS FOR CLOUD DEPLOYMENT
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Upload VGA Data & Images")

uploaded_csvs = st.sidebar.file_uploader(
    "Upload VGA CSV Files", 
    type=["csv", "txt"], 
    accept_multiple_files=True
)

uploaded_images = st.sidebar.file_uploader(
    "Upload Matching VGA PNG Images", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

image_dict = {}
if uploaded_images:
    for img_file in uploaded_images:
        image_dict[img_file.name] = img_file

# Process uploaded CSVs
summary_list = []
data_dict = {}

if uploaded_csvs:
    for uploaded_csv in uploaded_csvs:
        stats, raw_df, err = process_vga_uploaded_file(uploaded_csv)
        if not err:
            summary_list.append(stats)
            filename = uploaded_csv.name
            mall, level, base_name = parse_filename(filename)
            data_dict[(mall, level)] = {
                "filename": filename,
                "base_name": base_name,
                "df": raw_df
            }

# Helper to find uploaded images
def get_uploaded_image(base_name, selected_metric):
    metric_raw = selected_metric.strip()
    metric_hyphen = metric_raw.replace(" ", "-")
    metric_underscore = metric_raw.replace(" ", "_")
    
    file_candidates = [
        f"{base_name}-{metric_hyphen}.png",
        f"{base_name}-{metric_raw}.png",
        f"{base_name}-{metric_underscore}.png",
        f"{base_name}.png"
    ]
    
    for candidate in file_candidates:
        if candidate in image_dict:
            return image_dict[candidate]
    return None

# -----------------------------------------------------------------------------
# 3. DASHBOARD UI
# -----------------------------------------------------------------------------
if data_dict:
    summary_df = pd.DataFrame(summary_list)
    
    tab1, tab2, tab3 = st.tabs([
        "🔍 Single Analysis Inspector", 
        "🏢 Cross-Mall Metric Comparison", 
        "📈 Aggregated Export & Summary"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: SINGLE ANALYSIS INSPECTOR
    # -------------------------------------------------------------------------
    with tab1:
        available_malls = sorted(list(set(m for m, l in data_dict.keys())))
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            selected_mall = st.selectbox("Select Mall", available_malls)
            
        available_levels = sorted([l for m, l in data_dict.keys() if m == selected_mall])
        
        with col_m2:
            selected_level = st.selectbox("Select Level / Floorplan", available_levels)
            
        current_data = data_dict[(selected_mall, selected_level)]
        df_selected = current_data["df"]
        base_name = current_data["base_name"]
        
        numerical_columns = list(df_selected.select_dtypes(include=[np.number]).columns)
        
        with col_m3:
            selected_metric = st.selectbox("Select Spatial Metric", numerical_columns, key="single_metric")

        st.markdown("---")
        
        c1, c2, c3 = st.columns([1, 1.5, 1.5])
        
        with c1:
            st.subheader("Numerical Summary")
            metric_data = df_selected[selected_metric].dropna()
            total_n = len(metric_data)
            
            val_min = metric_data.min()
            val_max = metric_data.max()
            val_range = val_max - val_min
            
            r10_low = val_min + 0.10 * val_range
            r25_low = val_min + 0.25 * val_range
            r75_high = val_max - 0.25 * val_range
            r90_high = val_max - 0.10 * val_range
            
            count_r10_low = (metric_data <= r10_low).sum()
            count_r25_low = (metric_data <= r25_low).sum()
            count_r75_high = (metric_data >= r75_high).sum()
            count_r90_high = (metric_data >= r90_high).sum()

            st.metric("Total Grid Points", f"{total_n:,}")
            st.metric("Min (x)", f"{val_min:.4f}")
            st.metric("Max (y)", f"{val_max:.4f}")
            st.metric("Mean", f"{metric_data.mean():.4f}")
            st.metric("Std Dev", f"{metric_data.std():.4f}")
            
            st.markdown("##### Value-Range Spatial Density")
            st.caption(f"Range: [{val_min:.2f} to {val_max:.2f}]")
            
            st.metric(f"Lower 10% Range (≤ {r10_low:.2f})", f"{count_r10_low:,} pts", f"{(count_r10_low/total_n)*100:.1f}% of grid")
            st.metric(f"Lower 25% Range (≤ {r25_low:.2f})", f"{count_r25_low:,} pts", f"{(count_r25_low/total_n)*100:.1f}% of grid")
            st.metric(f"Upper 25% Range (≥ {r75_high:.2f})", f"{count_r75_high:,} pts", f"{(count_r75_high/total_n)*100:.1f}% of grid")
            st.metric(f"Upper 10% Range (≥ {r90_high:.2f})", f"{count_r90_high:,} pts", f"{(count_r90_high/total_n)*100:.1f}% of grid")

        with c2:
            st.subheader(f"Histogram: {selected_metric}")
            fig = draw_histogram(metric_data, selected_metric)
            st.pyplot(fig)

        with c3:
            st.subheader("VGA Plan Image")
            img_file = get_uploaded_image(base_name, selected_metric)
            if img_file:
                st.image(img_file, caption=f"Loaded: {img_file.name}", use_container_width=True)
            else:
                st.info("Upload matching image files in sidebar to view floorplan visual.")

    # -------------------------------------------------------------------------
    # TAB 2: CROSS-MALL METRIC COMPARISON
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("🏢 Comparative Spatial Analysis Across Malls")
        
        all_metrics = set()
        all_mall_keys = sorted(list(data_dict.keys()))
        all_malls_formatted = [f"{m} - {l}" for m, l in all_mall_keys]
        
        for key, item in data_dict.items():
            cols = list(item["df"].select_dtypes(include=[np.number]).columns)
            all_metrics.update(cols)
        sorted_metrics = sorted(list(all_metrics))

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            comp_metric = st.selectbox("Select Metric to Compare", sorted_metrics, key="comp_metric")
        with col_opt2:
            chart_style = st.selectbox("Overlay Plot Type", ["Line (Normalized Histogram)", "Smooth Curve (KDE Density)"])

        selected_malls_filter = st.multiselect(
            "Select Malls/Levels to Include in Comparison",
            options=all_malls_formatted,
            default=all_malls_formatted
        )

        st.markdown("---")

        filtered_data_dict = {
            (m, l): item for (m, l), item in data_dict.items() 
            if f"{m} - {l}" in selected_malls_filter
        }

        if filtered_data_dict:
            st.markdown(f"#### 📈 Normalized Distribution Overlay (`{comp_metric}`)")
            st.caption("Y-Axis shows relative proportion of floorplan area (Point Count / Total N) to allow fair comparison across malls of different sizes.")
            overlay_fig = draw_overlay_chart(filtered_data_dict, comp_metric, chart_style)
            st.pyplot(overlay_fig)

            st.markdown("---")

            comp_rows = []
            for (m_name, l_name), item in filtered_data_dict.items():
                m_df = item["df"]
                if comp_metric in m_df.columns:
                    m_series = m_df[comp_metric].dropna()
                    t_pts = len(m_series)
                    v_min, v_max = m_series.min(), m_series.max()
                    v_rng = v_max - v_min
                    
                    r10_l = v_min + 0.10 * v_rng
                    r25_l = v_min + 0.25 * v_rng
                    r75_h = v_max - 0.25 * v_rng
                    r90_h = v_max - 0.10 * v_rng
                    
                    c_low10 = (m_series <= r10_l).sum()
                    c_low25 = (m_series <= r25_l).sum()
                    c_high25 = (m_series >= r75_h).sum()
                    c_high10 = (m_series >= r90_h).sum()
                    
                    comp_rows.append({
                        "Mall": m_name,
                        "Level": l_name,
                        "Total Points": t_pts,
                        "Min": round(v_min, 4),
                        "Max": round(v_max, 4),
                        "Mean": round(m_series.mean(), 4),
                        "Std Dev": round(m_series.std(), 4),
                        "Lower 10% Range Pts": c_low10,
                        "Lower 10% Range %": round((c_low10 / t_pts) * 100, 1),
                        "Lower 25% Range Pts": c_low25,
                        "Lower 25% Range %": round((c_low25 / t_pts) * 100, 1),
                        "Upper 25% Range Pts": c_high25,
                        "Upper 25% Range %": round((c_high25 / t_pts) * 100, 1),
                        "Upper 10% Range Pts": c_high10,
                        "Upper 10% Range %": round((c_high10 / t_pts) * 100, 1),
                    })
            
            comp_df = pd.DataFrame(comp_rows)
            st.markdown(f"#### 📊 Comparative Summary Table: `{comp_metric}`")
            st.dataframe(comp_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown(f"#### 🖼️ Individual Floorplan Grid Breakdown: `{comp_metric}`")

            for (m_name, l_name), item in filtered_data_dict.items():
                m_df = item["df"]
                if comp_metric in m_df.columns:
                    m_series = m_df[comp_metric].dropna()
                    b_name = item["base_name"]
                    t_pts = len(m_series)
                    
                    v_min, v_max = m_series.min(), m_series.max()
                    v_rng = v_max - v_min
                    
                    r10_l = v_min + 0.10 * v_rng
                    r25_l = v_min + 0.25 * v_rng
                    r75_h = v_max - 0.25 * v_rng
                    r90_h = v_max - 0.10 * v_rng
                    
                    c_low10 = (m_series <= r10_l).sum()
                    c_low25 = (m_series <= r25_l).sum()
                    c_high25 = (m_series >= r75_h).sum()
                    c_high10 = (m_series >= r90_h).sum()
                    
                    st.markdown(f"##### 📍 {m_name} - {l_name}")
                    grid_c1, grid_c2, grid_c3 = st.columns([1.2, 1.2, 1.2])
                    
                    with grid_c1:
                        st.caption("Quick Stats")
                        st.write(f"**Total Grid Points:** {t_pts:,}")
                        st.write(f"**Range:** [{v_min:.2f} to {v_max:.2f}]")
                        st.write(f"**Mean / Std:** {m_series.mean():.2f} ± {m_series.std():.2f}")
                        
                        st.markdown("**Value-Range Density:**")
                        st.write(f"• **Lower 10% (≤ {r10_l:.2f}):** {c_low10:,} pts ({(c_low10/t_pts)*100:.1f}%)")
                        st.write(f"• **Lower 25% (≤ {r25_l:.2f}):** {c_low25:,} pts ({(c_low25/t_pts)*100:.1f}%)")
                        st.write(f"• **Upper 25% (≥ {r75_h:.2f}):** {c_high25:,} pts ({(c_high25/t_pts)*100:.1f}%)")
                        st.write(f"• **Upper 10% (≥ {r90_h:.2f}):** {c_high10:,} pts ({(c_high10/t_pts)*100:.1f}%)")

                    with grid_c2:
                        fig = draw_histogram(m_series, f"{m_name} {l_name}: {comp_metric}")
                        st.pyplot(fig)

                    with grid_c3:
                        img_file = get_uploaded_image(b_name, comp_metric)
                        if img_file:
                            st.image(img_file, caption=f"{m_name} {l_name}", use_container_width=True)
                        else:
                            st.info("Upload matching image files in sidebar to view floorplan visual.")
                    
                    st.divider()
        else:
            st.warning("Please select at least one Mall / Level from the filter above.")

    # -------------------------------------------------------------------------
    # TAB 3: AGGREGATED EXPORT
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Aggregated Numerical Calculations (All Analyses)")
        st.dataframe(summary_df, use_container_width=True)
        
        csv_data = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export All VGA Numerical Calculations to CSV",
            data=csv_data,
            file_name="aggregated_vga_summary_calculations.csv",
            mime="text/csv"
        )
else:
    st.info("👈 Upload CSV file(s) and VGA Images in the sidebar to start analysis.")
