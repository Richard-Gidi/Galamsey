import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from datetime import datetime, timedelta
import numpy as np
import random
import openai  # or another LLM API
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Water Quality Analysis Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: bold;
    }
    .info-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# Create parameter descriptions and health implications dictionary
parameter_info = {
    'As': {
        'full_name': 'Arsenic',
        'description': 'A toxic metalloid often found in groundwater. Can come from natural deposits or industrial/agricultural pollution.',
        'unit': 'mg/L',
        'health_implications': 'Chronic exposure can cause skin lesions, cancer, cardiovascular disease, and diabetes. Acute exposure can cause vomiting, abdominal pain, and diarrhea.',
        'who_health_implication': 'Exceeding the WHO limit (0.05 mg/L) increases cancer risk by approximately 1 in 10,000.',
        'epa_health_implication': 'Exceeding the EPA limit (0.01 mg/L) increases risk of cancer and skin disorders.',
        'ghana_health_implication': 'Exceeding Ghana\'s standard (0.01 mg/L) poses significant health risks to local populations.'
    },
    'Cd': {
        'full_name': 'Cadmium',
        'description': 'A toxic heavy metal released through mining, industrial processes, and from phosphate fertilizers.',
        'unit': 'mg/L',
        'health_implications': 'Accumulates in kidneys and can cause kidney damage. Also associated with bone demineralization and increased cancer risk.',
        'who_health_implication': 'Exceeding the WHO limit (0.005 mg/L) increases risk of kidney dysfunction.',
        'epa_health_implication': 'Exceeding the EPA limit (0.005 mg/L) can lead to kidney damage and possible skeletal effects.',
        'ghana_health_implication': 'Exceeding Ghana\'s stricter standard (0.003 mg/L) may lead to bioaccumulation in vital organs.'
    },
    'Cr': {
        'full_name': 'Chromium',
        'description': 'A heavy metal that exists in various forms. Chromium(VI) is highly toxic, while Chromium(III) is an essential nutrient.',
        'unit': 'mg/L',
        'health_implications': 'Chromium(VI) can cause skin irritation, ulcers, and is carcinogenic. Can damage liver and kidney function.',
        'who_health_implication': 'Exceeding the WHO limit (0.05 mg/L) increases risk of gastrointestinal issues and potentially cancer.',
        'epa_health_implication': 'Exceeding the EPA limit (0.05 mg/L) may cause allergic dermatitis and increase cancer risk.','ghana_health_implication': 'Exceeding Ghana\'s standard (0.05 mg/L) poses health risks particularly in mining areas.'
    },
    'Pb': {
        'full_name': 'Lead',
        'description': 'A highly toxic metal that can enter water through corroded plumbing, industrial discharge, or mining activities.',
        'unit': 'mg/L',
        'health_implications': 'Particularly harmful to children, causing developmental issues, lower IQ, and behavioral problems. In adults, can cause hypertension, kidney damage, and reproductive issues.',
        'who_health_implication': 'Exceeding the WHO limit (0.05 mg/L) has severe neurodevelopmental impacts, especially in children.',
        'epa_health_implication': 'Exceeding the EPA limit (0.015 mg/L) can impair cognitive development and cause cardiovascular effects.',
        'ghana_health_implication': 'Exceeding Ghana\'s standard (0.01 mg/L) poses serious health risks, particularly in areas with mining activities.'
    },
    'pH': {
        'full_name': 'pH',
        'description': 'A measure of how acidic or basic water is, on a scale from 0 to 14. 7 is neutral, below 7 is acidic, above 7 is basic.',
        'unit': 'pH units',
        'health_implications': 'Extreme pH values can cause irritation to eyes, skin, and mucous membranes. Acidic water can also leach metals from plumbing and fixtures.',
        'who_health_implication': 'Outside the WHO range (6.5-8.5) can affect disinfection efficiency and cause corrosion issues.',
        'epa_health_implication': 'Outside the EPA range (6.5-8.5) may cause metallic taste and plumbing damage.',
        'ghana_health_implication': 'Outside Ghana\'s standard range (6.5-8.5) affects water palatability and infrastructure.'
    },
    'TDS': {
        'full_name': 'Total Dissolved Solids',
        'description': 'A measure of all inorganic and organic substances dissolved in water, including minerals, salts, and metals.',
        'unit': 'mg/L',
        'health_implications': 'High levels can give water a bad taste and may indicate the presence of harmful contaminants. Very low levels might lack essential minerals.',
        'who_health_implication': 'Exceeding the WHO limit (1000 mg/L) affects taste, and may indicate high levels of specific harmful ions.',
        'epa_health_implication': 'Exceeding the EPA limit (500 mg/L) causes hardness, scale deposits, and bitter taste.',
        'ghana_health_implication': 'Exceeding Ghana\'s standard (1000 mg/L) affects palatability and household appliance efficiency.'
    },
    'Conductivity': {
        'full_name': 'Electrical Conductivity',
        'description': 'Measures water\'s ability to conduct electrical current, indicating the concentration of dissolved ions.',
        'unit': 'μS/cm',
        'health_implications': 'Not directly a health concern, but high conductivity indicates high dissolved solids which could contain harmful substances.',
        'who_health_implication': 'Exceeding the WHO guideline (1000 μS/cm) suggests high mineral content requiring further investigation.',
        'epa_health_implication': 'Exceeding the EPA guideline (1000 μS/cm) indicates elevated salt content that may be unpalatable.',
        'ghana_health_implication': 'Exceeding Ghana\'s standard (1000 μS/cm) suggests potential contamination from industrial or mining activities.'
    },
    'Hardness': {
        'full_name': 'Total Hardness',
        'description': 'A measure of dissolved calcium and magnesium in water, expressed as calcium carbonate.',
        'unit': 'mg/L as CaCO₃',
        'health_implications': 'Hard water is not a health concern and may contribute beneficial calcium and magnesium to diet. Very hard water can cause scaling in pipes and appliances.',
        'who_health_implication': 'Exceeding the WHO recommendation (500 mg/L) causes scale formation but has minimal health impact.',
        'epa_health_implication': 'Exceeding the EPA guideline (500 mg/L) leads to mineral buildup in plumbing.','ghana_health_implication': 'Exceeding Ghana\'s standard (500 mg/L) reduces soap effectiveness and increases scale formation.'
    },
    'Ca_Hardness': {
        'full_name': 'Calcium Hardness',
        'description': 'The portion of water hardness attributed to calcium ions.',
        'unit': 'mg/L as CaCO₃',
        'health_implications': 'Contributes essential calcium to diet. Very high levels can contribute to kidney stones in susceptible individuals.',
        'who_health_implication': 'No specific limit, but contributes to total hardness guidelines.',
        'epa_health_implication': 'No specific limit, but contributes to scaling issues when elevated.',
        'ghana_health_implication': 'No specific limit in Ghana standards, but affects water quality characteristics.'
    },
    'Mg_Hardness': {
        'full_name': 'Magnesium Hardness',
        'description': 'The portion of water hardness attributed to magnesium ions.',
        'unit': 'mg/L as CaCO₃',
        'health_implications': 'Contributes essential magnesium to diet. Very high levels can have laxative effects in some people.',
        'who_health_implication': 'No specific limit, but contributes to total hardness guidelines.',
        'epa_health_implication': 'No specific limit, but contributes to bitter taste when elevated.',
        'ghana_health_implication': 'No specific limit in Ghana standards, but affects water quality characteristics.'
    },
    'Heavy_Metal_Index': {
        'full_name': 'Heavy Metal Index',
        'description': 'A composite index calculated from multiple heavy metal concentrations, normalized to their respective standards.',
        'unit': 'Dimensionless',
        'health_implications': 'Higher values indicate greater potential for adverse health effects from heavy metals. Values >1 suggest concentrations exceeding safety standards.',
        'who_health_implication': 'Values >1 indicate concentrations exceeding WHO guidelines for multiple heavy metals.',
        'epa_health_implication': 'Values >1 suggest potential violation of EPA standards for heavy metals.',
        'ghana_health_implication': 'Values >1 indicate non-compliance with Ghana\'s heavy metal standards.'
    },
    'Water_Quality_Score': {
        'full_name': 'Water Quality Score',
        'description': 'A comprehensive score from 0-100 that accounts for heavy metals, pH, and TDS. Higher scores indicate better water quality.',
        'unit': 'Score (0-100)',
        'health_implications': 'Lower scores indicate higher potential for adverse health effects from multiple water quality parameters.',
        'who_health_implication': 'Scores <60 suggest significant deviation from WHO guidelines across multiple parameters.',
        'epa_health_implication': 'Scores <70 indicate potential non-compliance with EPA standards.',
        'ghana_health_implication': 'Scores <65 suggest water may not meet Ghana\'s water quality requirements.'
    }
}

# Statistical metrics explanations
statistical_metrics = {
    'Mean': 'The average value of the parameter across all samples. Represents the central tendency of the data.',
    'Standard Deviation': 'Measures the amount of variation or dispersion in the dataset. Higher values indicate greater spread from the mean.',
    'Range': 'The difference between the maximum and minimum values, showing the full spread of the data.',
    'Coefficient of Variation': 'Standard deviation divided by mean (expressed as percentage). Allows comparison of variability between different parameters regardless of their measurement scales.',
    'Skewness': {
        'description': 'Measures the asymmetry of the probability distribution. Indicates which direction the data is "tailed".',
        'interpretation': {
            'positive': 'Positive skewness (>0.5) indicates a distribution with a longer right tail, meaning most values are concentrated on the left with extreme values to the right.',
            'negative': 'Negative skewness (<-0.5) indicates a distribution with a longer left tail, meaning most values are concentrated on the right with extreme values to the left.',
            'neutral': 'Values between -0.5 and 0.5 suggest a relatively symmetrical distribution.'
        }
    },
    'Kurtosis': {
        'description': 'Measures the "tailedness" of the probability distribution. Indicates the presence of outliers.',
        'interpretation': {
            'high': 'High kurtosis (>3) indicates a distribution with heavier tails and more outliers compared to a normal distribution.',
            'low': 'Low kurtosis (<3) indicates a distribution with lighter tails and fewer outliers compared to a normal distribution.',
            'normal': 'A value of 3 corresponds to a normal distribution.'
        }
    },
    'Jarque-Bera': {
        'description': 'A test statistic for checking if the data have the skewness and kurtosis matching a normal distribution.',
        'interpretation': {
            'high': 'Higher values (typically with p-value <0.05) suggest the data is not normally distributed.',
            'low': 'Lower values suggest the data may follow a normal distribution.'
        }
    },
    'Shapiro-Wilk': {
        'description': 'A test for normality. Tests the null hypothesis that the data was drawn from a normal distribution.',
        'interpretation': {
            'high': 'Higher values (closer to 1) suggest the data follows a normal distribution.',
            'low': 'Lower values suggest the data does not follow a normal distribution.'
        }
    },
    'IQR': 'Interquartile Range. The difference between the 75th and 25th percentiles. Represents the middle 50% of the data and is resistant to outliers.',
    '95th Percentile': 'The value below which 95% of the observations may be found. Often used to assess the upper limit of "normal" data range.',
    '5th Percentile': 'The value below which 5% of the observations may be found. Often used to assess the lower limit of "normal" data range.',
    'CV': 'Coefficient of Variation. Standard deviation divided by mean (expressed as percentage). Allows comparison of variability between different parameters regardless of their measurement scales.'
}




# Load data
data = pd.read_excel("DATASET_v0.1.xlsx")

# Create standards data
standards = {
    'Standard': ['WHO', 'US EPA', 'Ghana Standard'],
    'As': [0.05, 0.01, 0.01],
    'Cd': [0.005, 0.005, 0.003],
    'Cr': [0.05, 0.05, 0.05],
    'Pb': [0.05, 0.015, 0.01],
    'pH_min': [6.5, 6.5, 6.5],
    'pH_max': [8.5, 8.5, 8.5],
    'TDS': [1000, 500, 1000],
    'Conductivity': [1000, 1000, 1000],
    'Hardness': [500, 500, 500],
}

df = pd.DataFrame(data)
standards_df = pd.DataFrame(standards)

# Add derived metrics
df['Heavy_Metal_Index'] = (df['As']/standards_df.loc[0, 'As'] + 
                          df['Cd']/standards_df.loc[0, 'Cd'] + 
                          df['Cr']/standards_df.loc[0, 'Cr'] + 
                          df['Pb']/standards_df.loc[0, 'Pb'])/4

df['Water_Quality_Score'] = 100 - (df['Heavy_Metal_Index'] * 20 + 
                                 abs(df['pH'] - 7) * 10 + 
                                 df['TDS']/standards_df.loc[0, 'TDS'] * 10)

# Title and Introduction
st.title("🌊 Water Quality Analysis Dashboard")
st.markdown("""
<div style='background-color: #ffffff; color: #000000; padding: 20px; border-radius: 10px; border: 2px solid #3498db; margin-bottom: 20px;'>
    <h3 style='color: #2c3e50; margin-bottom: 15px;'>Comprehensive Water Quality Analysis</h3>
    <p style='color: #2c3e50; line-height: 1.6;'>
        This dashboard analyzes water quality parameters across various rivers and a Galamsey pit,
        comparing them against WHO, US EPA, and Ghana Standards.
    </p>
    <h4 style='color: #2c3e50; margin-top: 20px;'>Key Parameters Explained</h4>
    <ul style='color: #2c3e50; line-height: 1.6;'>
        <li><strong>Heavy Metals (As, Cd, Cr, Pb)</strong>: Toxic elements that can cause serious health effects, including cancer, organ damage, and developmental issues.</li>
        <li><strong>pH</strong>: Measures how acidic or basic the water is. Extreme values can indicate pollution and affect water treatment effectiveness.</li>
        <li><strong>TDS</strong>: Total Dissolved Solids indicate the amount of minerals and salts in the water, affecting taste and usefulness.</li>
        <li><strong>Hardness</strong>: Mainly caused by calcium and magnesium, affecting water's ability to form lather with soap and creating scale in pipes.</li>
        <li><strong>Water Quality Score</strong>: A custom metric combining various parameters to assess overall water quality.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Parameter category selector
category_options = {
    'Heavy Metals': ['As', 'Cd', 'Cr', 'Pb'],
    'Water Properties': ['pH', 'TDS', 'Conductivity'],
    'Hardness Measures': ['Hardness', 'Ca_Hardness', 'Mg_Hardness'],
    'Derived Metrics': ['Heavy_Metal_Index', 'Water_Quality_Score']
}

category_group = st.sidebar.selectbox(
    "Select Parameter Group",
    options=list(category_options.keys())
)

parameter = st.sidebar.selectbox(
    "Select Parameter for Analysis",
    options=category_options[category_group]
)

# Parameter information and health implications
if parameter in parameter_info:
    st.sidebar.markdown(f"""
    <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px;'>
        <h4>{parameter_info[parameter]['full_name']} ({parameter})</h4>
        <p><strong>Description:</strong> {parameter_info[parameter]['description']}</p>
        <p><strong>Unit:</strong> {parameter_info[parameter]['unit']}</p>
        <p><strong>Health Implications:</strong> {parameter_info[parameter]['health_implications']}</p>
        <hr>
        <p><strong>Standard Exceedance Implications:</strong></p>
        <ul>
            <li><strong>WHO:</strong> {parameter_info[parameter]['who_health_implication']}</li>
            <li><strong>US EPA:</strong> {parameter_info[parameter]['epa_health_implication']}</li>
            <li><strong>Ghana:</strong> {parameter_info[parameter]['ghana_health_implication']}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# View options
ranking_view = st.sidebar.radio(
    "Data View",
    options=["All Locations", "Best 6", "Worst 6", "Custom Range"]
)

if ranking_view == "Custom Range":
    num_locations = st.sidebar.slider("Number of locations to show", 2, len(df), 6)

# Standard selection
selected_standard = st.sidebar.selectbox(
    "Select Standard for Comparison",
    options=['WHO', 'US EPA', 'Ghana Standard'],
    index=0
)

# Visualization type
viz_type = st.sidebar.selectbox(
    "Visualization Type",
    ["Bar Chart", "Radar Plot", "Heat Map", "Box Plot", "Violin Plot"]
)


# Calculate rankings
def get_parameter_ranking(df, parameter, standard_row):
    if parameter == 'pH':
        df['ranking_value'] = df[parameter].apply(lambda x: min(abs(x - 6.5), abs(x - 8.5)))
    elif parameter == 'Heavy_Metal_Index':
        df['ranking_value'] = df[parameter]
    elif parameter == 'Water_Quality_Score':
        df['ranking_value'] = -df[parameter]  # Higher score is better
    else:
        df['ranking_value'] = df[parameter]
    
    df_sorted = df.sort_values('ranking_value', ascending=True if parameter in ['pH', 'Water_Quality_Score'] else False)
    return df_sorted

ranked_df = get_parameter_ranking(df.copy(), parameter, standards_df.loc[0])

# Filter locations based on ranking view
if ranking_view == "Best 6":
    ranked_df = ranked_df.head(6)
elif ranking_view == "Worst 6":
    ranked_df = ranked_df.tail(6)
elif ranking_view == "Custom Range":
    ranked_df = ranked_df.head(num_locations)

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["📊 Main Analysis", "🔍 Detailed Insights", "📈 Trends & Patterns"])

with tab1:
    # Main visualization
    if viz_type == "Bar Chart":
        fig = px.bar(
            ranked_df,
            x='Sample',
            y=parameter,
            color=parameter,
            title=f"{parameter} Levels by Location",
            color_continuous_scale='Viridis'
        )
        
        # Add standard reference lines if applicable
        if parameter in standards_df.columns:
            standard_value = standards_df.loc[standards_df['Standard'] == selected_standard, parameter].values[0]
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=standard_value,
                x1=len(ranked_df)-0.5,
                y1=standard_value,
                line=dict(color="red", width=2, dash="dash")
            )
            fig.add_annotation(
                x=len(ranked_df)-1,
                y=standard_value*1.05,
                text=f"{selected_standard} Standard: {standard_value}",
                showarrow=False,
                font=dict(color="red")
            )
    elif viz_type == "Radar Plot":
        fig = go.Figure()
        for location in ranked_df['Sample']:
            location_data = ranked_df[ranked_df['Sample'] == location]
            if parameter in category_options['Heavy Metals']:
                params = category_options['Heavy Metals']
            else:
                params = [parameter]
            fig.add_trace(go.Scatterpolar(
                r=[location_data[p].iloc[0] for p in params],
                theta=params,
                name=location,
                fill='toself'
            ))
    elif viz_type == "Heat Map":
        correlation_matrix = ranked_df[category_options[category_group]].corr()
        fig = px.imshow(
            correlation_matrix,
            title="Parameter Correlation Matrix",
            color_continuous_scale='RdBu'
        )
    elif viz_type in ["Box Plot", "Violin Plot"]:
        fig = px.violin(
            ranked_df,
            y=parameter,
            box=True,
            points="all",
            title=f"Distribution of {parameter}"
        ) if viz_type == "Violin Plot" else px.box(
            ranked_df,
            y=parameter,
            title=f"Distribution of {parameter}"
        )
        
        # Add standard reference lines if applicable
        if parameter in standards_df.columns:
            standard_value = standards_df.loc[standards_df['Standard'] == selected_standard, parameter].values[0]
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=standard_value,
                x1=0.5,
                y1=standard_value,
                line=dict(color="red", width=2, dash="dash")
            )
            fig.add_annotation(
                x=0,
                y=standard_value*1.05,
                text=f"{selected_standard} Standard: {standard_value}",
                showarrow=False,
                font=dict(color="red")
            )

            fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=True,
        height=600
    )
        st.plotly_chart(fig, use_container_width=True)

    # Statistical insights with explanations
    st.subheader("Statistical Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean Value",
            f"{ranked_df[parameter].mean():.3f}",
            f"{ranked_df[parameter].mean() - df[parameter].mean():.3f}"
        )
        st.markdown(f"<div class='metric-explanation'>{statistical_metrics['Mean']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.metric("Standard Deviation", f"{ranked_df[parameter].std():.3f}")
        st.markdown(f"<div class='metric-explanation'>{statistical_metrics['Standard Deviation']}</div>", unsafe_allow_html=True)
    
    with col3:
        st.metric("Range", f"{ranked_df[parameter].max() - ranked_df[parameter].min():.3f}")
        st.markdown(f"<div class='metric-explanation'>{statistical_metrics['Range']}</div>", unsafe_allow_html=True)
    
    with col4:
        cv = ranked_df[parameter].std() / ranked_df[parameter].mean() * 100
        st.metric(
            "Coefficient of Variation",
            f"{cv:.1f}%"
        )
        st.markdown(f"<div class='metric-explanation'>{statistical_metrics['Coefficient of Variation']}</div>", unsafe_allow_html=True)
    
    # Health implications based on exceedances
    if parameter in standards_df.columns:
        standard_value = standards_df.loc[standards_df['Standard'] == selected_standard, parameter].values[0]
        exceedances = sum(ranked_df[parameter] > standard_value)
        
        if exceedances > 0:
            st.markdown(f"""
            <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 5px solid #f44336;'>
                <h4>Health Risk Alert</h4>
                <p>⚠️ <strong>{exceedances}</strong> out of {len(ranked_df)} locations exceed the {selected_standard} standard ({standard_value} {parameter_info[parameter]['unit']}) for {parameter_info[parameter]['full_name']}.</p>
                <p><strong>Health Implications:</strong> {parameter_info[parameter][selected_standard.lower().replace(' ', '_') + '_health_implication']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border-left: 5px solid #4caf50;'>
                <h4>Safe Levels</h4>
                <p>✅ All locations meet the {selected_standard} standard ({standard_value} {parameter_info[parameter]['unit']}) for {parameter_info[parameter]['full_name']}.</p>
            </div>
            """, unsafe_allow_html=True)


            with tab2:
                # Parameter distribution
                st.subheader("Parameter Deep Dive")
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=ranked_df[parameter],
                    nbinsx=20,
                    name="Distribution"
                ))
                fig.add_trace(go.Violin(
                    y=ranked_df[parameter],
                    name="Violin Plot",
                    side="positive",
                    box={"visible": True},
                    meanline={"visible": True}
                ))
                fig.update_layout(
                    title=f"{parameter} Distribution Analysis",
                    showlegend=True,
                    height=500,
                    xaxis_title="Frequency / Distribution",
                    yaxis_title=parameter
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Add distribution interpretation
    skewness = ranked_df[parameter].skew()
    kurtosis = ranked_df[parameter].kurtosis()
    
    if abs(skewness) < 0.5:
        skew_interp = "The distribution is approximately symmetric."
    elif skewness > 0:
        skew_interp = "The distribution is positively skewed (right-tailed), with more values below the mean and a few extreme high values."
    else:
        skew_interp = "The distribution is negatively skewed (left-tailed), with more values above the mean and a few extreme low values."
    
    if abs(kurtosis) < 0.5:
        kurt_interp = "The distribution has a normal 'peakedness' (similar to a normal distribution)."
    elif kurtosis > 0:
        kurt_interp = "The distribution is leptokurtic (more peaked than a normal distribution) with heavier tails, indicating frequent extreme values."
    else:
        kurt_interp = "The distribution is platykurtic (less peaked than a normal distribution) with lighter tails, indicating infrequent extreme values."
    
    st.markdown(f"""
    <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #dee2e6;'>
        <h4>Distribution Interpretation</h4>
        <p><strong>Skewness:</strong> {skewness:.3f} - {skew_interp}</p>
        <p><strong>Kurtosis:</strong> {kurtosis:.3f} - {kurt_interp}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Correlation analysis
    if parameter in category_options['Heavy Metals']:
        correlation_matrix = ranked_df[category_options['Heavy Metals']].corr()
        fig = px.imshow(
            correlation_matrix,
            title="Heavy Metals Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add correlation interpretation
        high_corr_pairs = []
        for i, metal1 in enumerate(correlation_matrix.columns):
            for j, metal2 in enumerate(correlation_matrix.columns):
                if i < j and abs(correlation_matrix.iloc[i, j]) > 0.5:
                    high_corr_pairs.append((metal1, metal2, correlation_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            st.markdown("""
            <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #dee2e6;'>
                <h4>Correlation Interpretation</h4>
                <p>Strong correlations were detected between:</p>
                <ul>
            """, unsafe_allow_html=True)
            
            for metal1, metal2, corr in high_corr_pairs:
                direction = "positive" if corr > 0 else "negative"
                interpretation = f"suggesting a common source or pathway" if corr > 0 else "suggesting different sources or chemical interactions"
                st.markdown(f"<li><strong>{metal1} and {metal2}:</strong> {corr:.3f} ({direction} correlation) - {interpretation}</li>", unsafe_allow_html=True)
            
            st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #dee2e6;'>
                <h4>Correlation Interpretation</h4>
                <p>No strong correlations (>0.5) were detected between heavy metals, suggesting independent sources or pathways.</p>
                        
                        </div>
            """, unsafe_allow_html=True)

with tab3:
    # Trends and patterns
    st.subheader("Trend Analysis")
    
    # Multi-parameter comparison
    if parameter in category_options['Heavy Metals']:
        fig = px.scatter_matrix(
            ranked_df,
            dimensions=category_options['Heavy Metals'],
            title="Heavy Metals Relationships"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        st.markdown("""
        <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #dee2e6;'>
            <h4>Relationship Interpretation</h4>
            <p>The scatter matrix above shows relationships between different heavy metals:</p>
            <ul>
                <li>Diagonal plots show the distribution of each metal</li>
                <li>Off-diagonal plots show relationships between pairs of metals</li>
                <li>Clustered patterns may indicate similar sources of contamination</li>
                <li>Outliers may represent specific contamination events or hotspots</li>
            </ul>
            <p>Look for patterns that might suggest mining impacts, industrial pollution, or natural geological sources.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Time series pattern (simulated)
    st.subheader("Temporal Pattern Simulation")
    # Create time periods matching locations in ranked_df
    times = pd.date_range(start='2023-01-01', periods=len(ranked_df), freq='M')
    temporal_data = pd.DataFrame({
        'Date': times,
        'Value': ranked_df[parameter].values,
        'Location': ranked_df['Sample'].values
    })
    
    fig = px.line(
        temporal_data,
        x='Date',
        y='Value',
        color='Location',
        title=f"{parameter} Temporal Pattern",
        markers=True
    )
    
    # Add standard reference if applicable
    if parameter in standards_df.columns:
        standard_value = standards_df.loc[standards_df['Standard'] == selected_standard, parameter].values[0]
        fig.add_shape(
            type="line",
            x0=times[0],
            y0=standard_value,
            x1=times[-1],
            y1=standard_value,
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x=times[-1],
            y=standard_value*1.05,
            text=f"{selected_standard} Standard: {standard_value}",
            showarrow=False,
            font=dict(color="red")
        )
    
    fig.update_layout(
        xaxis_title="Time Period",
        yaxis_title=f"{parameter} Value",
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add trend description
    trend_direction = "increasing" if temporal_data['Value'].iloc[-1] > temporal_data['Value'].iloc[0] else "decreasing"
    
    # Calculate trend statistics
    x = np.arange(len(temporal_data))
    y = temporal_data['Value'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Interpret trend significance
    if p_value < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"
    
    st.markdown(f"""
    <div style='background-color: #ffffff; color: #000000; padding: 20px; border-radius: 10px; border: 2px solid #3498db; margin: 15px 0;'>
        <h4 style='color: #2c3e50; margin-bottom: 15px;'>Temporal Trend Analysis</h4>
        <ul style='color: #2c3e50; line-height: 1.6;'>
            <li><strong>Overall trend:</strong> {trend_direction} (slope = {slope:.4f} units per month)</li>
            <li><strong>Trend significance:</strong> The trend is {significance} (p-value = {p_value:.4f})</li>
            <li><strong>R-squared:</strong> {r_value**2:.3f} - {r_value**2*100:.1f}% of variation explained by time</li>
            <li><strong>Maximum value:</strong> {temporal_data['Value'].max():.3f} ({temporal_data.loc[temporal_data['Value'].idxmax(), 'Location']})</li>
            <li><strong>Minimum value:</strong> {temporal_data['Value'].min():.3f} ({temporal_data.loc[temporal_data['Value'].idxmin(), 'Location']})</li>
            <li><strong>Trend period:</strong> {times[0].strftime('%B %Y')} to {times[-1].strftime('%B %Y')}</li>
        </ul>
        <p style='color: #2c3e50; margin-top: 10px;'>
            <strong>Interpretation:</strong> {
                f"The {trend_direction} trend is {significance}, suggesting that {parameter} levels are systematically {trend_direction} over time. This could indicate {'accumulation of pollutants' if trend_direction == 'increasing' else 'effectiveness of remediation efforts or natural attenuation'}." if parameter in category_options['Heavy Metals'] else
                f"The {trend_direction} trend is {significance}, which may be due to seasonal variations, changes in water flow, or {' increased pollutant inputs' if trend_direction == 'increasing' else ' natural processes or improved water management'}."
            }
        </p>
    </div>
    """, unsafe_allow_html=True)

# Advanced Analysis Section
st.header("🔬 Advanced Analysis")

adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs([
    "📊 Statistical Analysis", 
    "🔍 Anomaly Detection", 
    "📈 Predictive Analysis",
    "🌐 Spatial Analysis"
])

with adv_tab1:
    st.subheader("Advanced Statistical Analysis")
    
    # Create subplots for statistical analysis
    fig = make_subplots(rows=2, cols=2, 
                       subplot_titles=("Distribution", "Q-Q Plot", "Box Plot", "Violin Plot"))
    
    # Distribution plot
    fig.add_trace(go.Histogram(x=ranked_df[parameter], name="Distribution"), row=1, col=1)
    
    # Q-Q Plot
    qq = stats.probplot(ranked_df[parameter], dist="norm")
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name="Q-Q Plot"), row=1, col=2)
    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1], mode='lines', name="Theoretical"), row=1, col=2)
    
    # Box plot
    fig.add_trace(go.Box(y=ranked_df[parameter], name="Box Plot"), row=2, col=1)
    
    # Violin plot
    fig.add_trace(go.Violin(y=ranked_df[parameter], name="Violin Plot"), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Add Q-Q plot interpretation
    shapiro_stat, shapiro_p = stats.shapiro(ranked_df[parameter])
    
    if shapiro_p < 0.05:
        normality_interp = "The data significantly deviates from a normal distribution."
    else:
        normality_interp = "The data approximately follows a normal distribution."
    
    st.markdown(f"""
    <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #dee2e6;'>
        <h4>Normality Analysis Interpretation</h4>
        <p><strong>Q-Q Plot:</strong> Points following the diagonal line suggest a normal distribution. Deviations indicate non-normality.</p>
        <p><strong>Shapiro-Wilk Test:</strong> p-value = {shapiro_p:.4f} - {normality_interp}</p>
        <p><strong>Implication:</strong> {
            "Non-normal distribution suggests the presence of outliers, multiple data populations, or skewed processes affecting this parameter. Parametric statistical tests should be used with caution." if shapiro_p < 0.05 else
            "Normal distribution suggests a consistent, uniform process affecting this parameter. Parametric statistical tests are appropriate."
        }</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced statistics with interpretations
    st.markdown("""
    <div style='background-color: #ffffff; color: #000000; padding: 20px; border-radius: 10px; border: 2px solid #3498db; margin: 15px 0;'>
        <h4 style='color: #2c3e50;'>Advanced Statistical Metrics</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate statistics
    skewness = ranked_df[parameter].skew()
    kurtosis = ranked_df[parameter].kurtosis()
    jarque_bera, jb_p = stats.jarque_bera(ranked_df[parameter])
    shapiro_stat, shapiro_p = stats.shapiro(ranked_df[parameter])
    iqr = ranked_df[parameter].quantile(0.75) - ranked_df[parameter].quantile(0.25)
    percentile_95 = ranked_df[parameter].quantile(0.95)
    percentile_05 = ranked_df[parameter].quantile(0.05)
    cv = ranked_df[parameter].std() / ranked_df[parameter].mean() * 100
    
    # Interpret skewness
    if abs(skewness) < 0.5:
        skew_interpretation = "The data is approximately symmetric."
    elif skewness > 0.5:
        skew_interpretation = "The data is positively skewed with a longer right tail, indicating some unusually high values."
    else:
        skew_interpretation = "The data is negatively skewed with a longer left tail, indicating some unusually low values."
    
    # Interpret kurtosis
    if abs(kurtosis) < 0.5:
        kurt_interpretation = "The data has approximately normal tails."
    elif kurtosis > 0.5:
        kurt_interpretation = "The data has heavy tails with more extreme values than a normal distribution."
    else:
        kurt_interpretation = "The data has light tails with fewer extreme values than a normal distribution."
    
    # Create columns for statistics with interpretations
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Skewness", f"{skewness:.3f}")

        st.markdown(f"<div class='metric-explanation'>{skew_interpretation}</div>", unsafe_allow_html=True)
        
        st.metric("Kurtosis", f"{kurtosis:.3f}")
        st.markdown(f"<div class='metric-explanation'>{kurt_interpretation}</div>", unsafe_allow_html=True)
    with col2:
        st.metric("Jarque-Bera", f"{jarque_bera:.3f}")
        st.markdown(f"<div class='metric-explanation'>p-value: {jb_p:.4f} - {'Non-normal distribution' if jb_p < 0.05 else 'Normal distribution'}</div>", unsafe_allow_html=True)
        
        st.metric("Shapiro-Wilk", f"{shapiro_stat:.3f}")
        st.markdown(f"<div class='metric-explanation'>p-value: {shapiro_p:.4f} - {'Non-normal distribution' if shapiro_p < 0.05 else 'Normal distribution'}</div>", unsafe_allow_html=True)
    with col3:
        st.metric("IQR", f"{iqr:.3f}")
        st.markdown(f"<div class='metric-explanation'>Range containing the middle 50% of the data.</div>", unsafe_allow_html=True)
        
        st.metric("95th Percentile", f"{percentile_95:.3f}")
        st.markdown(f"<div class='metric-explanation'>95% of measurements are below this value.</div>", unsafe_allow_html=True)
    with col4:
        st.metric("5th Percentile", f"{percentile_05:.3f}")
        st.markdown(f"<div class='metric-explanation'>5% of measurements are below this value.</div>", unsafe_allow_html=True)
        
        st.metric("CV", f"{cv:.1f}%")
        st.markdown(f"<div class='metric-explanation'>{'High variability' if cv > 30 else 'Moderate variability' if cv > 15 else 'Low variability'} in the data.</div>", unsafe_allow_html=True)

with adv_tab2:
    st.subheader("Anomaly Detection")
    
    # Preparation for anomaly detection
    X = ranked_df[category_options[category_group]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Model explanation
    st.markdown("""
    <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #dee2e6;'>
        <h4>Anomaly Detection Model</h4>
        <p>This analysis uses <strong>Isolation Forest</strong> algorithm to detect anomalies by isolating observations through random feature splitting.</p>
        <p>Isolation Forest works particularly well for detecting outliers as they typically require fewer splits to be isolated from other observations.</p>
        <p>Points identified as anomalies may represent:</p>
        <ul>
            <li>Sampling or measurement errors</li>
            <li>Actual contamination events or hotspots</li>
            <li>Different water sources or unusual geological influences</li>
            <li>Potential point sources of pollution</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)
    
    # Create anomaly visualization
    fig = go.Figure()
    
    # Add normal points
    normal_mask = anomalies == 1
    fig.add_trace(go.Scatter(
        x=ranked_df.index[normal_mask],
        y=ranked_df[parameter][normal_mask],
        mode='markers',
        name='Normal',
        marker=dict(color='blue')
    ))
    
    # Add anomalies
    anomaly_mask = anomalies == -1
    fig.add_trace(go.Scatter(
        x=ranked_df.index[anomaly_mask],
        y=ranked_df[parameter][anomaly_mask],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title=f"Anomaly Detection for {parameter}",
        xaxis_title="Sample Index",
        yaxis_title=parameter,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display anomaly details
    if any(anomaly_mask):
        anomaly_df = ranked_df[anomaly_mask]
        
        # Interpret each anomaly
        anomaly_interpretations = []
        for idx, row in anomaly_df.iterrows():
            val = row[parameter]
            mean_val = ranked_df[parameter].mean()
            std_val = ranked_df[parameter].std()

            z_score = (val - mean_val) / std_val
            
            if parameter in standards_df.columns:
                standard_value = standards_df.loc[standards_df['Standard'] == selected_standard, parameter].values[0]
                exceedance = val > standard_value
                exceedance_text = f" and exceeds the {selected_standard} standard" if exceedance else ""
            else:
                exceedance_text = ""
                
            interpretation = f"The value of {val:.3f} is {abs(z_score):.1f} standard deviations {'above' if z_score > 0 else 'below'} the mean{exceedance_text}."
            anomaly_interpretations.append(interpretation)
        
        st.markdown("""
        <div style='background-color: #ffffff; color: #000000; padding: 20px; border-radius: 10px; border: 2px solid #e74c3c; margin: 15px 0;'>
            <h4 style='color: #2c3e50;'>Detected Anomalies</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(anomaly_df.style.background_gradient(cmap='Reds'))
        
        for i, (idx, row) in enumerate(anomaly_df.iterrows()):
            st.markdown(f"""
            <div style='background-color: #ffffff; color: #000000; padding: 10px; border-radius: 8px; margin-top: 10px; border-left: 3px solid #e74c3c;'>
                <p><strong>Anomaly {i+1} ({row['Sample']}):</strong> {anomaly_interpretations[i]}</p>
                <p><strong>Possible causes:</strong> {
                    "Point source pollution, industrial discharge, or recent mining activity." if parameter in category_options['Heavy Metals'] and row[parameter] > ranked_df[parameter].mean() else
                    "Natural geological variations, sampling from different depth, or recent precipitation events." if parameter in category_options['Heavy Metals'] else
                    "Recent rainfall, agricultural runoff, or industrial discharge." if parameter in category_options['Water Properties'] else
                    "Geological variations, mixing of different water sources, or seasonal changes."
                }</p>
                <p><strong>Recommended action:</strong> {
                    "Resample to confirm result and investigate potential nearby pollution sources." if parameter in category_options['Heavy Metals'] and row[parameter] > ranked_df[parameter].mean() else
                    "Check sampling methodology and consider geological context of the sampling location." if parameter in category_options['Heavy Metals'] else
                    "Monitor location more frequently and check for nearby activities affecting water quality." if parameter in category_options['Water Properties'] else
                    "Compare with historical data if available and consider seasonal variation."
                }</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; border: 2px solid #4caf50; margin: 15px 0;'>
            <h4 style='color: #2c3e50;'>No Anomalies Detected</h4>
            <p>All data points appear to be consistent with the overall distribution. This suggests relatively uniform conditions across sampling locations.</p>
        </div>
        """, unsafe_allow_html=True)

        with adv_tab3:
            st.subheader("Predictive Analysis")
    
            # Explanation of the prediction approach
            st.markdown("""
            <div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 8px; margin-top: 15px; border: 1px solid #dee2e6;'>
                <h4>Predictive Analysis Methodology</h4>
                <p>This analysis uses a simple linear regression model to identify and project trends in water quality parameters.</p>
                <p><strong>Note:</strong> This is a simulation of temporal patterns, as the actual dataset doesn't contain time series information.</p>
                <p>The model helps to:</p>
                <ul>
                    <li>Identify significant trends in water quality parameters</li>
                    <li>Project potential future values if trends continue</li>
                    <li>Assess the statistical significance of observed patterns</li>
                    <li>Establish confidence intervals for predictions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Create time series prediction
            X = np.array(range(len(ranked_df))).reshape(-1, 1)
            y = ranked_df[parameter].values
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
            
            # Create prediction visualization
            fig = go.Figure()
            
            # Add actual data
            fig.add_trace(go.Scatter(
                x=ranked_df.index,
                y=y,
                mode='markers',
                name='Actual'
            ))
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=ranked_df.index,
                y=slope * X.flatten() + intercept,
                mode='lines',
                name='Trend',
                line=dict(dash='dash')
            ))
            
            # Add prediction interval
            y_pred = slope * X.flatten() + intercept
            se = np.sqrt(np.sum((y - y_pred) ** 2) / (len(y) - 2))
            t = stats.t.ppf(0.975, len(y) - 2)
            margin = t * se * np.sqrt(1 + 1/len(y) + (X.flatten() - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
            
            fig.add_trace(go.Scatter(
                x=ranked_df.index,
                y=y_pred + margin,
                mode='lines',
                name='Upper Bound (95%)',
                line=dict(dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                x=ranked_df.index,
                y=y_pred - margin,
                mode='lines',
                name='Lower Bound (95%)',
                line=dict(dash='dot')
            ))
            
            # Add standard reference if applicable
            if parameter in standards_df.columns:
                standard_value = standards_df.loc[standards_df['Standard'] == selected_standard, parameter].values[0]
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=standard_value,
                    x1=len(ranked_df)-1,
                    y1=standard_value,
                    line=dict(color="red", width=2, dash="dash")
                )
                fig.add_annotation(
                    x=len(ranked_df)-1,
                    y=standard_value*1.05,
                    text=f"{selected_standard} Standard: {standard_value}",
                    showarrow=False,
                    font=dict(color="red")
                )
            
            fig.update_layout(
                title=f"Time Series Prediction for {parameter}",
                xaxis_title="Sample Index",
                yaxis_title=parameter,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Extend prediction to future periods
            future_periods = 3  # Number of periods to project into the future
            future_X = np.array(range(len(ranked_df), len(ranked_df) + future_periods)).reshape(-1, 1)
            future_y = slope * future_X.flatten() + intercept
            future_margin = t * se * np.sqrt(1 + 1/len(y) + (future_X.flatten() - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
            
            # Future prediction table
            future_prediction = pd.DataFrame({
                'Period': [f"Future {i+1}" for i in range(future_periods)],
                'Predicted Value': future_y,
                'Lower Bound (95%)': future_y - future_margin,
                'Upper Bound (95%)': future_y + future_margin
            })




    # Display prediction metrics
    st.markdown(f"""
    <div style='background-color: #ffffff; color: #000000; padding: 20px; border-radius: 10px; border: 2px solid #3498db; margin: 15px 0;'>
        <h4 style='color: #2c3e50;'>Prediction Metrics</h4>
        <ul style='color: #2c3e50;'>
            <li><strong>R-squared:</strong> {r_value**2:.3f}</li>
            <li><strong>P-value:</strong> {p_value:.3f}</li>
            <li><strong>Standard Error:</strong> {std_err:.3f}</li>
            <li><strong>Trend Direction:</strong> {'Increasing' if slope > 0 else 'Decreasing'}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with adv_tab4:
    st.subheader("Spatial Analysis")
    
    # Create PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = go.Figure()
    
    # Add PCA scatter plot
    fig.add_trace(go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        mode='markers+text',
        text=ranked_df['Sample'],
        textposition='top center',
        marker=dict(
            size=10,
            color=ranked_df[parameter],
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    # Add variance explained
    fig.add_annotation(
        text=f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%<br>PC2: {pca.explained_variance_ratio_[1]*100:.1f}%",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False
    )
    
    fig.update_layout(
        title="Principal Component Analysis",
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display loading factors
    st.markdown("""
    <div style='background-color: #ffffff; color: #000000; padding: 20px; border-radius: 10px; border: 2px solid #3498db; margin: 15px 0;'>
        <h4 style='color: #2c3e50;'>Principal Component Loadings</h4>
    </div>
    """, unsafe_allow_html=True)
    
    loadings_df = pd.DataFrame(
        pca.components_,
        columns=category_options[category_group],
        index=['PC1', 'PC2']
    )
    st.dataframe(loadings_df.style.background_gradient(cmap='RdBu'))

# Interactive 3D Analysis Section
st.header("🎮 Interactive 3D Analysis")

d3_tab1, d3_tab2 = st.tabs([
    "🌐 3D Parameter Space", 
    "🎯 Clustering Analysis"
])

with d3_tab1:
    st.subheader("3D Parameter Space Visualization")
    
    # Select parameters for 3D visualization
    col1, col2, col3 = st.columns(3)
    with col1:
        x_param = st.selectbox("X-axis Parameter", options=category_options[category_group], index=0)
    with col2:
        y_param = st.selectbox("Y-axis Parameter", options=category_options[category_group], index=1)
    with col3:
        z_param = st.selectbox("Z-axis Parameter", options=category_options[category_group], index=2)
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=ranked_df[x_param],
        y=ranked_df[y_param],
        z=ranked_df[z_param],
        mode='markers+text',
        text=ranked_df['Sample'],
        marker=dict(
            size=8,
            color=ranked_df[parameter],
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=f"3D Parameter Space: {x_param} vs {y_param} vs {z_param}",
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param
        ),
        height=800
    )
    st.plotly_chart(fig, use_container_width=True)

with d3_tab2:
    st.subheader("Advanced Clustering Analysis")
    
    # Clustering options
    clustering_method = st.selectbox(
        "Select Clustering Method",
        ["K-means", "DBSCAN"]
    )
    
    if clustering_method == "K-means":
        n_clusters = st.slider("Number of Clusters", 2, 5, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
    else:  # DBSCAN
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5)
        min_samples = st.slider("Minimum Samples", 2, 5, 3)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
    
    # Create t-SNE visualization
    tsne = TSNE(n_components=2, perplexity=min(5, len(ranked_df)-1), random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig = go.Figure()
    
    for cluster in np.unique(labels):
        mask = labels == cluster
        fig.add_trace(go.Scatter(
            x=X_tsne[mask, 0],
            y=X_tsne[mask, 1],
            mode='markers+text',
            text=ranked_df['Sample'][mask],
            name=f'Cluster {cluster}',
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title="t-SNE Clustering Visualization",
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster statistics
    st.markdown("""
    <div style='background-color: #ffffff; color: #000000; padding: 20px; border-radius: 10px; border: 2px solid #3498db; margin: 15px 0;'>
        <h4 style='color: #2c3e50;'>Cluster Statistics</h4>
    </div>
    """, unsafe_allow_html=True)
    
    for cluster in np.unique(labels):
        cluster_data = ranked_df[labels == cluster]
        st.markdown(f"**Cluster {cluster}** ({len(cluster_data)} samples)")
        st.dataframe(cluster_data.describe().style.background_gradient(cmap='RdBu'))

# Insights and Recommendations
st.header("Key Insights and Recommendations")
st.markdown(f"""
<div style='background-color: #ffffff; color: #000000; padding: 25px; border-radius: 10px; border: 2px solid #3498db; margin: 20px 0;'>
    <h3 style='color: #2c3e50; margin-bottom: 20px;'>Key Findings for {parameter}</h3>
    <ul style='color: #2c3e50; line-height: 1.8;'>
        <li><strong>Mean value:</strong> {ranked_df[parameter].mean():.3f}</li>
        <li><strong>Standard deviation:</strong> {ranked_df[parameter].std():.3f}</li>
        <li><strong>Number of locations exceeding standards:</strong> {sum(ranked_df[parameter] > standards_df.loc[0, parameter] if parameter in standards_df.columns else 0)}</li>
        <li><strong>Coefficient of variation:</strong> {(ranked_df[parameter].std() / ranked_df[parameter].mean() * 100):.1f}%</li>
    </ul>
</div>
""", unsafe_allow_html=True)




# Automatically fetches from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")


# Function to search web (mock or real version)
def search_web(query):
    # Replace with SerpAPI/Bing if needed
    return f"🔍 (Mock result) Here's what I found online about '{query}': Galamsey has significantly affected water bodies in Ghana, especially in the Ashanti and Eastern Regions."

# Function to generate AI response
def generate_response(prompt, data):
    prompt_lower = prompt.lower()
    
    if "average" in prompt_lower or "mean" in prompt_lower:
        if "ph" in prompt_lower and "ph" in data.columns:
            avg_ph = data["pH"].mean()
            return f"The average pH value across all sampled locations is **{avg_ph:.2f}**."
        elif "turbidity" in prompt_lower and "Turbidity" in data.columns:
            avg_turb = data["Turbidity"].mean()
            return f"The average turbidity across all locations is **{avg_turb:.2f} NTU**."
        elif "iron" in prompt_lower and "Iron" in data.columns:
            avg_iron = data["Iron"].mean()
            return f"The average iron concentration is **{avg_iron:.2f} mg/L**."
        else:
            return "I couldn't find that parameter in the dataset. Please check the column name."

    elif "galamsey" in prompt_lower or "illegal mining" in prompt_lower:
        return search_web(prompt)

    else:
        return "🤖 I'm here to help. Please ask about water quality parameters like pH, turbidity, or Galamsey-related issues in Ghana."

# Streamlit UI
st.header("💧 Galamsey & Water Quality Assistant")

st.markdown("""
<div style='background-color: #ffffff; color: #000000; padding: 25px; border-radius: 10px; border: 2px solid #3498db; margin: 20px 0;'>
<p>Welcome! Ask anything about <strong>Galamsey</strong> and its impact on water quality in Ghana. This assistant can:</p>
<ul>
    <li>🔎 Analyze pH, turbidity, and heavy metals in your dataset</li>
    <li>🌍 Provide contextual insights about illegal mining (Galamsey) in Ghana</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = generate_response(prompt, data)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)