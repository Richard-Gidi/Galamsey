# Water Quality Analysis Dashboard

This interactive dashboard visualizes and analyzes water quality data from various rivers and a Galamsey (illegal mining) pit to assess the environmental impact of illegal mining activities.

## Features

- Interactive visualizations of water quality parameters
- Heavy metals concentration analysis
- pH level monitoring
- Water hardness analysis
- Conductivity and TDS relationship visualization
- Comprehensive findings and recommendations

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

To run the dashboard, execute:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

## Dashboard Components

1. **Heavy Metals Distribution**: Visualizes the concentration of As, Cd, Cr, and Pb across different water bodies
2. **pH Levels**: Shows pH values with TDS correlation
3. **Water Hardness Analysis**: Displays calcium and magnesium hardness levels
4. **Conductivity vs TDS**: Scatter plot showing the relationship between conductivity and total dissolved solids

## Interactivity

- Use the sidebar filters to select specific locations
- Hover over data points for detailed information
- Click and drag to zoom into specific areas of the charts
- Double-click to reset the view

## Environmental Impact Analysis

The dashboard provides comprehensive insights into:
- Heavy metal contamination levels
- Water acidity and its implications
- Water hardness variations
- Recommended solutions for environmental protection

## Data Sources

The data used in this dashboard comes from water quality measurements taken from various rivers and a Galamsey pit, focusing on:
- Heavy metals (As, Cd, Cr, Pb)
- pH levels
- Total Dissolved Solids (TDS)
- Conductivity
- Water hardness parameters 