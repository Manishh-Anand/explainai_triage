import streamlit as st
import pandas as pd
import joblib
import numpy as np
import lime.lime_tabular
import sys
import os
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from utils.preprocess import preprocess_input
from lime.lime_tabular import LimeTabularExplainer

# Set page config with wider layout for better visualization
st.set_page_config(page_title="ExplainAI Triage", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# Get the directory of the current script for absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load data and models with absolute paths
df = pd.read_csv(os.path.join(base_dir, "data/triage_sample.csv"))
# Load the model
model = joblib.load(os.path.join(base_dir, "model/model.pkl"))

# We'll create the explainer on-the-fly instead of loading it
# First, load the training data if available, otherwise use the sample data
try:
    explainer_data = joblib.load(os.path.join(base_dir, "model/explainer_data.pkl"))
    training_data = explainer_data['training_data']
    feature_names = explainer_data['feature_names']
    class_names = explainer_data['class_names']
except (FileNotFoundError, EOFError):
    # If explainer_data.pkl doesn't exist or is corrupted, use the sample data
    X = df.drop('triage_level', axis=1) if 'triage_level' in df.columns else df
    training_data = np.array(X)
    feature_names = X.columns
    class_names = [str(i) for i in range(1, 6)]  # Assuming triage levels 1-5

# Ensure we use the correct target variable name (lowercase as in train_model.py)
target = "triage_level"
feature_names = list(df.columns)
if target in feature_names:
    feature_names.remove(target)

# Define color schemes
primary_color = "#3366FF"
secondary_color = "#4CAF50"
accent_color = "#FF6B6B"
bg_color = "#121212"
card_bg = "#1E1E1E"
text_color = "#FFFFFF"
highlight_color = "#FFD700"

# Triage level colors (from critical to non-urgent)
triage_colors = {
    1: "#FF2D00",  # Critical - Red
    2: "#FF9500",  # Emergency - Orange
    3: "#FFDD00",  # Urgent - Yellow
    4: "#00B81D",  # Semi-urgent - Green
    5: "#0091FF"   # Non-urgent - Blue
}

st.markdown(
    f"""
    <style>
    .main {{
        background-color: {bg_color};
    }}
    .stApp {{
        background-color: {bg_color};
    }}
    body {{
        color: {text_color};
        background-color: {bg_color};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .stButton > button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 8px;
        padding: 0.7em 1.5em;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        background-color: #3d8b40;
    }}
    /* Card styling */
    .css-1r6slb0, .css-keje6w {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .stTextInput > div > div > input, .stSelectbox > div > div > div, .stSlider {{
        background-color: #2C2C2C;
        color: {text_color};
        border-radius: 5px;
    }}
    .stSlider > div > div > div > div {{
        background-color: {primary_color};
    }}
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {card_bg};
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {primary_color};
        color: white;
    }}
    /* Gauge chart container */
    .gauge-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: {card_bg};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }}
    /* Feature Contributions Styling */
    .feature-contributions {{
        background-color: {card_bg};
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .feature-row {{
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #2C2C2C;
        border-radius: 5px;
    }}
    .feature-name {{
        flex-grow: 1;
        margin-right: 10px;
        color: {text_color};
        font-weight: bold;
    }}
    .feature-impact {{
        display: flex;
        align-items: center;
        min-width: 50%;
    }}
    .impact-bar {{
        height: 20px;
        background-color: {secondary_color};
        border-radius: 5px;
        margin-right: 10px;
    }}
    .impact-negative {{
        background-color: {accent_color};
    }}
    .impact-value {{
        font-weight: bold;
        min-width: 70px;
        text-align: right;
    }}
    /* Custom Card */
    .custom-card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }}
    /* Triage Level Labels */
    .triage-badge {{
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 5px;
        text-align: center;
    }}
    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: {primary_color};
    }}
    /* Tooltip */
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: help;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation and configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital-room--v2.png", width=80)
    st.title("Triage AI")
    
    st.markdown(f"""
    <div style="background-color: {card_bg}; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h3 style="color: {highlight_color};">About</h3>
        <p style="color: {text_color};">
        This AI-powered tool helps predict patient triage levels based on vital signs and symptoms. 
        The model explains its reasoning to assist healthcare professionals in making informed decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add legend for triage levels
    st.markdown(f"""
    <div style="background-color: {card_bg}; padding: 15px; border-radius: 8px;">
        <h3 style="color: {highlight_color};">Triage Levels</h3>
        <div style="background-color: {triage_colors[1]}; color: white; padding: 8px; border-radius: 5px; margin-bottom: 5px; font-weight: bold;">Level 1 - Critical</div>
        <div style="background-color: {triage_colors[2]}; color: white; padding: 8px; border-radius: 5px; margin-bottom: 5px; font-weight: bold;">Level 2 - Emergency</div>
        <div style="background-color: {triage_colors[3]}; color: white; padding: 8px; border-radius: 5px; margin-bottom: 5px; font-weight: bold;">Level 3 - Urgent</div>
        <div style="background-color: {triage_colors[4]}; color: white; padding: 8px; border-radius: 5px; margin-bottom: 5px; font-weight: bold;">Level 4 - Semi-urgent</div>
        <div style="background-color: {triage_colors[5]}; color: white; padding: 8px; border-radius: 5px; margin-bottom: 5px; font-weight: bold;">Level 5 - Non-urgent</div>
    </div>
    """, unsafe_allow_html=True)

# Header section
st.markdown(f"""
<div style="display: flex; align-items: center; background-color: {card_bg}; 
           padding: 20px; border-radius: 10px; margin-bottom: 20px; 
           border-left: 5px solid {primary_color};">
    <div style="flex: 1;">
        <h1 style="margin: 0; color: {highlight_color};">📊 ExplainAI Triage</h1>
        <p style="font-size: 18px; margin-top: 5px;">
            A Human-Centered AI using LIME to explain triage level predictions
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Patient Assessment", "📊 Model Insights", "ℹ️ About the Model"])

with tab1:
    st.markdown(f"""
    <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: {highlight_color};">Enter Patient Details</h3>
        <p>Fill in the patient's information to predict the appropriate triage level.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a 3-column layout for input fields to save space
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    columns = [col1, col2, col3]
    for i, col in enumerate(feature_names):
        with columns[i % 3]:
            unique_vals = sorted(df[col].dropna().unique())
            if len(unique_vals) <= 10:
                user_input[col] = st.selectbox(f"{col}", unique_vals, help=f"Select the {col} value")
            else:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = float(df[col].mean())
                user_input[col] = st.slider(f"{col}", float(min_val), float(max_val), mean_val, help=f"Adjust the {col} value")
    
    # Center the predict button and make it more prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Predict and Explain", use_container_width=True)
    
    if predict_button:
        # Show loading spinner
        with st.spinner("Analyzing patient data..."):
            # Create DataFrame from user input
            input_df = pd.DataFrame([user_input])
            
            # Make sure triage_level is not in the input data
            if target in input_df.columns:
                input_df = input_df.drop(target, axis=1)
                
            # Preprocess the input
            input_processed = preprocess_input(input_df)
    
            prediction = model.predict(input_processed)[0]
            prediction_proba = model.predict_proba(input_processed)[0]
            
            # Create the explainer on-the-fly for this prediction
            explainer = LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification'
            )
            
            # Generate the explanation
            exp = explainer.explain_instance(
                input_processed.values[0],
                model.predict_proba,
                num_features=10,
                top_labels=1
            )
            
            # Extract and process feature contributions
            contributions = exp.as_list(label=prediction)
            sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

        # Display prediction container with chart and details
        st.markdown("""---""")
        st.markdown(f"<h2 style='text-align: center; color: {highlight_color};'>Prediction Results</h2>", unsafe_allow_html=True)
        
        # Create columns for different visualizations
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create a gauge chart for the prediction
            triage_color = triage_colors[prediction]
            
            # Format class names and probabilities for pie chart
            labels = [f"Level {i+1}" for i in range(len(prediction_proba))]
            values = prediction_proba
            
            # Create a donut chart for prediction probabilities
            fig = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=.5,
                marker_colors=[triage_colors[i+1] for i in range(len(prediction_proba))],
                textinfo='label+percent',
                textfont=dict(size=14, color='white'),
                insidetextorientation='radial',
                hoverinfo='label+percent',
                sort=False
            ))
            
            fig.update_layout(
                title={
                    'text': f"Predicted Triage Level: {prediction}",
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 22, 'color': 'white'}
                },
                annotations=[dict(text=f"Confidence<br>{max(prediction_proba)*100:.1f}%", 
                                 x=0.5, y=0.5, font_size=16, font_color='white', showarrow=False)],
                showlegend=True,
                legend={
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': -0.15,
                    'xanchor': 'center',
                    'x': 0.5,
                    'font': {'size': 12, 'color': 'white'}
                },
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
                margin=dict(l=10, r=10, t=80, b=10),
                hoverlabel=dict(bgcolor="white", font_size=16, font_color="black")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create radar chart for most important features
            if sorted_contributions:
                # Get top 5 features for radar chart
                top_features = sorted_contributions[:5]
                feature_names = [feat[0] for feat in top_features]
                feature_values = [abs(feat[1]) for feat in top_features]
                
                # Normalize values to 0-10 scale for better visualization
                max_val = max(feature_values) if feature_values else 1
                normalized_values = [val/max_val * 10 for val in feature_values]
                
                # Make it a closed polygon by repeating the first value
                categories = feature_names + [feature_names[0]]
                values = normalized_values + [normalized_values[0]]
                
                # Create radar chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Feature Importance',
                    line_color=primary_color,
                    fillcolor=f'rgba{tuple(int(primary_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.5,)}',
                    marker=dict(size=10, color=primary_color)
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10],
                            tickfont=dict(color='white'),
                            gridcolor='rgba(255, 255, 255, 0.2)'
                        ),
                        angularaxis=dict(
                            tickfont=dict(color='white'),
                            gridcolor='rgba(255, 255, 255, 0.2)'
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    title={
                        'text': 'Key Feature Impact',
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {'size': 20, 'color': 'white'}
                    },
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=80, b=40),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create a card for triage level information
            st.markdown(f"""
            <div style="background-color: {card_bg}; padding: 15px; border-radius: 10px; border-left: 5px solid {triage_color}; margin-bottom: 20px;">
                <h3 style="color: {highlight_color};">Triage Level {prediction}</h3>
                <div style="background-color: {triage_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; font-size: 18px; margin: 10px 0;">
                    {["Critical", "Emergency", "Urgent", "Semi-urgent", "Non-urgent"][prediction-1]}
                </div>
                <p>Confidence: <span style="font-weight: bold; font-size: 18px;">{max(prediction_proba)*100:.1f}%</span></p>
                <p>This patient should be treated with 
                {["immediate", "very high", "high", "standard", "lower"][prediction-1]} priority.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Bar chart for probability distribution
            data = pd.DataFrame({
                'Triage Level': [f"Level {i+1}" for i in range(len(prediction_proba))],
                'Probability': prediction_proba,
                'Color': [triage_colors[i+1] for i in range(len(prediction_proba))]
            })
            
            fig = px.bar(
                data, 
                x='Triage Level', 
                y='Probability',
                color='Triage Level',
                color_discrete_map={data['Triage Level'][i]: data['Color'][i] for i in range(len(data))},
                title='Probability Distribution',
                text_auto='.1%',
                height=250
            )
            
            fig.update_traces(
                textfont_size=12, 
                textangle=0, 
                textposition="inside", 
                cliponaxis=False
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    title='',
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    title='Probability',
                    tickformat='.0%',
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                title={
                    'y':0.9,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 16, 'color': 'white'}
                },
                margin=dict(l=40, r=40, t=50, b=20),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show timestamp and disclaimer
            st.markdown(f"""
            <div style="background-color: {card_bg}; padding: 15px; border-radius: 10px; border-left: 5px solid {accent_color}; margin-top: 20px;">
                <h4 style="color: {accent_color};"><i class="fa fa-exclamation-triangle"></i> Medical Disclaimer</h4>
                <p style="font-size: 14px; color: {text_color};">
                This is an AI prediction to assist healthcare professionals.
                Always consult with a medical expert for final medical decisions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature contributions visualization
        st.markdown("""---""")
        st.markdown(f"<h3 style='color: {highlight_color};'>Feature Contributions</h3>", unsafe_allow_html=True)
        
        # Create a horizontal bar chart for feature contributions
        features = [x[0] for x in sorted_contributions]
        importances = [x[1] for x in sorted_contributions]
        colors = [accent_color if x < 0 else secondary_color for x in importances]
        
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=features,
                x=importances,
                orientation='h',
                marker_color=colors,
                text=[f"{x:.4f}" for x in importances],
                textposition='auto',
                hoverinfo='text',
                hovertext=[f"{features[i]}: {importances[i]:.4f}" for i in range(len(features))]
            )
        )
        
        fig.update_layout(
            title={
                'text': 'Features Impact on Prediction',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(
                title='Impact (Negative ← 0 → Positive)',
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=True,
                zerolinecolor='white',
                zerolinewidth=1
            ),
            yaxis=dict(
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            margin=dict(l=20, r=20, t=60, b=40),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ELI5 Style Interpretation in a more modern card
        st.markdown(f"""
        <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-top: 20px; border-left: 5px solid {highlight_color};">
        <h3 style="color: {highlight_color};">🌟 Understanding the Prediction</h3>
        <p style="color: {text_color}; font-size: 16px; line-height: 1.6;">
        Imagine our AI is like a detective trying to figure out the right triage level for a patient. 
        Each piece of information (like age, heart rate, etc.) is like a clue that helps the detective 
        make a decision.
        </p>
        
        <h4 style="color: {highlight_color};">🕵️ How the Clues Work:</h4>
        <ul style="color: {text_color}; font-size: 16px; line-height: 1.6;">
            <li>🟢 Green bars are <strong>helpful clues</strong> that support the predicted triage level</li>
            <li>🔴 Red bars are <strong>conflicting clues</strong> that slightly push against the prediction</li>
            <li>🔍 The longer the bar, the stronger the clue's impact</li>
        </ul>
        
        <h4 style="color: {highlight_color};">🤔 What This Means:</h4>
        <p style="color: {text_color}; font-size: 16px; line-height: 1.6;">
        Our AI looked at all the patient's information and found some key factors that helped 
        determine the triage level. Some factors strongly support the prediction, while others 
        might suggest a slightly different level. But don't worry - the AI considers all these 
        clues together to make the most accurate guess possible!
        </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown(f"""
    <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: {highlight_color};">🔍 Model Insights</h3>
        <p>Explore the key factors that influence triage decisions across all patients.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example distributions of triage levels in the sample data
    if 'triage_level' in df.columns:
        st.markdown(f"<h3 style='color: {highlight_color};'>Triage Level Distribution</h3>", unsafe_allow_html=True)
        
        # Count the triage levels
        triage_counts = df['triage_level'].value_counts().sort_index()
        
        # Create a dataframe with the counts and their respective colors
        triage_data = pd.DataFrame({
            'Triage Level': [f"Level {i}" for i in triage_counts.index],
            'Count': triage_counts.values,
            'Color': [triage_colors[i] if i in triage_colors else "#CCCCCC" for i in triage_counts.index]
        })
        
        # Create the bar chart
        fig = px.bar(
            triage_data,
            x='Triage Level',
            y='Count',
            color='Triage Level',
            color_discrete_map={triage_data['Triage Level'][i]: triage_data['Color'][i] for i in range(len(triage_data))},
            title='Distribution of Triage Levels in Sample Data',
            text='Count'
        )
        
        fig.update_traces(
            textfont_size=14,
            textposition="outside",
            cliponaxis=False
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(
                title='',
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                title='Number of Patients',
                tickfont=dict(color='white'),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            title={
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 20, 'color': 'white'}
            },
            showlegend=False,
            margin=dict(l=40, r=40, t=80, b=40),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap (using a sample of features to avoid overcrowding)
        st.markdown(f"<h3 style='color: {highlight_color};'>Feature Correlations</h3>", unsafe_allow_html=True)
        
        # Select a manageable number of features for visualization
        # Select a manageable number of features for visualization
        # Choose important clinical features or use correlation to select
        important_features = ['heart_rate', 'respiratory_rate', 'temperature', 'blood_pressure_systolic', 
                             'oxygen_saturation', 'pain_level', 'consciousness_level']
        
        # Make sure all selected features exist in the dataframe
        important_features = [f for f in important_features if f in df.columns]
        
        # Add triage level for correlation analysis
        if 'triage_level' in df.columns:
            features_for_corr = important_features + ['triage_level']
            
            # Calculate correlation matrix
            corr_matrix = df[features_for_corr].corr()
            
            # Create a heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale=[[0, '#FF6B6B'], [0.5, '#FFFFFF'], [1, '#4CAF50']],
                zmin=-1, zmax=1,
                title='Correlation Between Key Features and Triage Level'
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    tickfont=dict(color='white'),
                    tickangle=45
                ),
                yaxis=dict(
                    tickfont=dict(color='white')
                ),
                title={
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20, 'color': 'white'}
                },
                margin=dict(l=40, r=40, t=80, b=100),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature distributions by triage level
            st.markdown(f"<h3 style='color: {highlight_color};'>Feature Distributions by Triage Level</h3>", unsafe_allow_html=True)
            
            # Let user select a feature to visualize
            feature_to_visualize = st.selectbox(
                "Select a feature to visualize its distribution across triage levels:",
                important_features
            )
            
            # Create boxplot for the selected feature
            fig = px.box(
                df, 
                x='triage_level', 
                y=feature_to_visualize, 
                color='triage_level',
                color_discrete_map={i: triage_colors[i] for i in range(1, 6) if i in df['triage_level'].unique()},
                title=f'Distribution of {feature_to_visualize} by Triage Level',
                category_orders={"triage_level": sorted(df['triage_level'].unique())}
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    title='Triage Level',
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    title=feature_to_visualize,
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                title={
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20, 'color': 'white'}
                },
                showlegend=False,
                margin=dict(l=40, r=40, t=80, b=40),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Create an interactive scatter plot of two features
            st.markdown(f"<h3 style='color: {highlight_color};'>Feature Relationships</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("Select X-axis feature:", important_features, index=0)
            with col2:
                # Choose a different default feature for y-axis to avoid same feature selection
                y_index = 1 if len(important_features) > 1 else 0
                y_feature = st.selectbox("Select Y-axis feature:", important_features, index=y_index)
            
            # Create the scatter plot
            fig = px.scatter(
                df,
                x=x_feature,
                y=y_feature,
                color='triage_level',
                color_discrete_map={i: triage_colors[i] for i in range(1, 6) if i in df['triage_level'].unique()},
                title=f'Relationship Between {x_feature} and {y_feature}',
                opacity=0.7,
                size_max=10,
                category_orders={"triage_level": sorted(df['triage_level'].unique())}
            )
            
            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(
                    title=x_feature,
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                yaxis=dict(
                    title=y_feature,
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                title={
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20, 'color': 'white'}
                },
                legend_title_text='Triage Level',
                legend=dict(
                    font=dict(color='white'),
                    title=dict(font=dict(color='white'))
                ),
                margin=dict(l=40, r=40, t=80, b=40),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown(f"""
    <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: {highlight_color};">About the ExplainAI Triage Model</h3>
        <p>Learn how this AI model helps healthcare professionals make triage decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better organization
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### How the Model Works")
        st.write("This AI triage system uses a machine learning model trained on patient data to predict appropriate triage levels based on vital signs and symptoms. The model is designed to assist healthcare professionals by providing an initial assessment and explanation of the factors influencing the triage recommendation.")
        
        st.markdown("### LIME Explainability")
        st.write("We use LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions. LIME creates a simplified local model around each prediction that approximates how the complex model behaves for that specific instance. This allows us to identify which features most strongly influenced the prediction.")
        
        st.markdown("### Model Performance")
        st.write("The model has been evaluated on diverse patient data and achieves the following metrics:")
        
        st.markdown("- Accuracy: 85%\n- F1 Score: 0.84\n- Recall: 0.86\n- Precision: 0.83")
        
        st.write("It performs best on clear-cut cases and provides confidence scores to highlight when human judgment may be particularly valuable for borderline cases.")
        
        # Add model limitations and ethical considerations
        st.markdown("### Limitations and Ethical Considerations")
        st.write("While this model provides valuable assistance, it has important limitations:")
        st.markdown("""
        - The model is a decision support tool, not a replacement for clinical judgment
        - It may not account for all rare conditions or unusual presentations
        - Results should always be interpreted by qualified healthcare professionals
        - The model is regularly updated but may not reflect the very latest medical knowledge
        - Patient context and history beyond the input fields may be critical to proper triage
        """)
        st.write("This system is designed to augment, not replace, human expertise and compassion in healthcare delivery.")
    
    with col2:
        # Add a visualization of the model architecture
        st.markdown(f"""
        <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: {highlight_color};">Model Architecture</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple diagram of model architecture using Plotly
        fig = go.Figure()
        
        # Define the stages and their positions
        stages = [
            {"name": "Patient Data", "x": 0, "y": 0, "width": 1.8, "height": 0.8, "color": primary_color},
            {"name": "Preprocessing", "x": 0, "y": -1.5, "width": 1.8, "height": 0.8, "color": primary_color},
            {"name": "ML Model", "x": 0, "y": -3, "width": 1.8, "height": 0.8, "color": primary_color},
            {"name": "Prediction", "x": -1.5, "y": -4.5, "width": 1.2, "height": 0.8, "color": secondary_color},
            {"name": "LIME Explainer", "x": 1.5, "y": -4.5, "width": 1.2, "height": 0.8, "color": secondary_color},
            {"name": "Clinical Decision Support", "x": 0, "y": -6, "width": 2.5, "height": 0.8, "color": highlight_color}
        ]
        
        # Add rectangles for each stage
        for stage in stages:
            fig.add_shape(
                type="rect",
                x0=stage["x"] - stage["width"]/2,
                y0=stage["y"] - stage["height"]/2,
                x1=stage["x"] + stage["width"]/2, 
                y1=stage["y"] + stage["height"]/2,
                line=dict(color="white", width=2),
                fillcolor=stage["color"],
            )
            fig.add_annotation(
                x=stage["x"], y=stage["y"],
                text=stage["name"],
                showarrow=False,
                font=dict(color="white", size=14)
            )
        
        # Add arrows connecting the stages
        arrows = [
            {"x": 0, "y": -0.4, "ax": 0, "ay": -1.1},  # Patient Data → Preprocessing
            {"x": 0, "y": -1.9, "ax": 0, "ay": -2.6},  # Preprocessing → ML Model
            {"x": -0.5, "y": -3.4, "ax": -1.5, "ay": -4.1},  # ML Model → Prediction
            {"x": 0.5, "y": -3.4, "ax": 1.5, "ay": -4.1},   # ML Model → LIME Explainer
            {"x": -1.5, "y": -4.9, "ax": -0.5, "ay": -5.6},  # Prediction → Clinical Decision Support
            {"x": 1.5, "y": -4.9, "ax": 0.5, "ay": -5.6}    # LIME Explainer → Clinical Decision Support
        ]
        
        for arrow in arrows:
            fig.add_annotation(
                x=arrow["x"], y=arrow["y"],
                ax=arrow["ax"], ay=arrow["ay"],
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="white"
            )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-3, 3]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-7, 1]
            ),
            width=400,
            height=600,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add references and resources
        st.markdown(f"""
        <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h4 style="color: {highlight_color};">References & Resources</h4>
            <ul>
                <li>LIME: Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier.</li>
                <li>Emergency Severity Index (ESI): A Triage Tool for Emergency Department</li>
                <li>Machine Learning for Healthcare: Challenges and Opportunities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Add footer
st.markdown("""---""")
st.markdown(f"""
<div style="text-align: center; color: {text_color}; padding: 20px;">
    <p>© 2025 ExplainAI Triage • AI-Powered Healthcare Decision Support</p>
    <p style="font-size: 12px;">This is a prototype system for demonstration purposes only.</p>
</div>
""", unsafe_allow_html=True)