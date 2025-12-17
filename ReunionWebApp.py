"""
Bucknell Reunion Uplift Prediction App
A decision support system for optimizing peer-to-peer contact campaigns
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Bucknell Color Scheme
BUCKNELL_ORANGE = '#E87722'
BUCKNELL_BLUE = '#003865'
BUCKNELL_LIGHT_BLUE = '#4A90E2'

# Page Configuration
st.set_page_config(
    page_title="Bucknell Reunion Uplift Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Bucknell branding
st.markdown(f"""
    <style>
    .main {{
        background-color: #F5F5F5;
    }}
    .stButton>button {{
        background-color: {BUCKNELL_ORANGE};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: #D66612;
        color: white;
    }}
    h1, h2, h3 {{
        color: {BUCKNELL_BLUE};
    }}
    .metric-card {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {BUCKNELL_ORANGE};
    }}
    .sidebar .sidebar-content {{
        background-color: {BUCKNELL_BLUE};
    }}
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, {BUCKNELL_BLUE} 0%, {BUCKNELL_LIGHT_BLUE} 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🎓 Bucknell Reunion Uplift Predictor</h1>
        <p style='color: white; font-size: 1.2rem; margin-top: 0.5rem;'>Optimize Your Peer-to-Peer Contact Strategy</p>
    </div>
    """, unsafe_allow_html=True)

# Load the pre-trained model
pickle_filename = 'base_uplift_models_2025_12_12.pkl'
try:
    with open(pickle_filename, 'rb') as f:
        models = pickle.load(f)
    model_noPeer = models['model_noPeer']
    model_AllPeer = models['model_AllPeer']
    required_features = models['predictors']
    metadata = models.get('metadata', {})
    model_loaded = True
except FileNotFoundError:
    st.error(f"❌ Model file '{pickle_filename}' not found. Please ensure it's in the same directory as this app.")
    model_loaded = False
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model_loaded = False

# Sidebar
with st.sidebar:
    st.markdown(f"<h2 style='color: {BUCKNELL_ORANGE};'>📊 Data Upload</h2>", unsafe_allow_html=True)
    
    # Data Upload
    st.markdown("### 1️⃣ Upload Alumni Data")
    data_file = st.file_uploader(
        "Upload alumni data (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with alumni characteristics"
    )
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Display Settings")
    top_n = st.slider(
        "Show top N alumni",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of highest-uplift alumni to display"
    )
    
    show_negative = st.checkbox(
        "Include negative uplift",
        value=False,
        help="Show alumni with predicted negative uplift (peer contact may harm registration)"
    )

# Main Content
if not model_loaded:
    st.error(f"⚠️ Cannot proceed without model. Please ensure '{pickle_filename}' is in the application directory.")
    st.stop()

if data_file is None:
    # Welcome/Instructions Page
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: {BUCKNELL_ORANGE};'>📖 How to Use</h2>
                <ol style='font-size: 1.1rem; line-height: 2rem; color: #333333;'>
                    <li><strong>Upload Data:</strong> Provide alumni data (CSV or Excel)</li>
                    <li><strong>Review Results:</strong> View prioritized contact list</li>
                    <li><strong>Export:</strong> Download recommendations for action</li>
                </ol>
                <p style='font-size: 0.9rem; margin-top: 1rem; color: {BUCKNELL_BLUE};'>
                    ℹ️ Model automatically loaded: {pickle_filename}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class='metric-card' style='margin-top: 1rem;'>
                <h3 style='color: {BUCKNELL_BLUE};'>📋 Required Data Fields</h3>
                <ul style='font-size: 1rem; color: #333333;'>
                    <li>Reunion Years Out</li>
                    <li>Greek_Yes</li>
                    <li>Constituency indicators (Alumni - Undergrad, Donor, etc.)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class='metric-card'>
                <h2 style='color: {BUCKNELL_ORANGE};'>🎯 What is Uplift?</h2>
                <p style='font-size: 1.1rem; line-height: 1.8rem; color: #333333;'>
                    <strong>Uplift</strong> measures how much more likely an alumnus is to register 
                    <em>because</em> of peer-to-peer contact.
                </p>
                <p style='font-size: 1.1rem; line-height: 1.8rem; color: #333333;'>
                    <strong>High Uplift = Priority Contact</strong><br>
                    These alumni benefit most from personal outreach.
                </p>
                <p style='font-size: 1.1rem; line-height: 1.8rem; color: #333333;'>
                    <strong>Low/Negative Uplift = Lower Priority</strong><br>
                    These alumni will likely register anyway or may not respond well to contact.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample visualization
        sample_data = pd.DataFrame({
            'Uplift': np.random.normal(0.15, 0.1, 100)
        })
        fig = px.histogram(
            sample_data, 
            x='Uplift',
            nbins=30,
            title="Example: Uplift Distribution",
            color_discrete_sequence=[BUCKNELL_ORANGE]
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333333', size=12),
            title_font=dict(color='#333333', size=16),
            xaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
            yaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
            legend=dict(font=dict(color='#333333'))
        )
        st.plotly_chart(fig, width='stretch')

else:
    # Display model info
    st.success(f"✅ Model loaded: {pickle_filename}")
    
    with st.expander("📊 Model Information", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trained Date", metadata.get('trained_date', 'N/A'))
        with col2:
            st.metric("Control Model AUC", f"{metadata.get('auc_noPeer_test', 0):.3f}")
        with col3:
            st.metric("Treatment Model AUC", f"{metadata.get('auc_AllPeer_test', 0):.3f}")
        with col4:
            st.metric("Features", len(required_features))
    
    # Load the data
    try:
        if data_file.name.endswith('.csv'):
            input_data = pd.read_csv(data_file)
        else:
            input_data = pd.read_excel(data_file)
        
        st.success(f"✅ Data loaded: {len(input_data):,} alumni records")
        
        # Validate required features
        missing_features = [f for f in required_features if f not in input_data.columns]
        
        if missing_features:
            st.error(f"❌ Missing required features: {', '.join(missing_features)}")
            st.info("💡 Available columns: " + ", ".join(input_data.columns.tolist()))
            st.stop()
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()
    
    # Extract features for prediction
    X_input = input_data[required_features]
    
    # Make predictions
    with st.spinner("🔮 Calculating uplift scores..."):
        prob_no_contact = model_noPeer.predict_proba(X_input)[:, 1]
        prob_with_contact = model_AllPeer.predict_proba(X_input)[:, 1]
        uplift_scores = prob_with_contact - prob_no_contact
    
    # Create results dataframe
    results_df = input_data.copy()
    results_df['Prob_No_Contact'] = prob_no_contact
    results_df['Prob_With_Contact'] = prob_with_contact
    results_df['Uplift_Score'] = uplift_scores
    results_df['Uplift_Percentage'] = (uplift_scores * 100).round(1)
    
    # Filter negative uplift if needed
    if not show_negative:
        results_df = results_df[results_df['Uplift_Score'] > 0]
    
    # Sort by uplift
    results_df = results_df.sort_values('Uplift_Score', ascending=False).reset_index(drop=True)
    results_df['Priority_Rank'] = range(1, len(results_df) + 1)
    
    # Display metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: {BUCKNELL_BLUE}; margin: 0;'>Total Alumni</h3>
                <p style='font-size: 2rem; color: {BUCKNELL_ORANGE}; margin: 0.5rem 0;'>{len(input_data):,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        high_uplift = len(results_df[results_df['Uplift_Score'] > 0.2])
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: {BUCKNELL_BLUE}; margin: 0;'>High Uplift (>20%)</h3>
                <p style='font-size: 2rem; color: {BUCKNELL_ORANGE}; margin: 0.5rem 0;'>{high_uplift:,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        avg_uplift = results_df['Uplift_Percentage'].mean()
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: {BUCKNELL_BLUE}; margin: 0;'>Avg Uplift</h3>
                <p style='font-size: 2rem; color: {BUCKNELL_ORANGE}; margin: 0.5rem 0;'>{avg_uplift:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        negative_uplift = len(input_data) - len(results_df[results_df['Uplift_Score'] > 0])
        st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: {BUCKNELL_BLUE}; margin: 0;'>Negative Uplift</h3>
                <p style='font-size: 2rem; color: #DC3545; margin: 0.5rem 0;'>{negative_uplift:,}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Priority List", "📈 Distribution", "🎯 Insights"])
    
    with tab1:
        st.markdown(f"<h2 style='color: {BUCKNELL_BLUE};'>🎯 Top {min(top_n, len(results_df))} Priority Contacts</h2>", unsafe_allow_html=True)
        
        # Display top N
        display_df = results_df.head(top_n)
        
        # Select key columns for display
        display_cols = ['Priority_Rank', 'Uplift_Percentage', 'Prob_No_Contact', 'Prob_With_Contact']
        
        # Add identifier columns if they exist
        id_cols = [col for col in ['Participant', 'Source ID', 'Name', 'ID'] if col in display_df.columns]
        display_cols = id_cols + display_cols
        
        # Add feature columns
        feature_display_cols = ['Reunion Years Out', 'Greek_Yes', 'Donor', 'Alumni - Undergrad']
        feature_display_cols = [col for col in feature_display_cols if col in display_df.columns]
        display_cols += feature_display_cols
        
        # Format the dataframe
        styled_df = display_df[display_cols].copy()
        styled_df['Prob_No_Contact'] = (styled_df['Prob_No_Contact'] * 100).round(1)
        styled_df['Prob_With_Contact'] = (styled_df['Prob_With_Contact'] * 100).round(1)
        
        # Rename for clarity
        styled_df = styled_df.rename(columns={
            'Prob_No_Contact': 'Reg % (No Contact)',
            'Prob_With_Contact': 'Reg % (With Contact)',
            'Uplift_Percentage': 'Uplift %'
        })
        
        # Color-code uplift
        def highlight_uplift(val):
            if val > 20:
                return f'background-color: {BUCKNELL_ORANGE}; color: white; font-weight: bold;'
            elif val > 10:
                return f'background-color: {BUCKNELL_LIGHT_BLUE}; color: white;'
            else:
                return ''
        
        styled_table = styled_df.style.map(
            highlight_uplift,
            subset=['Uplift %']
        ).format({
            'Uplift %': '{:.1f}%',
            'Reg % (No Contact)': '{:.1f}%',
            'Reg % (With Contact)': '{:.1f}%'
        })
        
        st.dataframe(styled_table, width='stretch', height=600)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results (CSV)",
            data=csv,
            file_name=f"reunion_uplift_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    with tab2:
        st.markdown(f"<h2 style='color: {BUCKNELL_BLUE};'>📈 Uplift Distribution</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                results_df,
                x='Uplift_Percentage',
                nbins=50,
                title="Uplift Score Distribution",
                labels={'Uplift_Percentage': 'Uplift (%)'},
                color_discrete_sequence=[BUCKNELL_ORANGE]
            )
            fig_hist.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#333333', size=12),
                title_font=dict(color='#333333', size=16),
                xaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                yaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                legend=dict(font=dict(color='#333333'))
            )
            fig_hist.add_vline(
                x=results_df['Uplift_Percentage'].mean(),
                line_dash="dash",
                line_color=BUCKNELL_BLUE,
                annotation_text="Mean"
            )
            st.plotly_chart(fig_hist, width='stretch')
        
        with col2:
            # Box plot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=results_df['Uplift_Percentage'],
                name='Uplift',
                marker_color=BUCKNELL_ORANGE,
                boxmean='sd'
            ))
            fig_box.update_layout(
                title="Uplift Statistics",
                yaxis_title="Uplift (%)",
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='#333333', size=12),
                title_font=dict(color='#333333', size=16),
                xaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                yaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                legend=dict(font=dict(color='#333333'))
            )
            st.plotly_chart(fig_box, width='stretch')
        
        # Scatter plot
        fig_scatter = px.scatter(
                results_df.head(500),
                x='Prob_No_Contact',
                y='Prob_With_Contact',
                color='Uplift_Percentage',
                title="Registration Probability: No Contact vs With Contact",
                labels={
                    'Prob_No_Contact': 'Probability (No Contact)',
                    'Prob_With_Contact': 'Probability (With Contact)',
                    'Uplift_Percentage': 'Uplift %'
                },
                color_continuous_scale=['red', 'yellow', 'green'],
                hover_data=['Uplift_Percentage']
            )
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='No Effect Line',
                showlegend=True
            )
        )
        fig_scatter.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#333333', size=12),
            title_font=dict(color='#333333', size=16),
            xaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
            yaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
            legend=dict(font=dict(color='#333333'))
        )
        st.plotly_chart(fig_scatter, width='stretch')
    
    with tab3:
        st.markdown(f"<h2 style='color: {BUCKNELL_BLUE};'>🎯 Strategic Insights</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color: {BUCKNELL_ORANGE};'>💡 Recommendations</h3>
                    <ul style='font-size: 1.1rem; line-height: 2rem; color: #333333;'>
                        <li><strong>Top {min(50, len(results_df))} alumni:</strong> Highest priority for personal calls</li>
                        <li><strong>Next {min(100, len(results_df))}:</strong> Email or phone campaign</li>
                        <li><strong>Negative uplift:</strong> Standard invitation only (no personal contact)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Uplift by feature
            if 'Reunion Years Out' in results_df.columns:
                st.markdown(f"<h4 style='color: {BUCKNELL_BLUE};'>Uplift by Reunion Years</h4>", unsafe_allow_html=True)
                
                # Create bins
                results_df['Years_Bin'] = pd.cut(
                    results_df['Reunion Years Out'],
                    bins=[0, 15, 30, 50, 100],
                    labels=['5-15 yrs', '16-30 yrs', '31-50 yrs', '50+ yrs']
                )
                
                avg_by_years = results_df.groupby('Years_Bin', observed=False)['Uplift_Percentage'].mean().reset_index()
                
                fig_years = px.bar(
                    avg_by_years,
                    x='Years_Bin',
                    y='Uplift_Percentage',
                    title="Average Uplift by Reunion Years",
                    color='Uplift_Percentage',
                    color_continuous_scale=[BUCKNELL_BLUE, BUCKNELL_ORANGE]
                )
                fig_years.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#333333', size=12),
                    title_font=dict(color='#333333', size=16),
                    xaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                    yaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                    showlegend=False,
                    legend=dict(font=dict(color='#333333'))
                )
                st.plotly_chart(fig_years, width='stretch')
        
        with col2:
            # Expected impact
            expected_additional = (results_df['Uplift_Score'].sum())
            
            st.markdown(f"""
                <div class='metric-card'>
                    <h3 style='color: {BUCKNELL_ORANGE};'>📊 Expected Impact</h3>
                    <p style='font-size: 1.1rem; color: #333333;'>
                        If you contact the <strong>top {min(top_n, len(results_df))}</strong> alumni:
                    </p>
                    <p style='font-size: 2.5rem; color: {BUCKNELL_ORANGE}; margin: 1rem 0;'>
                        +{int(results_df.head(top_n)['Uplift_Score'].sum())} registrations
                    </p>
                    <p style='font-size: 1rem; color: {BUCKNELL_BLUE};'>
                        Average uplift: {results_df.head(top_n)['Uplift_Percentage'].mean():.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Uplift by Greek affiliation if available
            if 'Greek_Yes' in results_df.columns:
                st.markdown(f"<h4 style='color: {BUCKNELL_BLUE};'>Uplift by Greek Affiliation</h4>", unsafe_allow_html=True)
                
                avg_by_greek = results_df.groupby('Greek_Yes')['Uplift_Percentage'].mean().reset_index()
                avg_by_greek['Greek_Yes'] = avg_by_greek['Greek_Yes'].map({True: 'Greek', False: 'Non-Greek'})
                
                fig_greek = px.bar(
                    avg_by_greek,
                    x='Greek_Yes',
                    y='Uplift_Percentage',
                    title="Average Uplift by Greek Status",
                    color='Uplift_Percentage',
                    color_continuous_scale=[BUCKNELL_BLUE, BUCKNELL_ORANGE]
                )
                fig_greek.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#333333', size=12),
                    title_font=dict(color='#333333', size=16),
                    xaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                    yaxis=dict(title_font=dict(color='#333333', size=14), tickfont=dict(color='#333333', size=12)),
                    showlegend=False,
                    legend=dict(font=dict(color='#333333'))
                )
                st.plotly_chart(fig_greek, width='stretch')
        
        # Summary statistics
        st.markdown(f"<h3 style='color: {BUCKNELL_BLUE};'>📈 Summary Statistics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        stats = results_df['Uplift_Percentage'].describe()
        
        with col1:
            st.metric("Mean", f"{stats['mean']:.1f}%")
        with col2:
            st.metric("Median", f"{stats['50%']:.1f}%")
        with col3:
            st.metric("Std Dev", f"{stats['std']:.1f}%")
        with col4:
            st.metric("Min", f"{stats['min']:.1f}%")
        with col5:
            st.metric("Max", f"{stats['max']:.1f}%")

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: {BUCKNELL_BLUE}; padding: 1rem;'>
        <p>🎓 <strong>Bucknell University</strong> | Reunion Uplift Decision Support System</p>
        <p style='font-size: 0.9rem;'>Powered by Machine Learning & Causal Inference</p>
    </div>
    """, unsafe_allow_html=True)
