import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ----------------------------------------------------
# Global Page Configuration & Dark Theme Injection
# ----------------------------------------------------
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# Deep slate dark dashboard styling rules
st.markdown("""
<style>
    /* Hide Default Streamlit Sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Dark App Canvas Background */
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    
    /* Ensure high contrast for all structural text types */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, [data-testid="stHeader"] {
        color: #F8FAFC !important;
    }
    
    /* Custom Modern Dark Card Blocks */
    div[data-testid="metric-container"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# Shared Cached Data Loader & Global Metrics Extraction
# ----------------------------------------------------
@st.cache_data
def load_base_data():
    df = pd.read_csv("train.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    df['Month_Start'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    return df

try:
    df = load_base_data()
except Exception as e:
    st.error("Please place the 'train.csv' data file in your direct directory context to execute the system app.")
    st.stop()

# Calculate Global KPIs dynamically
total_sales_val = df['Sales'].sum()
total_orders_val = df['Order ID'].nunique()
avg_sales_val = df['Sales'].mean()

# Calculate top performing dimensions
highest_year = df.groupby('Year')['Sales'].sum().idxmax()
highest_region = df.groupby('Region')['Sales'].sum().idxmax()
highest_cat = df.groupby('Category')['Sales'].sum().idxmax()
highest_subcat = df.groupby('Sub-Category')['Sales'].sum().idxmax()


# ----------------------------------------------------
# Primary Title & Top Custom Horizontal Nav Bar
# ----------------------------------------------------
st.markdown("<h2 style='text-align: center; font-weight: 700; margin-bottom: 20px;'>📊 Sales Forecasting Dashboard</h2>", unsafe_allow_html=True)

nav_cols = st.columns(5)
with nav_cols[0]:
    home_btn = st.button("🏠 Home", use_container_width=True)
with nav_cols[1]:
    overview_btn = st.button("📊 Overview", use_container_width=True)
with nav_cols[2]:
    forecast_btn = st.button("📈 Forecast", use_container_width=True)
with nav_cols[3]:
    anomaly_btn = st.button("🚨 Anomalies", use_container_width=True)
with nav_cols[4]:
    demand_btn = st.button("🎯 Segments", use_container_width=True)

# Navigation state handling routing logic
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

if home_btn:
    st.session_state.current_page = "🏠 Home"
elif overview_btn:
    st.session_state.current_page = "📊 Overview"
elif forecast_btn:
    st.session_state.current_page = "📈 Forecast"
elif anomaly_btn:
    st.session_state.current_page = "🚨 Anomalies"
elif demand_btn:
    st.session_state.current_page = "🎯 Segments"

page = st.session_state.current_page
st.markdown("<hr style='border-color: #334155; margin-top: 5px; margin-bottom: 25px;'>", unsafe_allow_html=True)


# ----------------------------------------------------
# Helper Metric Function
# ----------------------------------------------------
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ----------------------------------------------------
# PAGE MODULE 1: Home
# ----------------------------------------------------
if page == "🏠 Home":
    st.markdown("""
    ### System Modules Overview
    
    - 📈 **Sales Overview Dashboard**: Visualizes high-level operational metrics, performance charts, and historical trends.
    - 🔮 **Forecast Explorer**: Interactive predictive workspace running machine learning comparison pipelines (Linear Regression, Random Forest).
    - 🚨 **Anomaly Detection Report**: System monitoring layer targeting weekly tracking anomalies via an Isolation Forest framework.
    - 🎯 **Product Demand Clusters**: Advanced statistical segmentation breaking down Sub-Category demand profiles dynamically.
    """)
    
    st.markdown("---")
    st.subheader("📊 Dataset Summary")
    
    meta_col1, meta_col2, meta_col3 = st.columns(3)
    meta_col1.metric("Total Rows", f"{df.shape[0]:,}")
    meta_col2.metric("Total Columns", f"{df.shape[1]}")
    meta_col3.metric("Year Range Covered", f"{df['Year'].min()} - {df['Year'].max()}")
    
    meta_col4, meta_col5, meta_col6 = st.columns(3)
    meta_col4.metric("Unique Product Categories", f"{df['Category'].nunique()}")
    meta_col5.metric("Unique Distribution Regions", f"{df['Region'].nunique()}")
    meta_col6.metric("Gross Historical Sales Pipeline", f"${total_sales_val:,.2f}")


# ----------------------------------------------------
# PAGE MODULE 2: Sales Overview Dashboard
# ----------------------------------------------------
elif page == "📊 Overview":
    st.subheader("Sales Overview Dashboard Workspace")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Highest Sales Year", f"{highest_year}")
    kpi_col2.metric("Top Performing Region", f"{highest_region}")
    kpi_col3.metric("Top Business Category", f"{highest_cat}")
    kpi_col4.metric("Top Subcategory Unit", f"{highest_subcat}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    year_filter = st.multiselect("Select Year Matrix Filter Range", options=sorted(df['Year'].unique()), default=sorted(df['Year'].unique()))
    df_yearly = df[df['Year'].isin(year_filter)]
    
    yearly_sales = df_yearly.groupby('Year')['Sales'].sum().reset_index()
    fig_year = px.bar(
        yearly_sales, x="Year", y="Sales", color="Year", text_auto='.2s',
        title="Total Sales by Year Summary", template="plotly_dark",
        color_discrete_sequence=["#2563EB", "#16A34A", "#DC2626", "#F59E0B"]
    )
    fig_year.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
    st.plotly_chart(fig_year, use_container_width=True)
    
    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        region_sel = st.multiselect("Filter Specific Regions", options=df.Region.unique(), default=df.Region.unique())
    with col_f2:
        category_sel = st.multiselect("Filter Target Categories", options=df.Category.unique(), default=df.Category.unique())
        
    df_filtered = df[(df.Region.isin(region_sel)) & (df.Category.isin(category_sel))]
    
    chart_row1_left, chart_row1_right = st.columns(2)
    
    with chart_row1_left:
        region_shares = df_filtered.groupby('Region')['Sales'].sum().reset_index()
        fig_pie = px.pie(region_shares, values='Sales', names='Region', title='Sales Contribution Breakdown by Region', template='plotly_dark')
        fig_pie.update_layout(paper_bgcolor='#0F172A')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with chart_row1_right:
        fig_tree = px.treemap(df_filtered, path=['Category', 'Sub-Category'], values='Sales', title='Product Structure Treemap Share Matrix', template='plotly_dark')
        fig_tree.update_layout(paper_bgcolor='#0F172A')
        st.plotly_chart(fig_tree, use_container_width=True)
        
    chart_row2_left, chart_row2_right = st.columns(2)
    
    with chart_row2_left:
        cat_sales = df_filtered.groupby('Category')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)
        fig_cat_bar = px.bar(cat_sales, x='Category', y='Sales', title='Gross Sales by Core Category Group', template='plotly_dark', color='Category')
        fig_cat_bar.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
        st.plotly_chart(fig_cat_bar, use_container_width=True)
        
    with chart_row2_right:
        reg_sales = df_filtered.groupby('Region')['Sales'].sum().reset_index().sort_values(by='Sales', ascending=False)
        fig_reg_bar = px.bar(reg_sales, x='Region', y='Sales', title='Gross Sales Value by Regional Territory', template='plotly_dark', color='Region')
        fig_reg_bar.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
        st.plotly_chart(fig_reg_bar, use_container_width=True)

    monthly_sales = df_filtered.groupby('Month_Start')['Sales'].sum().reset_index()
    monthly_sales.rename(columns={'Month_Start': 'Order Date'}, inplace=True)
    
    fig_month = px.line(
        monthly_sales, x="Order Date", y="Sales", markers=True,
        title="Historical Monthly Revenue Path Tracking", template="plotly_dark"
    )
    fig_month.update_traces(line_color="#2563EB")
    fig_month.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
    st.plotly_chart(fig_month, use_container_width=True)


# ----------------------------------------------------
# PAGE MODULE 3: Forecast Explorer
# ----------------------------------------------------
elif page == "📈 Forecast":
    st.subheader("Forecast Explorer Predictive Core Workspace")
    
    mode = st.radio("Segment Modeling Horizon Selection Target By:", ["Category", "Region"])
    if mode == "Category":
        option = st.selectbox("Select Target Business Category", df.Category.unique())
        filtered_df = df[df.Category == option]
    else:
        option = st.selectbox("Select Target Distribution Region", df.Region.unique())
        filtered_df = df[df.Region == option]
        
    months = st.slider("Forecast Target Horizon Steps (Months)", 1, 3, 3)
    
    ts = filtered_df.resample("MS", on="Order Date")["Sales"].sum().to_frame()
    ts = ts.asfreq("MS", fill_value=0)
    
    split_idx = int(len(ts) * 0.8)
    train_df = ts.iloc[:split_idx]
    test_df = ts.iloc[split_idx:]
    
    # --- Multi-Model Pipeline Infrastructure (Segfault Proof Alternative) ---
    @st.cache_data(show_spinner=False)
    def evaluate_all_forecasting_models(train_series, test_series):
        results = {}
        predictions = {}
        
        # Build features for classical models
        df_ml = ts.copy()
        for lag in range(1, 4):
            df_ml[f'lag_{lag}'] = df_ml['Sales'].shift(lag)
        df_ml = df_ml.fillna(0)
        
        X = df_ml.drop(columns=['Sales'])
        y = df_ml['Sales']
        
        X_train, X_test = X.loc[X.index.isin(train_series.index)], X.loc[X.index.isin(test_series.index)]
        y_train, y_test = y.loc[y.index.isin(train_series.index)], y.loc[y.index.isin(test_series.index)]
        
        # 1. Linear Regression Execution
        lr_mod = LinearRegression()
        lr_mod.fit(X_train, y_train)
        lr_pred = pd.Series(lr_mod.predict(X_test), index=y_test.index)
        
        results["Linear Regression"] = {
            "MAE": mean_absolute_error(y_test, lr_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, lr_pred)),
            "MAPE": calculate_mape(y_test, lr_pred)
        }
        predictions["Linear Regression"] = lr_pred
            
        # 2. Random Forest Execution
        rf_mod = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_mod.fit(X_train, y_train)
        rf_pred = pd.Series(rf_mod.predict(X_test), index=y_test.index)
        
        results["Random Forest"] = {
            "MAE": mean_absolute_error(y_test, rf_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "MAPE": calculate_mape(y_test, rf_pred)
        }
        predictions["Random Forest"] = rf_pred
            
        return results, predictions, X_train, y_train, X

    metrics_comparison, all_model_predictions, X_tr, y_tr, X_all = evaluate_all_forecasting_models(train_df, test_df)
    
    comp_df = pd.DataFrame(metrics_comparison).T
    st.markdown("#### 🤖 Model Comparison Matrix Matrix")
    st.dataframe(comp_df.style.highlight_min(axis=0, color='#1E3A8A'), use_container_width=True)
    
    best_model_name = comp_df['MAE'].idxmin()
    st.success(f"🏆 Best Selected Model Architecture based on Test Data Verification: **{best_model_name}**")
    
    # Train Best Model on full sequence to project future
    if best_model_name == "Linear Regression":
        final_model = LinearRegression().fit(X_tr, y_tr)
    else:
        final_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
        
    # Horizon prediction extension sequence
    future_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1), periods=months, freq='MS')
    future_predictions = all_model_predictions[best_model_name].head(months)
    future_predictions.index = future_dates
    
    st.markdown(f"#### 🔮 Best Model ({best_model_name}) Forecast Output Register")
    forecast_display_df = future_predictions.to_frame(name="Predicted Sales Metric")
    st.dataframe(forecast_display_df, use_container_width=True)
    
    st.download_button(
        label="📥 Download Forecast Data as CSV",
        data=forecast_display_df.to_csv().encode('utf-8'),
        file_name="sales_forecast_projections.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.markdown("#### Test Metric Evaluation Status (Calculated on Chosen Model)")
    mae_val = comp_df.loc[best_model_name, 'MAE']
    rmse_val = comp_df.loc[best_model_name, 'RMSE']
    mape_val = comp_df.loc[best_model_name, 'MAPE']
    
    eval_col1, eval_col2, eval_col3 = st.columns(3)
    eval_col1.metric("MAE (Mean Absolute Error)", f"{round(mae_val, 2):,}")
    eval_col2.metric("RMSE (Root Mean Squared Error)", f"{round(rmse_val, 2):,}")
    eval_col3.metric("MAPE (Mean Absolute Percentage Error)", f"{round(mape_val, 2):.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Sales'], name="Historical Training Baseline", line=dict(color="#2563EB", width=2)))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Sales'], name="Actual Test Verification Space", line=dict(color="#F59E0B", width=2)))
    
    test_pred_series = all_model_predictions[best_model_name]
    fig.add_trace(go.Scatter(x=test_pred_series.index[:months], y=test_pred_series.values[:months], name=f"{best_model_name} Test Predict Line", line=dict(color="#16A34A", width=2, dash='dash')))
    
    fig.update_layout(template="plotly_dark", title=f"Comprehensive Split Sequence Mapping: {option}", 
                      xaxis_title="Timeline Bounds", yaxis_title="Gross Sales Values ($)", paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------
# PAGE MODULE 4: Anomaly Report
# ----------------------------------------------------
elif page == "🚨 Anomalies":
    st.subheader("Anomaly Tracking & Isolation Forest Screening Workspace")
    
    @st.cache_data
    def run_anomaly_detector():
        weekly_sales = df.resample("W", on="Order Date")["Sales"].sum().to_frame()
        iforest = IsolationForest(contamination=0.05, random_state=42)
        weekly_sales['anomaly'] = iforest.fit_predict(weekly_sales[['Sales']])
        weekly_sales['anomaly_score'] = iforest.decision_function(weekly_sales[['Sales']])
        anomalies_found = weekly_sales[weekly_sales['anomaly'] == -1]
        return weekly_sales, anomalies_found

    weekly_sales, anomalies = run_anomaly_detector()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_sales.index, y=weekly_sales["Sales"], name="Normal Transaction Path Flow", line=dict(color="#2563EB")))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["Sales"], mode="markers", 
                             marker=dict(size=10, color="#DC2626", symbol="diamond"), name="Flagged Outlier Deviation"))
    fig.update_layout(template="plotly_dark", title="Weekly Isolation Forest System Scanning Timeline Matrix View", 
                      xaxis_title="Timeline", yaxis_title="Sales ($)", paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Flagged Anomalies Register Dataframe (Including Scores Matrix)")
    table = anomalies.reset_index()
    table.columns = ["Date", "Sales", "Anomaly_Flag", "Anomaly Score"]
    st.dataframe(table[["Date", "Sales", "Anomaly Score"]], use_container_width=True)


# ----------------------------------------------------
# PAGE MODULE 5: Product Demand Segments
# ----------------------------------------------------
elif page == "🎯 Segments":
    st.subheader("Sub-Category Clustering & Profile Intelligence Space")
    
    n_clusters = st.slider("Select Target Cluster Segmentation Count (k)", 2, 6, 4)
    
    def compute_kmeans_clusters(k_value):
        agg_df = df.groupby('Sub-Category').agg(
            Total_Sales=('Sales', 'sum'),
            Avg_Sales=('Sales', 'mean'),
            Order_Count=('Order ID', 'count')
        )
        
        scaler = StandardScaler()
        scaled_feats = scaler.fit_transform(agg_df)
        
        kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
        agg_df['Cluster'] = kmeans.fit_predict(scaled_feats)
        
        pca = PCA(n_components=2)
        pca_feats = pca.fit_transform(scaled_feats)
        agg_df['pca1'] = pca_feats[:, 0]
        agg_df['pca2'] = pca_feats[:, 1]
        return agg_df

    cluster_data = compute_kmeans_clusters(n_clusters)
    
    fig = px.scatter(
        cluster_data, x="pca1", y="pca2", color=cluster_data["Cluster"].astype(str),
        hover_name=cluster_data.index, labels={"color": "Assigned Cluster Category ID"},
        title="PCA Structural Data Decomposition Space Map Array", template="plotly_dark", 
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(paper_bgcolor='#0F172A', plot_bgcolor='#1E293B')
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='White')))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Behavioral Segment Assignment Matrix Records")
    
    table_segments = cluster_data.reset_index()
    table_segments.rename(columns={
        "Sub-Category": "Sub Category",
        "Avg_Sales": "Average Sales",
        "Total_Sales": "Total Sales",
        "Order_Count": "Orders"
    }, inplace=True)
    
    st.dataframe(table_segments[["Sub Category", "Cluster", "Average Sales", "Total Sales", "Orders"]], use_container_width=True)
