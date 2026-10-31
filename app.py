
import os
import sys
import base64
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Car Classification | Vehicle Intelligence",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent
DATA = BASE / "dashboard_data"
MODEL_PATH = BASE / "models" / "car-classification-pipeline"
BG_PATH = BASE / "assets" / "car_dashboard_bg.png"
HISTORY_PATH = BASE / "dashboard_data" / "prediction_history.csv"

# =====================================================
# 2. BACKGROUND AND DESIGN
# =====================================================

if BG_PATH.exists():
    encoded = base64.b64encode(BG_PATH.read_bytes()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(
                rgba(4, 9, 19, 0.76),
                rgba(4, 9, 19, 0.88)
            ),
            url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(12, 19, 32, 0.96);
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3 {{
        color: #ffffff !important;
    }}

    .main-title {{
        font-size: 38px;
        font-weight: 850;
        letter-spacing: 2px;
        color: white;
    }}

    .subtitle {{
        color: #b8c9dd;
        letter-spacing: 4px;
        font-size: 14px;
    }}

    .accent {{
        color: #ffad42;
    }}

    .glass {{
        background: rgba(15, 27, 45, 0.90);
        border: 1px solid rgba(255, 173, 66, 0.35);
        border-radius: 15px;
        padding: 22px;
        margin: 10px 0;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(15, 27, 45, 0.90);
        border: 1px solid rgba(255, 173, 66, 0.35);
        padding: 18px;
        border-radius: 12px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: #bdcce0;
    }}

    div[data-testid="stMetricValue"] {{
        color: #ffb347;
    }}

    div.stButton > button {{
        background: #ffad42;
        color: #101725;
        border: none;
        border-radius: 9px;
        font-weight: bold;
        min-height: 45px;
    }}

    div.stButton > button:hover {{
        background: #ffc36e;
        color: #101725;
    }}
    </style>
    """, unsafe_allow_html=True)

else:
    st.warning("Background image not found in assets folder.")

# =====================================================
# 3. LOAD DASHBOARD DATA
# =====================================================

@st.cache_data
def load_dashboard_data():

    metrics_df = pd.read_csv(DATA / "metrics.csv")
    labels = pd.read_csv(DATA / "label_mapping.csv")
    features = pd.read_csv(DATA / "feature_importance.csv")
    classes = pd.read_csv(DATA / "class_metrics.csv")
    confusion = pd.read_csv(DATA / "confusion_matrix.csv")

    metrics = dict(
        zip(metrics_df["metric"], metrics_df["value"])
    )

    original_data = pd.read_csv(
        BASE / "data" / "data.csv",
        low_memory=False
    )

    return (
        metrics,
        labels,
        features,
        classes,
        confusion,
        original_data
    )


try:
    (
        metrics,
        labels,
        features,
        classes,
        confusion,
        original_data
    ) = load_dashboard_data()

except Exception as e:
    st.error(f"Unable to load dashboard data: {e}")
    st.stop()

# =====================================================
# 4. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">'
    '<span class="accent">CAR</span> CLASSIFICATION'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">VEHICLE INTELLIGENCE SYSTEM</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("## 🏎️ VEHICLE INTELLIGENCE")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard Overview",
            "Vehicle Prediction",
            "Prediction History",
            "Model Performance",
            "Data Analytics"
        ]
    )

    st.markdown("---")
    st.caption("Car Type Classification")
    st.caption("Powered by Apache Spark MLlib")
    st.caption("Scala + Python + Streamlit")

# =====================================================
# 6. KPI FUNCTION
# =====================================================

def show_metrics():

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Model Accuracy",
        f"{float(metrics['accuracy']) * 100:.2f}%"
    )

    c2.metric(
        "Weighted F1",
        f"{float(metrics['weighted_f1']) * 100:.2f}%"
    )

    c3.metric(
        "Total Records",
        f"{int(float(metrics['total_records'])):,}"
    )

    c4.metric(
        "Vehicle Classes",
        len(labels)
    )

# =====================================================
# 7. SPARK MODEL LOADER
# =====================================================

@st.cache_resource
def load_spark_model():

    # Set Python and networking configuration
    # BEFORE importing or starting PySpark.

    os.environ["HADOOP_HOME"] = r"C:\Winutils"
    os.environ["hadoop_home_dir"] = r"C:\Winutils"

    os.environ["PATH"] = (
        r"C:\Winutils\bin;" + os.environ.get("PATH", "")
    )

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
    os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("CarClassificationDashboard")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.python.worker.reuse", "false")
        .config("spark.python.use.daemon", "false")
        .config("spark.python.worker.connect.timeout", "60s")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    model = PipelineModel.load(str(MODEL_PATH))

    return spark, model

# =====================================================
# 8. PAGE 1: DASHBOARD OVERVIEW
# =====================================================

if page == "Dashboard Overview":

    st.header("🏁 Vehicle Intelligence Overview")

    show_metrics()

    st.markdown("### Dataset Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Clean Records",
        f"{int(float(metrics['clean_records'])):,}"
    )

    c2.metric(
        "Training Records",
        f"{int(float(metrics['training_records'])):,}"
    )

    c3.metric(
        "Validation Records",
        f"{int(float(metrics['validation_records'])):,}"
    )

    c4.metric(
        "Testing Records",
        f"{int(float(metrics['testing_records'])):,}"
    )

    st.markdown("""
    <div class="glass">
        <h3>🚘 About This Project</h3>
        <p>
        An intelligent vehicle classification system that predicts
        vehicle styles using engine specifications, fuel type,
        transmission, driven wheels and vehicle size.
        </p>
        <p>
        The classification pipeline is trained using
        <b>Scala and Apache Spark MLlib</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Vehicle Style Distribution")

    distribution = (
        original_data["Vehicle Style"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    distribution.columns = ["Vehicle Style", "Count"]

    fig = px.bar(
        distribution,
        x="Vehicle Style",
        y="Count",
        color="Count",
        color_continuous_scale=["#263b59", "#ffad42"],
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Vehicle Style",
        yaxis_title="Number of Vehicles"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 9. PAGE 2: VEHICLE PREDICTION
# =====================================================

elif page == "Vehicle Prediction":

    st.header("🚗 Intelligent Vehicle Prediction")

    st.write(
        "Enter vehicle specifications below. "
        "The saved Spark MLlib model will predict its vehicle style."
    )

    # Load saved Spark model
    try:

        with st.spinner("Loading trained Spark model..."):
            spark, model = load_spark_model()

        st.success("Spark MLlib model loaded successfully.")

    except Exception as e:

        st.error(f"Unable to load Spark model: {e}")
        st.stop()

    # Extract categorical values from saved indexers
    fuel_options = list(model.stages[1].labels)
    transmission_options = list(model.stages[2].labels)
    drive_options = list(model.stages[3].labels)
    size_options = list(model.stages[4].labels)

    # Numeric ranges from original dataset
    numeric_columns = [
        "Year",
        "Engine HP",
        "Engine Cylinders",
        "highway MPG",
        "city mpg"
    ]

    numeric_defaults = {}

    for column in numeric_columns:

        values = pd.to_numeric(
            original_data[column],
            errors="coerce"
        ).dropna()

        numeric_defaults[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median())
        }

    # Engine inputs
    st.markdown("### Engine Specifications")

    c1, c2, c3 = st.columns(3)

    year = c1.number_input(
        "Manufacturing Year",
        min_value=int(numeric_defaults["Year"]["min"]),
        max_value=int(numeric_defaults["Year"]["max"]),
        value=int(numeric_defaults["Year"]["median"]),
        step=1
    )

    hp = c2.number_input(
        "Engine HP",
        min_value=float(numeric_defaults["Engine HP"]["min"]),
        max_value=float(numeric_defaults["Engine HP"]["max"]),
        value=float(numeric_defaults["Engine HP"]["median"])
    )

    cylinders = c3.number_input(
        "Engine Cylinders",
        min_value=float(numeric_defaults["Engine Cylinders"]["min"]),
        max_value=float(numeric_defaults["Engine Cylinders"]["max"]),
        value=float(numeric_defaults["Engine Cylinders"]["median"])
    )

    c4, c5 = st.columns(2)

    highway = c4.number_input(
        "Highway MPG",
        min_value=float(numeric_defaults["highway MPG"]["min"]),
        max_value=float(numeric_defaults["highway MPG"]["max"]),
        value=float(numeric_defaults["highway MPG"]["median"])
    )

    city = c5.number_input(
        "City MPG",
        min_value=float(numeric_defaults["city mpg"]["min"]),
        max_value=float(numeric_defaults["city mpg"]["max"]),
        value=float(numeric_defaults["city mpg"]["median"])
    )

    # Categorical inputs
    st.markdown("### Vehicle Configuration")

    c6, c7 = st.columns(2)

    fuel = c6.selectbox(
        "Engine Fuel Type",
        fuel_options
    )

    transmission = c7.selectbox(
        "Transmission Type",
        transmission_options
    )

    c8, c9 = st.columns(2)

    driven = c8.selectbox(
        "Driven Wheels",
        drive_options
    )

    size = c9.selectbox(
        "Vehicle Size",
        size_options
    )

    # Prediction button
    if st.button(
        "🚀 Predict Vehicle Style",
        use_container_width=True
    ):

        try:

            # Create the input directly using Spark SQL.
            # This avoids creating a Python RDD.

            def sql_number(value):
                return str(float(value))

            def sql_text(value):
                safe_value = str(value).replace("'", "''")
                return "'" + safe_value + "'"

            input_df = spark.sql(f"""
                SELECT
                    {sql_number(year)} AS `Year`,
                    {sql_number(hp)} AS `Engine HP`,
                    {sql_number(cylinders)} AS `Engine Cylinders`,
                    {sql_number(highway)} AS `highway MPG`,
                    {sql_number(city)} AS `city mpg`,
                    {sql_text(fuel)} AS `Engine Fuel Type`,
                    {sql_text(transmission)} AS `Transmission Type`,
                    {sql_text(driven)} AS `Driven_Wheels`,
                    {sql_text(size)} AS `Vehicle Size`
            """)

            # Run actual saved Spark pipeline
            with st.spinner("Predicting vehicle style..."):

                result = model.transform(input_df).first()

            # Decode prediction
            predicted_index = int(result["prediction"])

            predicted_style = (
                model.stages[0].labels[predicted_index]
            )

            # Prediction probabilities
            probability_vector = result["probability"]

            probabilities = probability_vector.toArray().tolist()

            confidence = max(probabilities) * 100
                        # Save prediction history

            history_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Year": year,
                "Engine HP": hp,
                "Engine Cylinders": cylinders,
                "Highway MPG": highway,
                "City MPG": city,
                "Fuel Type": fuel,
                "Transmission": transmission,
                "Driven Wheels": driven,
                "Vehicle Size": size,
                "Predicted Vehicle": predicted_style,
                "Confidence (%)": round(confidence, 2)
            }

            history_df = pd.DataFrame([history_record])

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            write_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
            history_df.to_csv(
                HISTORY_PATH,
                mode="a",
                header=write_header,
                index=False
            )

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            st.markdown(
                f"""
                <div class="glass">
                    <h2 style="color:#ffad42;">
                        {predicted_style}
                    </h2>
                    <p>Predicted Vehicle Style</p>
                    <h3>Model Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Top five class probabilities

            vehicle_labels = list(model.stages[0].labels)

            # Handle the additional Unknown class
            while len(vehicle_labels) < len(probabilities):
                vehicle_labels.append("Unknown / Unseen Label")

            vehicle_labels = vehicle_labels[:len(probabilities)]

            probability_df = pd.DataFrame({
                "Vehicle Style": vehicle_labels,
                "Probability": probabilities
            })

            probability_df = (
                probability_df
                .sort_values("Probability", ascending=False)
                .head(5)
            )

            fig = px.bar(
                probability_df,
                x="Probability",
                y="Vehicle Style",
                orientation="h",
                color="Probability",
                color_continuous_scale=["#263b59", "#ffad42"],
                template="plotly_dark",
                title="Top 5 Prediction Probabilities"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_tickformat=".0%"
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:

            st.error(f"Prediction failed: {e}")

            st.info(
                "Please check the Spark worker connection "
                "and the input values."
            )

# =====================================================
# 10. PAGE 3: PREDICTION HISTORY
# =====================================================

elif page == "Prediction History":

    st.header("📜 Prediction History")

    st.write(
        "View and download all previous vehicle predictions."
    )

    if HISTORY_PATH.exists():

        history = pd.read_csv(HISTORY_PATH)

        if not history.empty:

            # Summary metrics
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Predictions",
                len(history)
            )

            c2.metric(
                "Unique Vehicle Styles",
                history["Predicted Vehicle"].nunique()
            )

            c3.metric(
                "Average Confidence",
                f"{history['Confidence (%)'].mean():.2f}%"
            )

            st.markdown("---")

            st.subheader("Previous Predictions")

            st.dataframe(
                history.sort_values(
                    "timestamp",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            # Download CSV
            csv_data = history.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction History",
                data=csv_data,
                file_name="car_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Vehicle style distribution
            st.subheader("Prediction Distribution")

            prediction_counts = (
                history["Predicted Vehicle"]
                .value_counts()
                .reset_index()
            )

            prediction_counts.columns = [
                "Vehicle Style",
                "Predictions"
            ]

            fig = px.pie(
                prediction_counts,
                names="Vehicle Style",
                values="Predictions",
                hole=0.5,
                template="plotly_dark",
                title="Previously Predicted Vehicle Styles"
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:
            st.info("No predictions recorded yet.")

    else:
        st.info(
            "No prediction history available. "
            "Make your first prediction to see it here."
        )

# =====================================================
# 11. PAGE 4: MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.header("📊 Model Performance Analysis")

    show_metrics()
    
    # Baseline vs Optimized Model Comparison

    st.markdown("---")
    st.subheader("🏆 Baseline vs Optimized Model")

    comparison_data = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Weighted F1",
            "Macro F1"
        ],
        "Baseline Model": [
            67.07,
            66.21,
            56.87
        ],
        "Optimized Model": [
            74.85,
            74.62,
            68.58
        ]
    })

    # Calculate improvement
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Accuracy Improvement",
        "+7.78 pp"
    )

    c2.metric(
        "Weighted F1 Improvement",
        "+8.41 pp"
    )

    c3.metric(
        "Macro F1 Improvement",
        "+11.71 pp"
    )

    # Comparison chart
    comparison_long = comparison_data.melt(
        id_vars="Metric",
        var_name="Model",
        value_name="Score"
    )

    fig = px.bar(
        comparison_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text="Score",
        template="plotly_dark",
        color_discrete_map={
            "Baseline Model": "#526783",
            "Optimized Model": "#ffad42"
        },
        title="Performance Improvement After Hyperparameter Tuning"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="Performance (%)",
        yaxis_range=[0, 100],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title="Model"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("""
    <div class="glass">
        <h3>🔍 Model Optimization Findings</h3>
        <p>
        The optimized Decision Tree model achieved better
        performance across all three evaluation metrics.
        </p>
        <p>
        Hyperparameter tuning selected
        <b>Max Depth = 15</b> and
        <b>Max Bins = 64</b>.
        </p>
        <p>
        Macro F1 improved by 11.71 percentage points,
        indicating better overall class-wise performance.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("### Class-wise Performance")

    selected_classes = st.multiselect(
        "Select vehicle classes",
        classes["vehicle_style"].tolist(),
        default=classes["vehicle_style"].tolist()
    )

    filtered_classes = classes[
        classes["vehicle_style"].isin(selected_classes)
    ].copy()

    if not filtered_classes.empty:

        metric_choice = st.selectbox(
            "Choose evaluation metric",
            ["precision", "recall", "f1_score"]
        )

        fig = px.bar(
            filtered_classes,
            x="vehicle_style",
            y=metric_choice,
            color=metric_choice,
            color_continuous_scale=["#263b59", "#ffad42"],
            template="plotly_dark",
            title=f"Class-wise {metric_choice.replace('_', ' ').title()}"
        )

        fig.update_layout(
            yaxis_range=[0, 1],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Vehicle Style"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.info("Select at least one vehicle class.")

    st.markdown("### Confusion Matrix")

    cm = confusion.set_index("Actual/Predicted")

    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlOrBr",
        labels={
            "x": "Predicted Vehicle",
            "y": "Actual Vehicle",
            "color": "Count"
        },
        template="plotly_dark"
    )

    fig.update_layout(
        height=750,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Detailed Class Metrics")

    display_classes = filtered_classes.copy()

    for column in ["precision", "recall", "f1_score"]:

        display_classes[column] = (
            display_classes[column] * 100
        ).round(2)

    st.dataframe(
        display_classes,
        use_container_width=True,
        hide_index=True
    )

# =====================================================
# 12. PAGE 5: DATA ANALYTICS
# =====================================================

elif page == "Data Analytics":

    st.header("📈 Vehicle Dataset Analytics")

    # Feature importance
    st.markdown("### Feature Importance")

    display_features = features.copy()

    feature_names = {
        "sizeIndex": "Vehicle Size",
        "fuelIndex": "Engine Fuel Type",
        "driveIndex": "Driven Wheels",
        "transmissionIndex": "Transmission Type"
    }

    display_features["feature"] = (
        display_features["feature"].replace(feature_names)
    )

    fig = px.bar(
        display_features.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#263b59", "#ffad42"],
        template="plotly_dark",
        title="Decision Tree Feature Importance"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importance Score",
        yaxis_title="Feature"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Vehicle distribution
    st.markdown("### Vehicle Category Distribution")

    distribution = (
        original_data["Vehicle Style"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    distribution.columns = ["Vehicle Style", "Count"]

    c1, c2 = st.columns(2)

    with c1:

        fig = px.pie(
            distribution,
            names="Vehicle Style",
            values="Count",
            hole=0.48,
            template="plotly_dark",
            title="Vehicle Style Distribution"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        fig = px.bar(
            distribution,
            x="Vehicle Style",
            y="Count",
            color="Count",
            color_continuous_scale=["#263b59", "#ffad42"],
            template="plotly_dark",
            title="Vehicle Category Records"
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Dataset preview
    st.markdown("### Dataset Preview")

    st.dataframe(
        original_data.head(100),
        use_container_width=True
    )

# =====================================================
# 13. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <p style='text-align:center;color:#aab7c8;'>
        CAR CLASSIFICATION SYSTEM |
        Powered by Scala, Apache Spark MLlib & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
