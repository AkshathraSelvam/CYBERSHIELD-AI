import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CyberShield-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 20px;
        font-weight: 500;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# Only the models the app actually uses are loaded.
# ============================================================

@st.cache_resource
def load_models():

    scaler = joblib.load(
        MODELS_DIR / "standard_scaler.pkl"
    )

    binary_encoder = joblib.load(
        MODELS_DIR / "binary_label_encoder.pkl"
    )

    multiclass_encoder = joblib.load(
        MODELS_DIR / "multiclass_label_encoder.pkl"
    )

    feature_names = joblib.load(
        MODELS_DIR / "feature_names.pkl"
    )

    random_forest_binary = joblib.load(
        MODELS_DIR / "random_forest_binary.pkl"
    )

    decision_tree_multiclass = joblib.load(
        MODELS_DIR / "decision_tree_multiclass.pkl"
    )

    isolation_forest = joblib.load(
        MODELS_DIR / "isolation_forest.pkl"
    )

    return (
        scaler,
        binary_encoder,
        multiclass_encoder,
        feature_names,
        random_forest_binary,
        decision_tree_multiclass,
        isolation_forest
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

try:

    (
        scaler,
        binary_encoder,
        multiclass_encoder,
        feature_names,
        random_forest_binary,
        decision_tree_multiclass,
        isolation_forest
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        f"❌ Error loading models: {e}"
    )


# ============================================================
# SESSION HELPERS
# ============================================================

RESULT_KEYS = [
    "cybershield_results",
    "cybershield_input",
    "cybershield_original",
    "cybershield_row_ids",
    "cybershield_file"
]


def clear_results():

    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ CyberShield-AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Intelligent Network Intrusion Detection '
    'and Threat Monitoring System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "CyberShield-AI analyzes network traffic using Machine Learning, "
    "Anomaly Detection, Threat Classification, Risk Scoring, "
    "and Explainable AI."
)


# ============================================================
# MODEL STATUS
# ============================================================

if models_loaded:

    st.success(
        "✅ All CyberShield-AI models loaded successfully!"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model Features",
            len(feature_names)
        )

    with col2:
        st.metric(
            "Binary Detection",
            "Random Forest"
        )

    with col3:
        st.metric(
            "Attack Classification",
            "Decision Tree"
        )

    with col4:
        st.metric(
            "Anomaly Detection",
            "Isolation Forest"
        )

else:

    st.error(
        "❌ Models could not be loaded."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ CyberShield-AI")

    st.markdown("---")

    st.subheader("System Modules")

    st.write("🎯 Binary Attack Detection")
    st.write("🏷️ Attack Type Classification")
    st.write("🚨 Anomaly Detection")
    st.write("🔗 Threat Categorization")
    st.write("⚠️ Risk Scoring")
    st.write("🧠 Explainable AI")

    st.markdown("---")

    st.subheader("Risk Levels")

    st.write("🟢 Low: 0–25")
    st.write("🟡 Medium: 26–50")
    st.write("🟠 High: 51–75")
    st.write("🔴 Critical: 76–100")


# ============================================================
# CSV UPLOAD
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">'
    '📂 Network Traffic Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a CICIDS2017 network traffic CSV file for analysis."
)

uploaded_file = st.file_uploader(
    "Upload Network Traffic CSV",
    type=["csv"]
)

# Old results must never stay on screen after the file changes.
if uploaded_file is None:

    clear_results()

elif st.session_state.get("cybershield_file") != uploaded_file.name:

    # A new file was uploaded: keep the previous file's results only
    # until the user analyzes the new one. Remove them now.
    if "cybershield_file" in st.session_state:
        clear_results()


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    st.success(
        f"✅ File uploaded successfully: "
        f"{uploaded_file.name}"
    )

    file_size_mb = (
        uploaded_file.size / (1024 * 1024)
    )

    st.info(
        f"📁 File size: {file_size_mb:.2f} MB"
    )

    # ========================================================
    # READ CSV
    # ========================================================

    try:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(
            f"❌ Unable to read CSV file: {e}"
        )

        st.stop()

    # ========================================================
    # CLEAN COLUMN NAMES
    # ========================================================

    uploaded_df.columns = (
        uploaded_df.columns
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Dataset Information'
        '</div>',
        unsafe_allow_html=True
    )

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:

        st.metric(
            "Rows",
            f"{uploaded_df.shape[0]:,}"
        )

    with info_col2:

        st.metric(
            "Original Columns",
            uploaded_df.shape[1]
        )

    with info_col3:

        st.metric(
            "Model Features Required",
            len(feature_names)
        )

    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 Feature Validation'
        '</div>',
        unsafe_allow_html=True
    )

    # Remove target columns if they exist
    input_df = uploaded_df.drop(
        columns=[
            "Label",
            "Traffic_Type"
        ],
        errors="ignore"
    ).copy()

    expected_features = list(
        feature_names
    )

    # ========================================================
    # CHECK MISSING FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in expected_features
        if feature not in input_df.columns
    ]

    if missing_features:

        st.error(
            f"❌ {len(missing_features)} required "
            f"features are missing."
        )

        st.write(
            "Missing features:"
        )

        st.write(
            missing_features
        )

        st.stop()

    else:

        st.success(
            f"✅ All {len(expected_features)} "
            f"required model features found!"
        )

    # ========================================================
    # SELECT EXACT 70 FEATURES
    # ========================================================

    X_uploaded = input_df[
        expected_features
    ].copy()

    # ========================================================
    # CONVERT FEATURES TO NUMERIC
    # ========================================================

    for column in X_uploaded.columns:

        X_uploaded[column] = pd.to_numeric(
            X_uploaded[column],
            errors="coerce"
        )

    # ========================================================
    # HANDLE INFINITE VALUES
    # ========================================================

    X_uploaded = X_uploaded.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # ========================================================
    # REMOVE MISSING / INVALID NUMERIC ROWS
    # ========================================================

    original_row_count = len(
        X_uploaded
    )

    missing_mask = X_uploaded.isna().any(
        axis=1
    )

    removed_missing_rows = int(
        missing_mask.sum()
    )

    X_uploaded = X_uploaded.loc[
        ~missing_mask
    ].copy()

    # ========================================================
    # INVALID NEGATIVE VALUES
    # ========================================================

    # IMPORTANT: this list must match the list in
    # notebooks/02_data_preprocessing.ipynb (the invalid negative
    # features cell). If you change one, change the other.
    #
    # These features cannot physically be negative.
    # Rows with a negative value in any of them are removed.
    # Init_Win_bytes_forward and Init_Win_bytes_backward are
    # NOT in this list, because CICIDS2017 uses -1 as a
    # sentinel value for them.

    invalid_negative_features = [
        "Flow IAT Min",
        "Flow Packets/s",
        "Flow Duration",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Flow Bytes/s",
        "Fwd Header Length",
        "Fwd Header Length.1",
        "min_seg_size_forward",
        "Bwd Header Length",
        "Fwd IAT Min"
    ]

    # Only check features that exist in the model's feature list
    # (some may have been removed as constant features).
    negative_check_features = [
        column
        for column in invalid_negative_features
        if column in X_uploaded.columns
    ]

    negative_mask = (
        X_uploaded[
            negative_check_features
        ] < 0
    ).any(axis=1)

    removed_negative_rows = int(
        negative_mask.sum()
    )

    X_uploaded = X_uploaded.loc[
        ~negative_mask
    ].copy()

    # Keep the original row numbers of the rows that survived cleaning.
    # The ground-truth check needs them to line up with the Label column.
    kept_row_ids = X_uploaded.index.to_numpy()

    # ========================================================
    # RESET INDEX
    # ========================================================

    X_uploaded.reset_index(
        drop=True,
        inplace=True
    )

    # ========================================================
    # CLEANING SUMMARY
    # ========================================================

    removed_total = (
        original_row_count
        - len(X_uploaded)
    )

    clean_col1, clean_col2, clean_col3 = (
        st.columns(3)
    )

    with clean_col1:

        st.metric(
            "Original Rows",
            f"{original_row_count:,}"
        )

    with clean_col2:

        st.metric(
            "Rows Removed",
            f"{removed_total:,}"
        )

    with clean_col3:

        st.metric(
            "Rows Ready",
            f"{len(X_uploaded):,}"
        )

    if removed_missing_rows > 0:

        st.warning(
            f"⚠️ {removed_missing_rows:,} rows contained "
            "missing or infinite values and were removed."
        )

    if removed_negative_rows > 0:

        st.warning(
            f"⚠️ {removed_negative_rows:,} rows contained "
            "physically invalid negative network-flow "
            "values and were removed."
        )

    if len(X_uploaded) == 0:

        st.error(
            "❌ No valid rows remain after preprocessing."
        )

        st.stop()

    st.success(
        f"✅ Input prepared successfully: "
        f"{len(X_uploaded):,} rows × "
        f"{len(expected_features)} features"
    )

    # ========================================================
    # FEATURE PREVIEW
    # ========================================================

    with st.expander(
        "🔎 View Model Input Features"
    ):

        st.write(
            "The following 70 features are supplied "
            "to the trained models:"
        )

        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": expected_features
                }
            ),
            use_container_width=True
        )

    # ========================================================
    # RUN ANALYSIS BUTTON
    # ========================================================

    st.markdown("---")

    analyze_button = st.button(
        "🚀 Analyze Network Traffic",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        # ====================================================
        # STEP 1 — BINARY ATTACK DETECTION
        # ====================================================

        with st.spinner(
            "Running binary attack detection..."
        ):

            binary_prediction_encoded = (
                random_forest_binary.predict(
                    X_uploaded
                )
            )

            binary_prediction_labels = (
                binary_encoder.inverse_transform(
                    binary_prediction_encoded
                )
            )

        # ====================================================
        # BINARY CONFIDENCE
        # ====================================================

        try:

            binary_probabilities = (
                random_forest_binary.predict_proba(
                    X_uploaded
                )
            )

            binary_confidence = (
                np.max(
                    binary_probabilities,
                    axis=1
                ) * 100
            )

        except Exception:

            binary_confidence = np.full(
                len(X_uploaded),
                np.nan
            )

        # ====================================================
        # STEP 2 — MULTICLASS ATTACK CLASSIFICATION
        #
        # IMPORTANT:
        # Only rows predicted as ATTACK by the binary model
        # are sent to the multiclass attack classifier.
        # ====================================================

        attack_mask = (
            binary_prediction_labels == "ATTACK"
        )

        multiclass_prediction_labels = np.array(
            ["BENIGN"] * len(X_uploaded),
            dtype=object
        )

        attack_confidence = np.zeros(
            len(X_uploaded)
        )

        if attack_mask.any():

            with st.spinner(
                "Classifying detected attacks..."
            ):

                attack_rows = X_uploaded.loc[
                    attack_mask
                ]

                # Ignore the BENIGN class for rows already flagged
                # as ATTACK, so the output can never say
                # Binary = ATTACK but Attack_Type = BENIGN.
                attack_probabilities = (
                    decision_tree_multiclass.predict_proba(
                        attack_rows
                    )
                )

                benign_code = multiclass_encoder.transform(
                    ["BENIGN"]
                )[0]

                benign_column = list(
                    decision_tree_multiclass.classes_
                ).index(benign_code)

                # -1 (not 0): when a leaf is pure BENIGN, every other
                # class is 0 too, and argmax would tie back to BENIGN.
                attack_probabilities[:, benign_column] = -1

                best_columns = attack_probabilities.argmax(
                    axis=1
                )

                multiclass_prediction_encoded = (
                    decision_tree_multiclass.classes_[
                        best_columns
                    ]
                )

                # Confidence of the best attack class for each row.
                # BENIGN was set to -1 above, so this is the best
                # real attack class probability.
                best_scores = attack_probabilities.max(
                    axis=1
                )

                decoded_labels = (
                    multiclass_encoder.inverse_transform(
                        multiclass_prediction_encoded
                    )
                )

                # If no real attack class has any probability in this
                # leaf, the label would be a guess. Mark it as
                # unclassified instead of showing a random attack type.
                predicted_attack_labels = np.where(
                    best_scores > 0,
                    decoded_labels,
                    "Unclassified_Attack"
                )

                multiclass_prediction_labels[
                    attack_mask
                ] = predicted_attack_labels

            # --------------------------------------------
            # ATTACK CLASSIFICATION CONFIDENCE
            # --------------------------------------------

            try:

                attack_confidence[
                    attack_mask
                ] = (
                    np.max(
                        attack_probabilities,
                        axis=1
                    ) * 100
                )

            except Exception:

                attack_confidence[
                    attack_mask
                ] = np.nan

        # ====================================================
        # STEP 3 — ANOMALY DETECTION
        # ====================================================

        with st.spinner(
            "Running anomaly detection..."
        ):

            X_uploaded_scaled = (
                scaler.transform(
                    X_uploaded
                )
            )

            anomaly_prediction = (
                isolation_forest.predict(
                    X_uploaded_scaled
                )
            )

            anomaly_flag = (
                anomaly_prediction == -1
            ).astype(int)

        # ====================================================
        # STEP 4 — THREAT CATEGORY MAPPING
        # ====================================================

        def map_threat_category(
            attack_label
        ):

            if attack_label == "BENIGN":

                return "BENIGN"

            elif attack_label == "PortScan":

                return "Reconnaissance"

            elif attack_label in [
                "DDoS",
                "DoS_Hulk",
                "DoS_GoldenEye",
                "DoS_Slowhttptest",
                "DoS_Slowloris"
            ]:

                return "Denial_of_Service"

            elif attack_label in [
                "FTP_Patator",
                "SSH_Patator"
            ]:

                return "Credential_Attack"

            elif attack_label in [
                "WebAttack_BruteForce",
                "WebAttack_SQLInjection",
                "WebAttack_XSS"
            ]:

                return "Web_Attack"

            elif attack_label == "Bot":

                return "Malware_or_Bot"

            elif attack_label == "Infiltration":

                return "Infiltration"

            elif attack_label == "Heartbleed":

                return "Vulnerability_Exploit"

            elif attack_label == "Unclassified_Attack":

                return "Unclassified"

            else:

                return "Other"

        threat_categories = [
            map_threat_category(
                label
            )
            for label in multiclass_prediction_labels
        ]

        # ====================================================
        # STEP 5 — RISK SEVERITY POLICY
        # ====================================================

        severity_scores = {

            "BENIGN": 0,

            "Reconnaissance": 40,

            "Denial_of_Service": 70,

            "Credential_Attack": 75,

            "Web_Attack": 70,

            "Malware_or_Bot": 85,

            "Infiltration": 95,

            "Vulnerability_Exploit": 90,

            "Unclassified": 50,

            "Other": 50
        }

        severity = np.array(
            [
                severity_scores.get(
                    category,
                    50
                )
                for category in threat_categories
            ]
        )

        # ====================================================
        # STEP 6 — ANOMALY BONUS
        # ====================================================

        # An anomaly bonus is applied only to detected
        # attack traffic. This prevents a BENIGN prediction
        # from becoming a high-risk attack simply because
        # Isolation Forest marked it unusual.

        anomaly_bonus = (
            anomaly_flag
            * 15
            * attack_mask.astype(int)
        )

        # ====================================================
        # STEP 7 — FINAL RISK SCORE
        # ====================================================

        risk_score = (
            severity
            + anomaly_bonus
        )

        risk_score = np.clip(
            risk_score,
            0,
            100
        )

        # ====================================================
        # STEP 8 — RISK LEVEL
        # ====================================================

        def assign_risk_level(
            score
        ):

            if score <= 25:

                return "Low"

            elif score <= 50:

                return "Medium"

            elif score <= 75:

                return "High"

            else:

                return "Critical"

        risk_levels = [
            assign_risk_level(
                score
            )
            for score in risk_score
        ]

        # ====================================================
        # STEP 9 — RESULT DATAFRAME
        # ====================================================

        results_df = pd.DataFrame(
            {

                "Binary_Detection":
                    binary_prediction_labels,

                "Attack_Type":
                    multiclass_prediction_labels,

                "Threat_Category":
                    threat_categories,

                "Binary_Confidence_%":
                    np.round(
                        binary_confidence,
                        2
                    ),

                "Attack_Confidence_%":
                    np.round(
                        attack_confidence,
                        2
                    ),

                "Anomaly_Flag":
                    anomaly_flag,

                "Severity_Score":
                    severity,

                "Anomaly_Bonus":
                    anomaly_bonus,

                "Risk_Score":
                    risk_score,

                "Risk_Level":
                    risk_levels
            }
        )

        # ====================================================
        # STORE RESULTS IN SESSION
        # ====================================================

        st.session_state[
            "cybershield_results"
        ] = results_df

        st.session_state[
            "cybershield_input"
        ] = X_uploaded

        st.session_state[
            "cybershield_original"
        ] = uploaded_df

        st.session_state[
            "cybershield_row_ids"
        ] = kept_row_ids

        st.session_state[
            "cybershield_file"
        ] = uploaded_file.name

        st.success(
            "✅ Network traffic analysis completed successfully!"
        )


# ============================================================
# DISPLAY RESULTS
# Only shown when the results belong to the file currently uploaded.
# ============================================================

if (
    "cybershield_results" in st.session_state
    and uploaded_file is not None
    and st.session_state.get("cybershield_file") == uploaded_file.name
):

    results_df = st.session_state[
        "cybershield_results"
    ]

    st.caption(
        f"Results for file: {st.session_state['cybershield_file']}"
    )

    # ========================================================
    # DASHBOARD OVERVIEW
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📊 Threat Monitoring Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    total_analyzed = len(
        results_df
    )

    total_attacks = int(
        (
            results_df[
                "Binary_Detection"
            ] == "ATTACK"
        ).sum()
    )

    total_benign = int(
        (
            results_df[
                "Binary_Detection"
            ] == "BENIGN"
        ).sum()
    )

    total_anomalies = int(
        results_df[
            "Anomaly_Flag"
        ].sum()
    )

    critical_count = int(
        (
            results_df[
                "Risk_Level"
            ] == "Critical"
        ).sum()
    )

    metric1, metric2, metric3, metric4, metric5 = (
        st.columns(5)
    )

    with metric1:

        st.metric(
            "Traffic Analyzed",
            f"{total_analyzed:,}"
        )

    with metric2:

        st.metric(
            "Attacks Detected",
            f"{total_attacks:,}"
        )

    with metric3:

        st.metric(
            "Benign Traffic",
            f"{total_benign:,}"
        )

    with metric4:

        st.metric(
            "Anomalies",
            f"{total_anomalies:,}"
        )

    with metric5:

        st.metric(
            "Critical Risk",
            f"{critical_count:,}"
        )

    # ========================================================
    # BINARY + ATTACK TYPE
    # ========================================================

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader(
            "🎯 Binary Detection"
        )

        binary_counts = (
            results_df[
                "Binary_Detection"
            ]
            .value_counts()
            .reindex(
                [
                    "ATTACK",
                    "BENIGN"
                ],
                fill_value=0
            )
        )

        st.bar_chart(
            binary_counts
        )

    with chart_col2:

        st.subheader(
            "🏷️ Attack Type Distribution"
        )

        attack_only_df = results_df[
            results_df[
                "Binary_Detection"
            ] == "ATTACK"
        ]

        attack_counts = (
            attack_only_df[
                "Attack_Type"
            ]
            .value_counts()
        )

        st.bar_chart(
            attack_counts
        )

    # ========================================================
    # THREAT CATEGORY
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔗 Threat Category Distribution"
    )

    attack_category_df = results_df[
        results_df[
            "Binary_Detection"
        ] == "ATTACK"
    ]

    threat_counts = (
        attack_category_df[
            "Threat_Category"
        ]
        .value_counts()
    )

    st.bar_chart(
        threat_counts
    )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    st.markdown("---")

    st.subheader(
        "⚠️ Risk Level Distribution"
    )

    risk_counts = (
        results_df[
            "Risk_Level"
        ]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
                "Critical"
            ],
            fill_value=0
        )
    )

    st.bar_chart(
        risk_counts
    )

    # ========================================================
    # ANOMALY DETECTION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🚨 Anomaly Detection"
    )

    anomaly_col1, anomaly_col2 = (
        st.columns(2)
    )

    with anomaly_col1:

        normal_count = int(
            (
                results_df[
                    "Anomaly_Flag"
                ] == 0
            ).sum()
        )

        anomaly_count = int(
            (
                results_df[
                    "Anomaly_Flag"
                ] == 1
            ).sum()
        )

        anomaly_summary = pd.Series(
            {
                "Normal": normal_count,
                "Anomaly": anomaly_count
            }
        )

        st.bar_chart(
            anomaly_summary
        )

    with anomaly_col2:

        anomaly_percentage = (
            anomaly_count
            / total_analyzed
            * 100
        )

        st.metric(
            "Anomaly Rate",
            f"{anomaly_percentage:.2f}%"
        )

        st.info(
            "Isolation Forest identifies traffic "
            "that is unusual relative to benign "
            "training traffic. An anomaly is not "
            "automatically equivalent to an attack."
        )

    # ========================================================
    # HIGH-RISK TRAFFIC
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🚨 High-Risk Traffic"
    )

    high_risk_df = results_df[
        results_df[
            "Risk_Level"
        ].isin(
            [
                "High",
                "Critical"
            ]
        )
    ].copy()

    if len(high_risk_df) > 0:

        st.dataframe(
            high_risk_df.head(100),
            use_container_width=True
        )

    else:

        st.success(
            "✅ No High or Critical risk "
            "traffic detected."
        )

    # ========================================================
    # COMPLETE RESULTS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📋 Detailed Prediction Results"
    )

    st.dataframe(
        results_df.head(500),
        use_container_width=True
    )

    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    csv_results = (
        results_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Prediction Results",
        data=csv_results,
        file_name="cybershield_ai_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ========================================================
    # GROUND-TRUTH CHECK (only when the file has a Label column)
    # ========================================================

    original_df = st.session_state.get(
        "cybershield_original"
    )

    row_ids = st.session_state.get(
        "cybershield_row_ids"
    )

    if (
        original_df is not None
        and row_ids is not None
        and "Label" in original_df.columns
    ):

        st.markdown("---")

        st.subheader(
            "✅ Ground-Truth Check (Binary Detection)"
        )

        st.write(
            "The file's own Label column is compared with the "
            "predictions for the same rows. This shows how well "
            "the binary model performs on this file."
        )

        true_labels = (
            original_df.loc[row_ids, "Label"]
            .astype(str)
            .str.strip()
        )

        y_true = np.where(
            true_labels == "BENIGN",
            "BENIGN",
            "ATTACK"
        )

        y_pred = results_df[
            "Binary_Detection"
        ].to_numpy()

        gt_accuracy = accuracy_score(
            y_true,
            y_pred
        )

        gt_precision = precision_score(
            y_true,
            y_pred,
            pos_label="ATTACK",
            zero_division=0
        )

        gt_recall = recall_score(
            y_true,
            y_pred,
            pos_label="ATTACK",
            zero_division=0
        )

        gt_f1 = f1_score(
            y_true,
            y_pred,
            pos_label="ATTACK",
            zero_division=0
        )

        gt_col1, gt_col2, gt_col3, gt_col4 = st.columns(4)

        with gt_col1:

            st.metric(
                "Accuracy",
                f"{gt_accuracy * 100:.2f}%"
            )

        with gt_col2:

            st.metric(
                "Precision (ATTACK)",
                f"{gt_precision * 100:.2f}%"
            )

        with gt_col3:

            st.metric(
                "Recall (ATTACK)",
                f"{gt_recall * 100:.2f}%"
            )

        with gt_col4:

            st.metric(
                "F1 (ATTACK)",
                f"{gt_f1 * 100:.2f}%"
            )

        gt_matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=["ATTACK", "BENIGN"]
        )

        st.write("Confusion matrix (rows = true label, columns = predicted):")

        st.dataframe(
            pd.DataFrame(
                gt_matrix,
                index=["True ATTACK", "True BENIGN"],
                columns=["Predicted ATTACK", "Predicted BENIGN"]
            ),
            use_container_width=True
        )

    # ========================================================
    # XAI
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🧠 Explainable AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The following features had the highest relative "
        "importance in the trained Random Forest model "
        "for distinguishing benign and attack traffic."
    )

    try:

        feature_importances = (
            random_forest_binary
            .feature_importances_
        )

        importance_df = pd.DataFrame(
            {
                "Feature":
                    list(feature_names),

                "Importance":
                    feature_importances
            }
        )

        top_15 = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .head(15)
            .set_index("Feature")
        )

        st.bar_chart(
            top_15["Importance"]
        )

    except Exception as e:

        st.warning(
            f"Unable to display feature importance: {e}"
        )

    # ========================================================
    # ORIGINAL DATASET LABEL INFORMATION
    # ========================================================

    if (
        original_df is not None
        and "Label" in original_df.columns
    ):

        st.markdown("---")

        st.subheader(
            "📌 Dataset Label Information"
        )

        st.info(
            "The uploaded CICIDS2017 file contains its "
            "original Label column. This information is "
            "shown for reference only. CyberShield-AI "
            "generates predictions independently using "
            "the trained models."
        )

        actual_label_counts = (
            original_df[
                "Label"
            ]
            .value_counts()
            .head(15)
        )

        st.bar_chart(
            actual_label_counts
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CyberShield-AI | AI-Based Intelligent Network "
    "Intrusion Detection and Threat Monitoring System"
)
