# ============================================================
# ETHEKWINI LOCAL GOVERNMENT ELECTION ANALYTICS
# STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="eThekwini Election Analytics",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 0px;
}

.sub-title {
    font-size: 17px;
    color: #777;
    margin-top: 0px;
    margin-bottom: 25px;
}

.section-box {
    padding: 18px;
    border-radius: 12px;
    background-color: rgba(120, 120, 120, 0.08);
    margin-top: 10px;
    margin-bottom: 20px;
}

.small-text {
    color: #777;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("ETH_Ward.csv")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert date
    if "DateGenerated" in df.columns:
        df["DateGenerated"] = pd.to_datetime(
            df["DateGenerated"],
            errors="coerce"
        )

    return df


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "ETH_Ward.csv was not found.\n\n"
        "Place ETH_Ward.csv in the same folder as app.py."
    )

    st.stop()


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Ward",
    "VotingDistrict",
    "VotingStationName",
    "RegisteredVoters",
    "PartyName",
    "TotalValidVotes"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "The dataset is missing these required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# 5. PREPARE ELECTION DATA
# ============================================================

# IMPORTANT:
# RegisteredVoters repeats across party rows.
# We remove duplicates before calculating registered voters.

district_voters = df[
    [
        "VotingDistrict",
        "VotingStationName",
        "Ward",
        "RegisteredVoters"
    ]
].drop_duplicates()


# ------------------------------------------------------------
# PARTY TOTALS
# ------------------------------------------------------------

party_votes = (
    df.groupby("PartyName")["TotalValidVotes"]
    .sum()
    .sort_values(ascending=False)
)


total_valid_votes = party_votes.sum()


party_summary = party_votes.reset_index()

party_summary.columns = [
    "PartyName",
    "TotalVotes"
]


party_summary["VotePercentage"] = (
    party_summary["TotalVotes"]
    / total_valid_votes
    * 100
).round(2)


# ------------------------------------------------------------
# REGISTERED VOTERS BY WARD
# ------------------------------------------------------------

registered_by_ward = (
    district_voters
    .groupby("Ward")["RegisteredVoters"]
    .sum()
    .reset_index()
)


# ------------------------------------------------------------
# VALID VOTES BY WARD
# ------------------------------------------------------------

votes_by_ward = (
    df.groupby("Ward")["TotalValidVotes"]
    .sum()
    .reset_index()
)


# ------------------------------------------------------------
# PARTY VOTES BY WARD
# ------------------------------------------------------------

ward_party_votes = (
    df.groupby(
        ["Ward", "PartyName"]
    )["TotalValidVotes"]
    .sum()
    .reset_index()
)


# ------------------------------------------------------------
# HISTORICAL LEADING PARTY
# ------------------------------------------------------------

historical_winners = ward_party_votes.loc[
    ward_party_votes.groupby("Ward")[
        "TotalValidVotes"
    ].idxmax()
].copy()


historical_winners = (
    historical_winners
    .sort_values("Ward")
    .reset_index(drop=True)
)


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.title("🗳️ Election Analytics")

st.sidebar.caption(
    "eThekwini Municipality"
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🏛️ Party Analysis",
        "📍 Ward Analysis",
        "🤖 Machine Learning",
        "🔮 Predictions",
        "🔎 Data Explorer",
        "ℹ️ About Project"
    ]
)


st.sidebar.markdown("---")


st.sidebar.write(
    "**Dataset:** eThekwini Election Data"
)


st.sidebar.write(
    f"**Records:** {len(df):,}"
)


st.sidebar.write(
    f"**Wards:** {df['Ward'].nunique()}"
)


st.sidebar.write(
    f"**Parties:** {df['PartyName'].nunique()}"
)


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.markdown(
    """
    <p class="main-title">
    🗳️ eThekwini Election Analytics
    </p>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <p class="sub-title">
    Local Government Election Data Analytics,
    Machine Learning and Interactive Visualisation
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("Election Overview")


    st.write(
        """
        This page provides a high-level summary of the
        eThekwini Local Government Election dataset.
        """
    )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Wards",
            f"{df['Ward'].nunique():,}"
        )


    with col2:

        st.metric(
            "Voting Districts",
            f"{df['VotingDistrict'].nunique():,}"
        )


    with col3:

        st.metric(
            "Political Parties",
            f"{df['PartyName'].nunique():,}"
        )


    with col4:

        total_registered = (
            district_voters[
                "RegisteredVoters"
            ].sum()
        )

        st.metric(
            "Registered Voters",
            f"{total_registered:,}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # SECOND KPI ROW
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )


    with col2:

        st.metric(
            "Total Valid Votes",
            f"{total_valid_votes:,}"
        )


    with col3:

        st.metric(
            "Voting Stations",
            f"{df['VotingStationName'].nunique():,}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # TOP POLITICAL PARTIES
    # --------------------------------------------------------

    st.subheader(
        "Top 10 Political Parties"
    )


    top_10_parties = (
        party_summary
        .head(10)
        .sort_values(
            "TotalVotes",
            ascending=True
        )
    )


    fig = px.bar(
        top_10_parties,
        x="TotalVotes",
        y="PartyName",
        orientation="h",
        text="TotalVotes",
        title=(
            "Top 10 Political Parties "
            "by Total Valid Votes"
        )
    )


    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )


    fig.update_layout(
        xaxis_title="Total Valid Votes",
        yaxis_title="Political Party"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # PARTY VOTE SHARE
    # --------------------------------------------------------

    st.subheader(
        "Top 10 Party Vote Share"
    )


    top_share = (
        party_summary
        .head(10)
    )


    fig = px.pie(
        top_share,
        values="TotalVotes",
        names="PartyName",
        hole=0.4,
        title="Vote Share of Top 10 Political Parties"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # REGISTERED VOTERS BY WARD
    # --------------------------------------------------------

    st.subheader(
        "Top Wards by Registered Voters"
    )


    top_registered = (
        registered_by_ward
        .sort_values(
            "RegisteredVoters",
            ascending=False
        )
        .head(10)
        .sort_values(
            "RegisteredVoters"
        )
    )


    fig = px.bar(
        top_registered,
        x="RegisteredVoters",
        y="Ward",
        orientation="h",
        title=(
            "Top 10 Wards by Registered Voters"
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE 2: PARTY ANALYSIS
# ============================================================

elif page == "🏛️ Party Analysis":

    st.header(
        "Political Party Analysis"
    )


    selected_party = st.selectbox(
        "Select Political Party",
        sorted(
            df["PartyName"]
            .dropna()
            .unique()
        )
    )


    party_data = df[
        df["PartyName"] == selected_party
    ].copy()


    party_total_votes = (
        party_data[
            "TotalValidVotes"
        ].sum()
    )


    if total_valid_votes > 0:

        party_percentage = (
            party_total_votes
            / total_valid_votes
            * 100
        )

    else:

        party_percentage = 0


    party_wards = (
        party_data[
            "Ward"
        ].nunique()
    )


    party_historical_wins = (
        historical_winners[
            historical_winners[
                "PartyName"
            ] == selected_party
        ]["Ward"].nunique()
    )


    # --------------------------------------------------------
    # PARTY KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "Total Votes",
            f"{party_total_votes:,}"
        )


    with col2:

        st.metric(
            "Overall Vote Share",
            f"{party_percentage:.2f}%"
        )


    with col3:

        st.metric(
            "Wards Contested",
            f"{party_wards:,}"
        )


    with col4:

        st.metric(
            "Historical Ward Leads",
            f"{party_historical_wins:,}"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # PARTY PERFORMANCE BY WARD
    # --------------------------------------------------------

    party_ward_results = (
        party_data
        .groupby("Ward")[
            "TotalValidVotes"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "TotalValidVotes",
            ascending=False
        )
    )


    st.subheader(
        f"{selected_party} Performance by Ward"
    )


    fig = px.bar(
        party_ward_results.head(20),
        x="Ward",
        y="TotalValidVotes",
        title=(
            f"Top 20 Wards for "
            f"{selected_party}"
        )
    )


    fig.update_layout(
        xaxis_title="Ward",
        yaxis_title="Valid Votes"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    st.subheader(
        "Party Ward Results"
    )


    st.dataframe(
        party_ward_results,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3: WARD ANALYSIS
# ============================================================

elif page == "📍 Ward Analysis":

    st.header(
        "Ward Analysis"
    )


    selected_ward = st.selectbox(
        "Select Ward",
        sorted(
            df["Ward"]
            .dropna()
            .unique()
        )
    )


    ward_data = df[
        df["Ward"] == selected_ward
    ].copy()


    ward_registered = (
        district_voters[
            district_voters[
                "Ward"
            ] == selected_ward
        ]["RegisteredVoters"].sum()
    )


    ward_valid_votes = (
        ward_data[
            "TotalValidVotes"
        ].sum()
    )


    ward_districts = (
        ward_data[
            "VotingDistrict"
        ].nunique()
    )


    ward_stations = (
        ward_data[
            "VotingStationName"
        ].nunique()
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            "Registered Voters",
            f"{ward_registered:,}"
        )


    with col2:

        st.metric(
            "Valid Votes",
            f"{ward_valid_votes:,}"
        )


    with col3:

        st.metric(
            "Voting Districts",
            ward_districts
        )


    with col4:

        st.metric(
            "Voting Stations",
            ward_stations
        )


    # --------------------------------------------------------
    # WARD PARTY RESULTS
    # --------------------------------------------------------

    ward_results = (
        ward_data
        .groupby("PartyName")[
            "TotalValidVotes"
        ]
        .sum()
        .reset_index()
        .sort_values(
            "TotalValidVotes",
            ascending=False
        )
    )


    if not ward_results.empty:

        leader = ward_results.iloc[0]


        st.success(
            f"Historical leading party in "
            f"{selected_ward}: "
            f"{leader['PartyName']} "
            f"with "
            f"{int(leader['TotalValidVotes']):,} "
            f"valid votes."
        )


    st.subheader(
        f"Political Party Results — "
        f"{selected_ward}"
    )


    chart_data = (
        ward_results
        .head(15)
        .sort_values(
            "TotalValidVotes"
        )
    )


    fig = px.bar(
        chart_data,
        x="TotalValidVotes",
        y="PartyName",
        orientation="h",
        title=(
            f"Top Political Parties "
            f"in {selected_ward}"
        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.dataframe(
        ward_results,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 4: MACHINE LEARNING
# ============================================================

elif page == "🤖 Machine Learning":

    st.header(
        "Machine Learning Model Evaluation"
    )


    st.info(
        """
        The machine-learning component investigates
        whether available election features can classify
        a political party as a ward winner or non-winner.
        """
    )


    # --------------------------------------------------------
    # MODEL SUMMARY FILE
    # --------------------------------------------------------

    if os.path.exists(
        "model_evaluation_summary.csv"
    ):

        model_summary = pd.read_csv(
            "model_evaluation_summary.csv"
        )


        st.subheader(
            "Final Model Performance"
        )


        st.dataframe(
            model_summary,
            use_container_width=True,
            hide_index=True
        )


        if (
            "Metric" in model_summary.columns
            and
            "Score" in model_summary.columns
        ):

            fig = px.bar(
                model_summary,
                x="Metric",
                y="Score",
                text="Score",
                title=(
                    "Final Model Evaluation Metrics"
                )
            )


            fig.update_yaxes(
                range=[0, 1]
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


    else:

        st.warning(
            "model_evaluation_summary.csv "
            "was not found.\n\n"
            "Run Step 6 in the notebook first."
        )


    st.markdown("---")


    # --------------------------------------------------------
    # MODEL EXPLANATION
    # --------------------------------------------------------

    st.subheader(
        "Machine Learning Approach"
    )


    st.markdown(
        """
        **Problem Type:** Binary Classification

        **Target:** `Winner`

        - `Winner = 1` means the party is the leading
          party in a ward.
        - `Winner = 0` means the party is not the
          leading party.

        **Models Tested:**

        1. Logistic Regression
        2. Decision Tree
        3. Random Forest

        **Evaluation Metrics:**

        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix
        - ROC-AUC

        The Random Forest model was further improved
        using Grid Search and Stratified Cross-Validation.
        """
    )


    st.warning(
        """
        Accuracy should not be interpreted alone because
        the target variable is imbalanced. F1-score,
        precision and recall provide additional information
        about the model's ability to identify winners.
        """
    )


# ============================================================
# PAGE 5: PREDICTIONS
# ============================================================

elif page == "🔮 Predictions":

    st.header(
        "Ward-Level Model Predictions"
    )


    st.warning(
        """
        These results are machine-learning classifications
        based on the available historical election data.

        They are not guaranteed predictions of the
        2026 Local Government Election.
        """
    )


    prediction_file = (
        "predicted_ward_winners.csv"
    )


    if os.path.exists(
        prediction_file
    ):

        predictions = pd.read_csv(
            prediction_file
        )


        required_prediction_columns = [
            "Ward",
            "PartyName",
            "WinnerProbability"
        ]


        prediction_columns_exist = all(
            column in predictions.columns
            for column
            in required_prediction_columns
        )


        if prediction_columns_exist:

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            predicted_counts = (
                predictions[
                    "PartyName"
                ]
                .value_counts()
                .reset_index()
            )


            predicted_counts.columns = [
                "PartyName",
                "PredictedWards"
            ]


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Predicted Wards",
                    predictions[
                        "Ward"
                    ].nunique()
                )


            with col2:

                st.metric(
                    "Predicted Leading Parties",
                    predictions[
                        "PartyName"
                    ].nunique()
                )


            # ------------------------------------------------
            # PREDICTED WARDS GRAPH
            # ------------------------------------------------

            st.subheader(
                "Predicted Leading Parties"
            )


            chart_counts = (
                predicted_counts
                .sort_values(
                    "PredictedWards"
                )
            )


            fig = px.bar(
                chart_counts,
                x="PredictedWards",
                y="PartyName",
                orientation="h",
                title=(
                    "Model-Selected Leading "
                    "Parties by Number of Wards"
                )
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            # ------------------------------------------------
            # WARD SEARCH
            # ------------------------------------------------

            st.subheader(
                "View a Ward Prediction"
            )


            prediction_ward = st.selectbox(
                "Select Ward",
                sorted(
                    predictions[
                        "Ward"
                    ].unique()
                )
            )


            selected_prediction = (
                predictions[
                    predictions[
                        "Ward"
                    ] == prediction_ward
                ]
            )


            if not selected_prediction.empty:

                result = (
                    selected_prediction
                    .iloc[0]
                )


                probability = (
                    float(
                        result[
                            "WinnerProbability"
                        ]
                    )
                    * 100
                )


                col1, col2 = (
                    st.columns(2)
                )


                with col1:

                    st.metric(
                        "Model-Selected Party",
                        result[
                            "PartyName"
                        ]
                    )


                with col2:

                    st.metric(
                        "Model Probability",
                        f"{probability:.2f}%"
                    )


            # ------------------------------------------------
            # PREDICTION TABLE
            # ------------------------------------------------

            st.subheader(
                "All Ward Predictions"
            )


            display_predictions = (
                predictions.copy()
            )


            display_predictions[
                "WinnerProbability"
            ] = (
                display_predictions[
                    "WinnerProbability"
                ]
                * 100
            ).round(2)


            display_predictions = (
                display_predictions.rename(
                    columns={
                        "WinnerProbability":
                        "Winner Probability (%)"
                    }
                )
            )


            st.dataframe(
                display_predictions,
                use_container_width=True,
                hide_index=True
            )


            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            prediction_csv = (
                display_predictions
                .to_csv(
                    index=False
                )
                .encode("utf-8")
            )


            st.download_button(
                "Download Prediction Results",
                data=prediction_csv,
                file_name=(
                    "ward_prediction_results.csv"
                ),
                mime="text/csv"
            )


        else:

            st.error(
                "The prediction file exists, "
                "but it does not contain the "
                "required columns."
            )


    else:

        st.error(
            "predicted_ward_winners.csv "
            "was not found."
        )


        st.info(
            """
            Go to your Jupyter Notebook and run Step 6.

            Step 6 should generate:

            - predicted_ward_winners.csv
            - actual_vs_predicted_winners.csv
            - model_evaluation_summary.csv
            """
        )


# ============================================================
# PAGE 6: DATA EXPLORER
# ============================================================

elif page == "🔎 Data Explorer":

    st.header(
        "Election Data Explorer"
    )


    st.write(
        """
        Filter the election dataset using the controls below.
        """
    )


    col1, col2 = st.columns(2)


    with col1:

        selected_wards = st.multiselect(
            "Filter by Ward",
            sorted(
                df["Ward"]
                .dropna()
                .unique()
            )
        )


    with col2:

        selected_parties = (
            st.multiselect(
                "Filter by Political Party",
                sorted(
                    df["PartyName"]
                    .dropna()
                    .unique()
                )
            )
        )


    filtered_df = df.copy()


    if selected_wards:

        filtered_df = filtered_df[
            filtered_df[
                "Ward"
            ].isin(
                selected_wards
            )
        ]


    if selected_parties:

        filtered_df = filtered_df[
            filtered_df[
                "PartyName"
            ].isin(
                selected_parties
            )
        ]


    # --------------------------------------------------------
    # FILTER METRICS
    # --------------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )


    with col1:

        st.metric(
            "Filtered Records",
            f"{len(filtered_df):,}"
        )


    with col2:

        st.metric(
            "Wards",
            filtered_df[
                "Ward"
            ].nunique()
        )


    with col3:

        st.metric(
            "Political Parties",
            filtered_df[
                "PartyName"
            ].nunique()
        )


    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # DOWNLOAD FILTERED DATA
    # --------------------------------------------------------

    filtered_csv = (
        filtered_df
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    st.download_button(
        label="Download Filtered Dataset",
        data=filtered_csv,
        file_name=(
            "filtered_election_data.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# PAGE 7: ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.header(
        "About the Project"
    )


    st.markdown(
        """
        ### Project Title

        **eThekwini Local Government Election Analytics
        and Machine Learning**

        ### Project Purpose

        This project demonstrates how election data can
        be analysed using Python, machine learning and
        interactive visualisation.

        ### Main Technologies

        - Python
        - Pandas
        - NumPy
        - Matplotlib
        - Plotly
        - Scikit-learn
        - Streamlit
        - Joblib

        ### Machine Learning Models

        - Logistic Regression
        - Decision Tree
        - Random Forest

        ### Final Model

        Random Forest was further optimised using
        Grid Search and Stratified Cross-Validation.

        ### Dashboard Features

        - Election overview
        - Political party analysis
        - Ward analysis
        - Machine-learning evaluation
        - Ward-level model predictions
        - Interactive data exploration
        """
    )


    st.subheader(
        "Important Project Limitation"
    )


    st.warning(
        """
        The available dataset contains limited historical
        election information.

        Therefore, the machine-learning outputs should
        be interpreted as analytical classifications based
        on the available data rather than guaranteed
        predictions of future election results.

        A stronger future forecasting system would require
        multiple historical election periods and additional
        variables such as previous vote share, turnout trends,
        party growth or decline and demographic information.
        """
    )


    st.subheader(
        "Student Information"
    )


    st.write(
        """
        **Student:** Amahle Ndlela 22410926

        **Programme:** Diploma in Information Technology:
        Software Development

        **Institution:** Mangosuthu University of Technology
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.caption(
    "eThekwini Local Government Election Analytics "
    "| Data Analytics • Machine Learning • Streamlit"
)