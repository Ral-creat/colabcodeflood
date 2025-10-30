# app.py
# Streamlit wrapper for floodpatternv2 (keeps logic intact, converts prints/displays/Colab upload to Streamlit UI)
# Author: converted for Streamlit (logic preserved)
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# ML / stats imports (kept as in original)
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Flood Pattern - Dashboard", layout="wide")

st.title("Flood Pattern Data Mining & Forecasting — Streamlit Port")
st.caption("Notebook converted to a tab-based Streamlit app — original logic preserved. ✨")

tabs = st.tabs([
    "Upload & Overview",
    "Data Cleaning & Preprocessing",
    "K-Means Clustering",
    "Flood Prediction (Random Forest)",
    "Flood Severity Model",
    "Time Series Forecast (SARIMA)",
    "Raw Notebook Output (logs)"
])

# We'll keep a simple log area to show textual outputs from notebook-style prints
log_lines = []

def log(txt):
    log_lines.append(txt)

# ---------- TAB 1: Upload & Overview ----------
with tabs[0]:
    st.header("Upload & Overview")
    st.write("Upload your `FloodDataMDRRMO.csv` file (the notebook expects that filename).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload the CSV to proceed. The original notebook used Colab's files.upload().")
    else:
        # Read file into the same variable used in notebook: df
        try:
            uploaded_bytes = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(uploaded_bytes))
            st.success(f"Loaded `{uploaded_file.name}` — preserving variable name `df`.")
            st.subheader("DataFrame shape & dtypes")
            st.write(df.shape)
            st.dataframe(df.head())
            st.write("Data types:")
            st.write(df.dtypes)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

# We set df in session_state so subsequent tabs can access it
if 'df' not in st.session_state:
    st.session_state['df'] = None
if uploaded_file is not None:
    st.session_state['df'] = df

# ---------- TAB 2: Data Cleaning & Preprocessing ----------
with tabs[1]:
    st.header("Data Cleaning & Preprocessing")
    st.write("This tab runs the cleaning steps from the original notebook. I kept variable names and transformations identical.")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload your CSV in the Upload tab first.")
    else:
        try:
            # The original notebook did many prints - we replicate the outputs here using st
            log("Starting data cleaning...")

            st.subheader("Initial shape and info")
            st.write(df.shape)
            st.write(df.dtypes)

            # Show missing values originally printed
            st.subheader("Missing values per column (before cleaning)")
            st.write(df.isnull().sum())

            # The notebook used df['Water Level'] cleaning steps:
            st.subheader("Cleaning 'Water Level' column (original transformations applied)")
            # ensure string operations behave the same
            df['Water Level'] = df['Water Level'].astype(str).str.replace(' ft.', '', regex=False).str.replace(' ft', '', regex=False).str.replace(' ', '', regex=False)
            df['Water Level'] = df['Water Level'].str.replace('ft', '', regex=False).replace('nan', pd.NA)
            df['Water Level'] = pd.to_numeric(df['Water Level'], errors='coerce')

            # Impute median as in original
            median_water_level = df['Water Level'].median()
            df['Water Level'].fillna(median_water_level, inplace=True)

            st.write("Unique sample of 'Water Level' after cleaning:")
            st.write(df['Water Level'].unique()[:20])

            st.subheader("Converting and imputing other columns (as in notebook)")
            # No. of Families affected
            df['No. of Families affected'] = pd.to_numeric(df['No. of Families affected'], errors='coerce')
            median_families = df['No. of Families affected'].median()
            df['No. of Families affected'].fillna(median_families, inplace=True)

            # Damage Infrastructure / Agriculture cleaning
            df['Damage Infrastructure'] = df['Damage Infrastructure'].astype(str).str.replace(',', '', regex=False)
            df['Damage Infrastructure'] = pd.to_numeric(df['Damage Infrastructure'], errors='coerce')

            df['Damage Agriculture'] = df['Damage Agriculture'].astype(str).str.replace(',', '', regex=False)
            df['Damage Agriculture'] = df['Damage Agriculture'].str.replace('422.510.5', '4225105', regex=False)
            df['Damage Agriculture'] = pd.to_numeric(df['Damage Agriculture'], errors='coerce')

            df['Damage Infrastructure'].fillna(0, inplace=True)
            df['Damage Agriculture'].fillna(0, inplace=True)

            # Month/Day/Year imputation (backfill)
            df['Month'].fillna(method='bfill', inplace=True)
            df['Day'].fillna(method='bfill', inplace=True)
            df['Year'].fillna(method='bfill', inplace=True)

            st.write("Missing values per column after cleaning:")
            st.write(df.isnull().sum())

            st.subheader("Summary statistics (numeric columns)")
            st.dataframe(df.describe())

            # show histogram for Water Level
            st.subheader("Water Level distribution (histogram)")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df['Water Level'], bins=20, edgecolor='black')
            ax.set_title('Distribution of Water Level')
            ax.set_xlabel('Water Level')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

            # Save cleaned df back to session_state
            st.session_state['df'] = df
            log("Data cleaning complete.")
            st.success("Data cleaning steps executed and saved to session state.")
        except Exception as e:
            st.error(f"Error during cleaning steps: {e}")
            log(f"Cleaning error: {e}")

# ---------- TAB 3: K-Means Clustering ----------
with tabs[2]:
    st.header("K-Means Clustering")
    st.write("This runs the K-Means section exactly like the notebook and shows cluster summaries.")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload & clean data first.")
    else:
        try:
            # pick selected columns as in notebook
            selected_columns_df = df[['Municipality', 'Barangay', 'Flood Cause', 'Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']]

            st.subheader("Selected columns (head)")
            st.dataframe(selected_columns_df.head())

            # one-hot encode categorical cols
            categorical_cols = selected_columns_df.select_dtypes(include='object').columns
            encoded_df = pd.get_dummies(selected_columns_df, columns=categorical_cols, dummy_na=False)

            st.subheader("Transformed Data (sample)")
            st.dataframe(encoded_df.head())

            # run KMeans with n_clusters=3 (as original)
            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(encoded_df)

            # add cluster labels to df (same variable as notebook)
            df['Cluster'] = kmeans.labels_

            st.subheader("Cluster counts")
            st.write(df['Cluster'].value_counts())

            st.subheader("Cluster numerical summaries")
            cluster_summary_numerical = df.groupby('Cluster')[['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']].agg(['count', 'mean', 'median', 'std']).reset_index()
            st.dataframe(cluster_summary_numerical)

            st.subheader("Cluster categorical distributions (examples)")
            categorical_cols_list = ['Municipality', 'Barangay', 'Flood Cause']
            for col in categorical_cols_list:
                st.write(f"Distribution of '{col}' per cluster (normalized):")
                cluster_summary_categorical = df.groupby('Cluster')[col].value_counts(normalize=True).unstack(fill_value=0)
                st.dataframe(cluster_summary_categorical)

            st.session_state['df'] = df
            log("KMeans clustering complete.")
            st.success("KMeans clustering done.")
        except Exception as e:
            st.error(f"Error during clustering: {e}")
            log(f"KMeans error: {e}")

# ---------- TAB 4: Flood Prediction (Random Forest) ----------
with tabs[3]:
    st.header("Flood Prediction (Random Forest)")
    st.write("This executes the flood occurrence prediction portion (RandomForestClassifier) as in the notebook.")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload & clean data first.")
    else:
        try:
            # create binary target flood_occurred as in notebook
            df['flood_occurred'] = (df['Water Level'] > 0).astype(int)

            st.subheader("Target distribution: flood_occurred")
            st.write(df['flood_occurred'].value_counts())

            # month dummies creation (not changing column names)
            df['Month'] = df['Month'].fillna('Unknown')
            month_dummies = pd.get_dummies(df['Month'], prefix='Month')

            # feature_cols as in notebook
            feature_cols = ['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture'] + list(month_dummies.columns)
            X = pd.concat([df[['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']], month_dummies], axis=1)
            y = df['flood_occurred']

            st.subheader("Feature sample (X) and target (y) sample")
            st.dataframe(X.head())
            st.write(y.head())

            # Train RandomForest (not changing params)
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            st.success("RandomForest model trained on full data (as in notebook).")

            # Evaluate - the notebook used a train_test_split after model.fit; reproduce that sequence
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.subheader("Evaluation on test split")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Monthly flood probabilities (as in notebook)
            monthly_flood_counts = df.groupby('Month')['flood_occurred'].sum()
            monthly_total_counts = df.groupby('Month')['flood_occurred'].count()
            monthly_flood_probability = monthly_flood_counts / monthly_total_counts
            sorted_monthly_flood_probability = monthly_flood_probability.sort_values(ascending=False)

            st.subheader("Monthly Flood Probabilities (sorted)")
            st.dataframe(sorted_monthly_flood_probability.reset_index().rename(columns={0: 'probability'}))

            # plot monthly probabilities
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            sorted_monthly_flood_probability.plot(kind='bar', ax=ax2)
            ax2.set_title('Monthly Flood Probability')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Probability of Flood Occurrence')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig2)

            # store model & related in session for later tabs if needed
            st.session_state['model'] = model
            st.session_state['month_dummies'] = month_dummies
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['df'] = df
            log("RandomForest prediction complete.")
        except Exception as e:
            st.error(f"Error in Flood Prediction tab: {e}")
            log(f"Flood prediction error: {e}")

# ---------- TAB 5: Flood Severity Model ----------
with tabs[4]:
    st.header("Flood Severity Model")
    st.write("This converts `Water Level` to severity categories and trains a multiclass RandomForest as per notebook.")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload & clean data first.")
    else:
        try:
            # Categorize severity exactly as in notebook
            def categorize_severity(water_level):
                if water_level <= 5:
                    return 'Low'
                elif 5 < water_level <= 15:
                    return 'Medium'
                else:
                    return 'High'

            df['Flood_Severity'] = df['Water Level'].apply(categorize_severity)
            st.subheader("Flood_Severity distribution")
            st.write(df['Flood_Severity'].value_counts())

            # create municipality / barangay dummies (not modifying original approach)
            municipality_dummies = pd.get_dummies(df['Municipality'], prefix='Municipality', dummy_na=False)
            barangay_dummies = pd.get_dummies(df['Barangay'], prefix='Barangay', dummy_na=False)

            # concatenate dummies as in notebook
            df = pd.concat([df, municipality_dummies, barangay_dummies], axis=1)

            # features for severity
            month_dummies = st.session_state.get('month_dummies', pd.get_dummies(df['Month'], prefix='Month'))
            feature_cols_severity = ['No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture'] + list(month_dummies.columns) + list(municipality_dummies.columns) + list(barangay_dummies.columns)

            # check missing columns
            missing_cols = [col for col in feature_cols_severity if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing feature columns before severity modeling (these were expected in the notebook if issues occurred): {missing_cols}")

            # only proceed if features exist
            existing_features = [c for c in feature_cols_severity if c in df.columns]
            X_severity = df[existing_features]
            y_severity = df['Flood_Severity']

            st.subheader("X_severity sample")
            st.dataframe(X_severity.head())

            # Train model (RandomForest) & evaluate as original
            model_severity = RandomForestClassifier(random_state=42)
            # stratify as in notebook if possible
            try:
                X_train_severity, X_test_severity, y_train_severity, y_test_severity = train_test_split(X_severity, y_severity, test_size=0.3, random_state=42, stratify=y_severity)
            except Exception:
                X_train_severity, X_test_severity, y_train_severity, y_test_severity = train_test_split(X_severity, y_severity, test_size=0.3, random_state=42)

            model_severity.fit(X_train_severity, y_train_severity)
            y_pred_severity = model_severity.predict(X_test_severity)
            accuracy_severity = accuracy_score(y_test_severity, y_pred_severity)

            st.subheader("Severity Model Evaluation")
            st.write(f"Accuracy: {accuracy_severity:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test_severity, y_pred_severity))

            st.session_state['model_severity'] = model_severity
            st.session_state['df'] = df
            log("Flood severity model complete.")
        except Exception as e:
            st.error(f"Error in Flood Severity tab: {e}")
            log(f"Severity model error: {e}")

# ---------- TAB 6: Time Series Forecast (SARIMA) ----------
with tabs[5]:
    st.header("Time Series Forecast (SARIMA)")
    st.write("This runs the time-series steps: creating Date index, resampling, stationarity tests, SARIMA fit and predictions. Logic preserved from notebook.")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Upload & clean data first.")
    else:
        try:
            # step: fill and build datetime index like notebook
            # Copy the columns since earlier we may have dropped them; the notebook does fillna and map months
            # But to preserve logic: first ensure Year/Month/Day columns exist (some notebooks earlier dropped them)
            # If they were removed, we cannot reconstruct dates; warn.
            if not all(col in df.columns for col in ['Year', 'Month', 'Day']):
                st.warning("Year/Month/Day columns not present — the notebook later re-created Date from them. If they were dropped earlier, time series can't proceed exactly as in notebook.")
            else:
                # replicate mapping from notebook
                df['Year'] = df['Year'].fillna(method='ffill')
                df['Month'] = df['Month'].fillna(method='ffill')
                df['Day'] = df['Day'].fillna(method='ffill')

                df['Year'] = df['Year'].astype(int)
                df['Day'] = df['Day'].astype(int)

                month_map = {'JANUARY': 1, 'FEBRUARY': 2, 'MARCH': 3, 'APRIL': 4, 'MAY': 5, 'JUNE': 6,
                             'JULY': 7, 'AUGUST': 8, 'SEPTEMBER': 9, 'OCTOBER': 10, 'NOVEMBER': 11, 'DECEMBER': 12, 'Unknown': 1}

                df['Month_Num'] = df['Month'].map(month_map)
                temp_date_df = df[['Year', 'Month_Num', 'Day']].copy()
                temp_date_df.rename(columns={'Month_Num': 'month'}, inplace=True)
                df['Date'] = pd.to_datetime(temp_date_df, errors='coerce')

                df.set_index('Date', inplace=True)

                # drop helper columns as original
                df.drop(columns=['Year', 'Month', 'Day', 'Month_Num'], inplace=True, errors='ignore')

                # resample daily averaging Water Level
                ts_df = df['Water Level'].resample('D').mean()

                st.subheader("Time series (first rows)")
                st.write(ts_df.head())

                # Plot original series
                fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
                ax_ts.plot(ts_df)
                ax_ts.set_title('Daily Average Water Level Over Time')
                ax_ts.set_xlabel('Date')
                ax_ts.set_ylabel('Average Water Level')
                st.pyplot(fig_ts)

                # ADF test on original
                st.subheader("ADF test on original (dropping NaNs)")
                try:
                    adf_result = adfuller(ts_df.dropna())
                    st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                    st.write(f"P-value: {adf_result[1]:.4f}")
                    for key, value in adf_result[4].items():
                        st.write(f"   {key}: {value:.4f}")
                except Exception as e:
                    st.warning(f"ADF test failed: {e}")

                # If non-stationary - fill NaNs and difference as notebook
                ts_df_filled = ts_df.fillna(method='ffill').fillna(method='bfill')
                ts_df_diff = ts_df_filled.diff().dropna()

                st.subheader("Differenced series sample")
                st.write(ts_df_diff.head())

                # ACF / PACF plots (as notebook did)
                fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
                plot_acf(ts_df_diff, lags=40, ax=ax_acf)
                ax_acf.set_title('ACF of Differenced Water Level')
                st.pyplot(fig_acf)

                fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
                plot_pacf(ts_df_diff, lags=40, ax=ax_pacf)
                ax_pacf.set_title('PACF of Differenced Water Level')
                st.pyplot(fig_pacf)

                st.write("Suggested SARIMA param ranges were discussed in the notebook.")

                # Train SARIMA with example params used in notebook
                sarima_order = (1, 1, 1)
                seasonal_order = (1, 0, 1, 7)  # weekly seasonality as example from notebook

                st.subheader("Fitting SARIMA model (example params)")
                try:
                    model_sarima = SARIMAX(ts_df_filled, order=sarima_order, seasonal_order=seasonal_order)
                    results_sarima = model_sarima.fit(disp=False)
                    st.write(results_sarima.summary())

                    # residual diagnostics plots (may be heavy)
                    st.subheader("SARIMA diagnostics (may take time)")
                    fig_diag = results_sarima.plot_diagnostics(figsize=(12, 8))
                    st.pyplot(fig_diag)
                except Exception as e:
                    st.error(f"SARIMA model fitting failed: {e}")
                    results_sarima = None

                # Make predictions example
                if 'results_sarima' in locals() and results_sarima is not None:
                    steps_ahead = st.number_input("Steps ahead to predict (days)", min_value=1, max_value=365, value=30)
                    last_date = ts_df_filled.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps_ahead, freq='D')

                    try:
                        predictions = results_sarima.predict(start=future_dates[0], end=future_dates[-1])
                        st.subheader(f"SARIMA predictions next {steps_ahead} days")
                        st.write(predictions)
                        # plot original, fitted, predictions
                        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
                        ax_pred.plot(ts_df_filled.index, ts_df_filled, label='Original (Filled)')
                        if hasattr(results_sarima, 'fittedvalues') and results_sarima.fittedvalues is not None:
                            ax_pred.plot(results_sarima.fittedvalues.index, results_sarima.fittedvalues, label='Fitted')
                        ax_pred.plot(predictions.index, predictions, label='Predictions')
                        ax_pred.set_title('SARIMA Model Fit and Predictions')
                        ax_pred.set_xlabel('Date')
                        ax_pred.set_ylabel('Average Water Level')
                        ax_pred.legend()
                        st.pyplot(fig_pred)
                    except Exception as e:
                        st.error(f"Prediction step failed: {e}")

                st.session_state['df'] = df
                st.session_state['ts_df'] = ts_df
                st.session_state['ts_df_filled'] = ts_df_filled
                log("SARIMA steps executed (where possible).")
        except Exception as e:
            st.error(f"Error in Time Series tab: {e}")
            log(f"SARIMA error: {e}")

# ---------- TAB 7: Raw Notebook Output (logs) ----------
with tabs[6]:
    st.header("Raw Notebook Output (logs)")
    st.write("This tab collects textual logs from each step (converted from the notebook prints).")
    if log_lines:
        st.text("\n".join(log_lines))
    else:
        st.info("No logs yet. Run steps in previous tabs to generate logs.")

# Footer
st.markdown("---")
st.caption("Converted to Streamlit app — original notebook logic preserved. If something fails, it's usually due to missing columns or mismatched CSV format. Holler and I'll keep it exactly the same while fixing UI niceties.")
