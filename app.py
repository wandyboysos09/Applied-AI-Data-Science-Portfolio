
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Add scikit-learn imports for clustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --- Page Configuration ---
st.set_page_config(
    page_title="Crypto Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Title and Introduction ---
st.title("Cryptocurrency Trading Strategy Analysis")
st.write("Explore the results of our cryptocurrency analysis, including clustering, model performance, and backtesting.")

# --- Load Data ---
# Explicitly define the directory where CSV files are located
DATA_DIR = "."

@st.cache_data # Cache data loading to improve performance
def load_data():
    try:
        file_names = {
            'summary_df': 'backtesting_summary.csv',
            'cleaned_df': 'cleaned_data.csv',
            'weekly_df_with_ma': 'weekly_ma_data.csv',
            'monthly_df_with_ma': 'monthly_ma_data.csv',
            'quarterly_df_with_ma': 'quarterly_ma_data.csv',
            'model_eval_df': 'model_evaluation_results.csv',
            'detailed_trades_df': 'detailed_trades.csv',
            'model_predictions_signals_df': 'model_predictions_signals.csv' # ADDED: Load model_predictions_signals.csv
        }

        loaded_data = {}
        for df_name, file_name in file_names.items():
            full_path = os.path.join(DATA_DIR, file_name)
            st.warning(f"Debug: Attempting to load {file_name} from {full_path}. Exists: {os.path.exists(full_path)}")
            
            # Corrected logic for parse_dates based on file_name
            if file_name == 'detailed_trades.csv':
                loaded_data[df_name] = pd.read_csv(full_path, parse_dates=['buy_date', 'sell_date'])
            elif file_name == 'model_predictions_signals.csv': # ADDED: parse_dates for model_predictions_signals.csv
                loaded_data[df_name] = pd.read_csv(full_path, parse_dates=['Date'])
            elif file_name in ['cleaned_data.csv', 'weekly_ma_data.csv', 'monthly_ma_data.csv', 'quarterly_ma_data.csv']:
                loaded_data[df_name] = pd.read_csv(full_path, parse_dates=['Date'])
            else:
                loaded_data[df_name] = pd.read_csv(full_path)

        return (
            loaded_data['summary_df'],
            loaded_data['cleaned_df'],
            loaded_data['weekly_df_with_ma'],
            loaded_data['monthly_df_with_ma'],
            loaded_data['quarterly_df_with_ma'],
            loaded_data['model_eval_df'],
            loaded_data['detailed_trades_df'],
            loaded_data['model_predictions_signals_df'] # ADDED: Return model_predictions_signals_df
        )
    except FileNotFoundError as e:
        print(f"Error: FileNotFoundError caught in load_data: {e}")
        st.error(f"One or more data files not found. Please ensure all CSV files are in the '{DATA_DIR}' directory.")
        return None, None, None, None, None, None, None, None # MODIFIED: Add None for model_predictions_signals_df
    except Exception as e:
        print(f"Error: An unexpected error occurred during data loading: {e}")
        st.error(f"An unexpected error occurred during data loading: {e}")
        return None, None, None, None, None, None, None, None # MODIFIED: Add None for model_predictions_signals_df

summary_df, cleaned_df, weekly_df_with_ma, monthly_df_with_ma, quarterly_df_with_ma, model_eval_df, detailed_trades_df, model_predictions_signals_df = load_data() # MODIFIED: Assign model_predictions_signals_df

if summary_df is not None:
    # --- Sidebar for Navigation/Filters ---
    st.sidebar.header("Dashboard Navigation")
    section = st.sidebar.radio(
        "Go to",
        ("Overview", "Backtesting Summary", "Detailed Backtesting", "EDA Visualizations", "Correlation Analysis", "Model Performance", "Clustering Results", "Model Predictions & Signals") # ADDED: Model Predictions & Signals
    )

    # --- Overview Section ---
    if section == "Overview":
        st.header("Project Overview")
        st.markdown("""
        This dashboard presents a multi-stage analysis of cryptocurrency market data.
        We started with data acquisition and cleaning, followed by clustering to identify representative assets.
        Various machine learning models were trained for price prediction, and a simple trading strategy was backtested.
        """)
        st.subheader("Representative Cryptocurrencies (from Clustering)")
        representative_symbols = ['DASH-USD', 'BTC-USD', 'ETH-USD', 'BCH-USD'] # Example, adapt as needed
        st.write(representative_symbols)

    # --- Backtesting Summary Section ---
    elif section == "Backtesting Summary":
        st.header("Trading Strategy Backtesting Results")

        st.subheader("Overall Backtesting Performance")
        st.dataframe(summary_df)

        st.subheader("Filter by Cryptocurrency and Interval")
        selected_symbol_bt = st.selectbox("Select Cryptocurrency", ['All'] + sorted(summary_df['Symbol'].unique().tolist()), key='select_symbol_bt')
        selected_interval_bt = st.selectbox("Select Interval", ['All'] + summary_df['Interval'].unique().tolist(), key='select_interval_bt')

        filtered_summary_df = summary_df.copy()
        if selected_symbol_bt != 'All':
            filtered_summary_df = filtered_summary_df[filtered_summary_df['Symbol'] == selected_symbol_bt]
        if selected_interval_bt != 'All':
            filtered_summary_df = filtered_summary_df[filtered_summary_df['Interval'] == selected_interval_bt]

        st.dataframe(filtered_summary_df)

        # Basic visualization of total profit
        if not filtered_summary_df.empty:
            fig_profit = plt.figure(figsize=(10, 5))
            sns.barplot(x='Symbol', y='Total Profit ($)', hue='Interval', data=filtered_summary_df)
            plt.title('Total Profit by Cryptocurrency and Interval')
            plt.xticks(rotation=45)
            st.pyplot(fig_profit)

    # --- Detailed Backtesting Section ---
    elif section == "Detailed Backtesting":
        st.header("Detailed Individual Trade Records")

        if detailed_trades_df is not None and not detailed_trades_df.empty:
            st.subheader("Individual Trades")

            unique_symbols_dt = sorted(detailed_trades_df['symbol'].unique().tolist())
            unique_intervals_dt = sorted(detailed_trades_df['interval'].unique().tolist())

            selected_symbol_dt = st.selectbox("Select Cryptocurrency", ['All'] + unique_symbols_dt, key='select_symbol_dt')
            selected_interval_dt = st.selectbox("Select Interval", ['All'] + unique_intervals_dt, key='select_interval_dt')
            selected_exit_reason = st.selectbox("Select Exit Reason", ['All'] + sorted(detailed_trades_df['exit_reason'].unique().tolist()), key='select_exit_reason')

            filtered_detailed_trades_df = detailed_trades_df.copy()
            if selected_symbol_dt != 'All':
                filtered_detailed_trades_df = filtered_detailed_trades_df[filtered_detailed_trades_df['symbol'] == selected_symbol_dt]
            if selected_interval_dt != 'All':
                filtered_detailed_trades_df = filtered_detailed_trades_df[filtered_detailed_trades_df['interval'] == selected_interval_dt]
            if selected_exit_reason != 'All':
                filtered_detailed_trades_df = filtered_detailed_trades_df[filtered_detailed_trades_df['exit_reason'] == selected_exit_reason]

            st.dataframe(filtered_detailed_trades_df)

            if not filtered_detailed_trades_df.empty:
                st.subheader("Profit/Loss Distribution of Individual Trades")
                fig_pl, ax_pl = plt.subplots(figsize=(10, 5))
                sns.histplot(filtered_detailed_trades_df['profit_loss'], kde=True, bins=50, ax=ax_pl)
                ax_pl.set_title(f"Profit/Loss Distribution for {selected_symbol_dt} ({selected_interval_dt})")
                ax_pl.set_xlabel("Profit/Loss ($)")
                ax_pl.set_ylabel("Frequency")
                st.pyplot(fig_pl)

                st.subheader("Cumulative Returns Over Time")
                # Calculate cumulative returns (equity curve)
                initial_capital = 1000  # Starting capital for the equity curve

                if not filtered_detailed_trades_df.empty:
                    # Prepare data for cumulative returns calculation
                    # Create a series of daily profits, setting profits on sell dates
                    # Ensure all dates from buy to sell are covered
                    min_overall_date = filtered_detailed_trades_df['buy_date'].min()
                    max_overall_date = filtered_detailed_trades_df['sell_date'].max()

                    if pd.notna(min_overall_date) and pd.notna(max_overall_date):
                        full_date_range = pd.date_range(start=min_overall_date, end=max_overall_date, freq='D')
                        daily_profits_series = pd.Series(0.0, index=full_date_range)

                        # Sum profits/losses for each sell date
                        # Group by date only before summing to avoid issues with different times on the same day
                        trade_profits_by_date = filtered_detailed_trades_df.groupby(filtered_detailed_trades_df['sell_date'].dt.date)['profit_loss'].sum()
                        trade_profits_by_date.index = pd.to_datetime(trade_profits_by_date.index)
                        
                        daily_profits_series.loc[trade_profits_by_date.index] += trade_profits_by_date

                        # Calculate cumulative capital
                        cumulative_capital_series = initial_capital + daily_profits_series.cumsum()
                        
                        # Add initial capital point if not already included by first trade date
                        if daily_profits_series.index.min() > min_overall_date:
                            initial_capital_point = pd.Series(initial_capital, index=[min_overall_date])
                            cumulative_capital_series = pd.concat([initial_capital_point, cumulative_capital_series]).sort_index()
                            cumulative_capital_series = initial_capital + daily_profits_series.cumsum() # Recalculate if initial point was added to correct start
                            # If the first date of daily_profits_series is later than min_overall_date, fill in the gap with initial_capital
                            if min_overall_date < daily_profits_series.index.min():
                                prev_dates = pd.date_range(start=min_overall_date, end=daily_profits_series.index.min() - pd.Timedelta(days=1), freq='D')
                                initial_segment = pd.Series(initial_capital, index=prev_dates)
                                cumulative_capital_series = pd.concat([initial_segment, cumulative_capital_series])
                                cumulative_capital_series = cumulative_capital_series.sort_index().ffill().fillna(initial_capital) # fill forward any gaps

                        equity_curve_df = pd.DataFrame({
                            'Date': cumulative_capital_series.index,
                            'Cumulative Capital': cumulative_capital_series.values
                        })

                        fig_equity, ax_equity = plt.subplots(figsize=(12, 6))
                        sns.lineplot(x='Date', y='Cumulative Capital', data=equity_curve_df, ax=ax_equity)
                        ax_equity.set_title(f'Cumulative Capital for {selected_symbol_dt} ({selected_interval_dt})')
                        ax_equity.set_xlabel('Date')
                        ax_equity.set_ylabel('Capital ($)')
                        ax_equity.grid(True)
                        st.pyplot(fig_equity)
                    else:
                        st.info("No data to calculate a meaningful cumulative return curve for the selected filters.")
                else:
                    st.info("No trades to calculate cumulative returns.")


                st.subheader("Buy/Sell Prices Over Time")
                # For plotting, we need the original cleaned_df or similar to show price context
                # Let's use the appropriate df (daily, weekly, monthly, quarterly) based on selected_interval_dt
                if selected_interval_dt == 'Weekly': source_df = weekly_df_with_ma
                elif selected_interval_dt == 'Monthly': source_df = monthly_df_with_ma
                elif selected_interval_dt == 'Quarterly': source_df = quarterly_df_with_ma
                else: source_df = cleaned_df # Default to daily if 'All' or unknown interval

                if source_df is not None and not source_df.empty and selected_symbol_dt != 'All':
                    plot_df = source_df[source_df['symbol'] == selected_symbol_dt].copy()
                    if not plot_df.empty:
                        fig_trades, ax_trades = plt.subplots(figsize=(12, 6))
                        sns.lineplot(x='Date', y='Close', data=plot_df, label='Close Price', ax=ax_trades, color='blue')

                        # Plot buy signals
                        buy_trades = filtered_detailed_trades_df[filtered_detailed_trades_df['exit_reason'].isin(['Target Profit', 'Stop Loss', 'End of Backtest'])] # Filter for actual buys
                        ax_trades.scatter(buy_trades['buy_date'], buy_trades['buy_price'],
                                        marker='^', color='green', s=100, label='Buy Signal', zorder=5)
                        # Plot sell signals
                        ax_trades.scatter(buy_trades['sell_date'], buy_trades['sell_price'],
                                        marker='v', color='red', s=100, label='Sell Signal', zorder=5)
                        
                        ax_trades.set_title(f'Buy/Sell Signals for {selected_symbol_dt} ({selected_interval_dt})')
                        ax_trades.set_xlabel('Date')
                        ax_trades.set_ylabel('Price (USD)')
                        ax_trades.legend()
                        ax_trades.grid(True)
                        st.pyplot(fig_trades)
                    else:
                         st.warning(f"No price data available for plotting {selected_symbol_dt} in {selected_interval_dt} for trade signals.")
                else:
                    st.warning("Cannot plot trade signals without a specific cryptocurrency selected or if base data is missing.")

            else:
                st.info("No trades to display for the selected filters.")
        else:
            st.warning("Detailed trades data not loaded or empty. Cannot display individual trades.")

    # --- EDA Visualizations Section ---
    elif section == "EDA Visualizations":
        st.header("Exploratory Data Analysis")

        if cleaned_df is not None:
            representative_symbols = ['DASH-USD', 'BTC-USD', 'ETH-USD', 'BCH-USD'] # Example, adapt as needed
            selected_crypto_eda = st.selectbox("Select Cryptocurrency for EDA", sorted(representative_symbols))

            # Ensure 'Date' is datetime for filtering
            daily_crypto_df_eda = cleaned_df[cleaned_df['symbol'] == selected_crypto_eda].sort_values('Date').copy()
            daily_crypto_df_eda['Date'] = pd.to_datetime(daily_crypto_df_eda['Date'])

            if not daily_crypto_df_eda.empty:
                st.subheader(f"{selected_crypto_eda} Daily Close Price Over Time")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=daily_crypto_df_eda, x='Date', y='Close', ax=ax1)
                ax1.set_title(f'{selected_crypto_eda} Daily Close Price Over Time')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Close Price (USD)')
                ax1.grid(True)
                st.pyplot(fig1)

                st.subheader(f"Distribution of {selected_crypto_eda} Daily Close Prices")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.histplot(daily_crypto_df_eda['Close'], kde=True, bins=50, ax=ax2)
                ax2.set_title(f'Distribution of {selected_crypto_eda} Daily Close Prices')
                ax2.set_xlabel('Close Price (USD)')
                ax2.set_ylabel('Frequency')
                st.pyplot(fig2)

                # Added Volume Distribution Plot
                if 'Volume' in daily_crypto_df_eda.columns:
                    st.subheader(f"Distribution of {selected_crypto_eda} Daily Trading Volume")
                    fig_vol, ax_vol = plt.subplots(figsize=(10, 6))
                    sns.histplot(daily_crypto_df_eda['Volume'], kde=True, bins=50, ax=ax_vol)
                    ax_vol.set_title(f'Distribution of {selected_crypto_eda} Daily Trading Volume')
                    ax_vol.set_xlabel('Trading Volume')
                    ax_vol.set_ylabel('Frequency')
                    st.pyplot(fig_vol)
                else:
                    st.warning(f"Volume data not available for {selected_crypto_eda}.")

            else:
                st.warning(f"No daily data available for {selected_crypto_eda} for EDA.")
        else:
            st.warning("Daily data not loaded. Cannot perform EDA.")

    # --- Correlation Analysis Section ---
    elif section == "Correlation Analysis":
        st.header("Correlation Analysis")
        st.write("Examine the correlation between representative cryptocurrencies across different time intervals.")

        selected_interval_corr = st.selectbox("Select Interval for Correlation", ['Weekly', 'Monthly', 'Quarterly'])

        df_for_corr = None
        if selected_interval_corr == 'Weekly':
            df_for_corr = weekly_df_with_ma
        elif selected_interval_corr == 'Monthly':
            df_for_corr = monthly_df_with_ma
        elif selected_interval_corr == 'Quarterly':
            df_for_corr = quarterly_df_with_ma

        if df_for_corr is not None and not df_for_corr.empty:
            representative_symbols = ['DASH-USD', 'BTC-USD', 'ETH-USD', 'BCH-USD'] # Example, adapt as needed
            df_filtered_corr = df_for_corr[df_for_corr['symbol'].isin(representative_symbols)].copy()
            df_filtered_corr['Date'] = pd.to_datetime(df_filtered_corr['Date'])

            if not df_filtered_corr.empty:
                correlation_pivot = df_filtered_corr.pivot(index='Date', columns='symbol', values='Close')
                correlation_pivot.dropna(axis=1, how='all', inplace=True)
                correlation_pivot.fillna(method='ffill', axis=1, inplace=True) # Fill forward along columns
                correlation_pivot.fillna(method='bfill', axis=1, inplace=True) # Fill backward along columns
                correlation_pivot.fillna(correlation_pivot.mean(), inplace=True)

                if correlation_pivot.shape[1] >= 2:
                    correlation_matrix = correlation_pivot.corr()

                    st.subheader(f"Correlation Matrix ({selected_interval_corr})")
                    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
                    st.pyplot(fig_corr)

                    # FIX: Explicitly rename MultiIndex levels to prevent collision during reset_index
                    stacked_corr = correlation_matrix.stack()
                    corr_pairs = stacked_corr.rename_axis(index=['Crypto1', 'Crypto2']).reset_index(name='Correlation')

                    # Remove self-correlations and duplicate pairs
                    corr_pairs = corr_pairs[corr_pairs['Crypto1'] != corr_pairs['Crypto2']]
                    corr_pairs['Pair'] = corr_pairs.apply(lambda x: tuple(sorted((x['Crypto1'], x['Crypto2']))), axis=1)
                    corr_pairs.drop_duplicates(subset='Pair', inplace=True)

                    st.subheader(f"Top Most Positively Correlated Pairs ({selected_interval_corr})")
                    st.dataframe(corr_pairs.sort_values(by='Correlation', ascending=False).head(4).drop(columns='Pair'))

                    st.subheader(f"Top Most Negatively Correlated Pairs ({selected_interval_corr})")
                    st.dataframe(corr_pairs.sort_values(by='Correlation', ascending=True).head(4).drop(columns='Pair'))
                else:
                    st.warning(f"Not enough data points or cryptocurrencies (at least 2) for correlation in {selected_interval_corr} interval.")
            else:
                st.warning(f"No data for representative cryptocurrencies in {selected_interval_corr} interval for correlation analysis.")
        else:
            st.warning(f"Data for {selected_interval_corr} interval not loaded. Cannot perform correlation analysis.")

    # --- Model Performance Section ---
    elif section == "Model Performance":
        st.header("Machine Learning Model Performance")
        st.write("Compare the RMSE and MAE of different models across cryptocurrencies and intervals.")

        if model_eval_df is not None and not model_eval_df.empty:
            st.subheader("Overall Model Evaluation Results")
            st.dataframe(model_eval_df)

            st.subheader("Filter by Cryptocurrency and Interval")

            # Dynamically get unique symbols and intervals from model_eval_df
            unique_symbols = sorted(model_eval_df['Symbol'].unique().tolist())
            unique_intervals = sorted(model_eval_df['Interval'].unique().tolist())

            selected_symbol_model = st.selectbox("Select Cryptocurrency", ['All'] + unique_symbols, key='select_symbol_model')
            selected_interval_model = st.selectbox("Select Interval", ['All'] + unique_intervals, key='select_interval_model')

            filtered_model_eval_df = model_eval_df.copy()
            if selected_symbol_model != 'All':
                filtered_model_eval_df = filtered_model_eval_df[filtered_model_eval_df['Symbol'] == selected_symbol_model]
            if selected_interval_model != 'All':
                filtered_model_eval_df = filtered_model_eval_df[filtered_model_eval_df['Interval'] == selected_interval_model]

            st.dataframe(filtered_model_eval_df)

            # Visualization of Comparative Model Performance (e.g., RMSE)
            if not filtered_model_eval_df.empty:
                metric_choice = st.radio("Select Metric to Visualize", ['RMSE', 'MAE'], key='metric_choice')

                # Prepare data for plotting
                plot_df = filtered_model_eval_df.melt(id_vars=['Symbol', 'Interval'],
                                                       value_vars=[col for col in filtered_model_eval_df.columns if metric_choice in col],
                                                       var_name='Model_Metric',
                                                       value_name=metric_choice)

                # Clean up Model_Metric names for better readability
                plot_df['Model'] = plot_df['Model_Metric'].apply(lambda x: x.replace(f'_{metric_choice}', ''))

                # Remove rows where the metric value is NaN (for models that might not have run)
                plot_df.dropna(subset=[metric_choice], inplace=True)

                if not plot_df.empty:
                    fig_model_perf = plt.figure(figsize=(12, 6))
                    sns.barplot(data=plot_df, x='Model', y=metric_choice, hue='Symbol', palette='viridis')
                    plt.title(f'Comparative Model {metric_choice} for Selected Cryptocurrencies and Intervals')
                    plt.xlabel('Model')
                    plt.ylabel(f'{metric_choice}')
                    plt.xticks(rotation=45)
                    plt.legend(title='Cryptocurrency')
                    st.pyplot(fig_model_perf)
                else:
                    st.warning(f"No data to plot for the selected filters and metric ({metric_choice}).")
            else:
                st.info("Filter the table above to see comparative model performance visualization.")

        else:
            st.warning("Model evaluation results not loaded or empty. Cannot display model performance.")

    # --- Clustering Results Section ---
    elif section == "Clustering Results":
        st.header("Cryptocurrency Clustering Results")
        st.write("Visualizing PCA components and K-Means clusters for identifying representative assets.")

        if cleaned_df is not None and not cleaned_df.empty:
            # Ensure 'Date' is datetime for filtering
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

            # --- Replicate Clustering Logic ---
            latest_date = cleaned_df['Date'].max()
            one_year_ago = latest_date - pd.Timedelta(days=365)
            recent_data = cleaned_df[cleaned_df['Date'] > one_year_ago].copy()

            if not recent_data.empty:
                pivot_df = recent_data.pivot(index='symbol', columns='Date', values='Close')
                pivot_df.dropna(thresh=int(pivot_df.shape[1] * 0.8), axis=0, inplace=True)
                pivot_df.fillna(method='ffill', axis=1, inplace=True)
                pivot_df.fillna(method='bfill', axis=1, inplace=True)
                pivot_df.fillna(pivot_df.mean(), inplace=True)
                pivot_df.dropna(axis=0, inplace=True)

                if not pivot_df.empty and pivot_df.shape[0] >= 4: # Need at least k=4 symbols for clustering
                    scaler = StandardScaler()
                    scaled_pivot_data = scaler.fit_transform(pivot_df)

                    n_components = 2
                    pca = PCA(n_components=n_components)
                    pca_components = pca.fit_transform(scaled_pivot_data)

                    pca_df = pd.DataFrame(data=pca_components, columns=[f'PC{i+1}' for i in range(n_components)])
                    pca_df['symbol'] = pivot_df.index

                    n_clusters = 4
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    pca_df['Cluster'] = kmeans.fit_predict(pca_components)

                    # --- Plotting Clustering Results ---
                    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 8))
                    sns.scatterplot(
                        x='PC1', y='PC2', hue='Cluster', data=pca_df,
                        palette='viridis', s=100, alpha=0.8, ax=ax_cluster
                    )
                    # Annotate representative cryptos (assuming these are fixed for display)
                    representative_symbols_display = ['DASH-USD', 'BTC-USD', 'ETH-USD', 'BCH-USD'] # Hardcode for consistent display
                    for i, row in pca_df.iterrows():
                        if row['symbol'] in representative_symbols_display:
                            ax_cluster.text(row['PC1'] + 0.1, row['PC2'] + 0.1, row['symbol'],
                                            fontsize=9, color='red', weight='bold')

                    ax_cluster.set_title('PCA of Cryptocurrency Close Prices (Last Year) with K-Means Clusters')
                    ax_cluster.set_xlabel('Principal Component 1')
                    ax_cluster.set_ylabel('Principal Component 2')
                    st.pyplot(fig_cluster)

                    st.subheader("Representative Cryptocurrencies")
                    st.write(representative_symbols_display)
                    st.dataframe(pca_df[['symbol', 'Cluster', 'PC1', 'PC2']])

                else:
                    st.warning("Not enough data or symbols for clustering after filtering for the last year. Need at least 4 symbols.")
            else:
                st.warning("No recent data available for clustering (last year).")
        else:
            st.warning("Daily data not loaded. Cannot perform clustering.")
    
    # --- Model Predictions & Signals Section ---
    elif section == "Model Predictions & Signals":
        st.header("Model Predictions and Generated Signals")
        st.write("Visualize the best model's predictions against actual prices and the generated buy/hold signals.")

        if model_predictions_signals_df is not None and not model_predictions_signals_df.empty:
            unique_symbols_mps = sorted(model_predictions_signals_df['symbol'].unique().tolist())
            unique_intervals_mps = sorted(model_predictions_signals_df['interval'].unique().tolist())

            selected_symbol_mps = st.selectbox("Select Cryptocurrency", unique_symbols_mps, key='select_symbol_mps')
            selected_interval_mps = st.selectbox("Select Interval", unique_intervals_mps, key='select_interval_mps')

            filtered_mps_df = model_predictions_signals_df[
                (model_predictions_signals_df['symbol'] == selected_symbol_mps) &
                (model_predictions_signals_df['interval'] == selected_interval_mps)
            ].copy()

            st.dataframe(filtered_mps_df)

            if not filtered_mps_df.empty:
                st.subheader(f"Predictions and Signals for {selected_symbol_mps} ({selected_interval_mps})")
                fig_mps, ax_mps = plt.subplots(figsize=(12, 6))

                # Plot actual Close price
                sns.lineplot(x='Date', y='Close', data=filtered_mps_df, label='Actual Close Price', color='blue', ax=ax_mps)
                # Plot Predicted Price
                sns.lineplot(x='Date', y='Predicted Price', data=filtered_mps_df, label='Predicted Price', color='orange', linestyle='--', ax=ax_mps)

                # Plot Buy signals
                buy_signals = filtered_mps_df[filtered_mps_df['Signal'] == 'Buy']
                ax_mps.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)

                ax_mps.set_title(f'Model Predictions and Signals for {selected_symbol_mps} ({selected_interval_mps})')
                ax_mps.set_xlabel('Date')
                ax_mps.set_ylabel('Price (USD)')
                ax_mps.legend()
                ax_mps.grid(True)
                st.pyplot(fig_mps)
            else:
                st.info("No predictions or signals to display for the selected filters.")
        else:
            st.warning("Model predictions and signals data not loaded or empty. Cannot display.")
