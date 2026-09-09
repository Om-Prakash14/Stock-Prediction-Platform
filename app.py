import streamlit as st
import os
import subprocess
import re
import glob
from datetime import datetime
import pandas as pd
from PIL import Image

# Force matplotlib to run headless to prevent GUI blocking on plt.show()
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="Stock Prediction Deep Learning Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Prediction Deep Learning Portal")
st.markdown("An interactive web interface to train LSTM neural networks and forecast stock market trends using TensorFlow.")

# Sidebar - Settings
st.sidebar.header("Virtual Environment Info")
st.sidebar.code("Python: 3.12 (venv)\nTensorFlow: 2.18.1")

# Tabs
tab1, tab2 = st.tabs(["🚀 Train New Model", "🔮 Run Inference & Forecast"])

# Helper function to find trained run directories
def get_trained_runs():
    # Scan for directories starting with a ticker pattern or having model_config.json
    folders = []
    for entry in os.scandir('.'):
        if entry.is_dir() and not entry.name.startswith('.') and entry.name != 'venv' and entry.name != '__pycache__':
            # Check if model_config.json exists inside the folder
            if os.path.exists(os.path.join(entry.path, 'model_config.json')):
                folders.append(entry.name)
    return sorted(folders)

with tab1:
    st.header("1. Model Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Stock Ticker Symbol", value="^FTSE", help="Yahoo Finance symbol (e.g. AAPL, GOOG, TSLA, ^FTSE)")
        start_date = st.date_input("Start Date", value=datetime.strptime("2017-11-01", "%Y-%m-%d"))
        validation_date = st.date_input("Validation/Split Date", value=datetime.strptime("2021-09-01", "%Y-%m-%d"))
        
    with col2:
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=10, help="Number of training epochs (Lower for faster tests)")
        batch_size = st.number_input("Batch Size", min_value=1, max_value=500, value=10)
        time_steps = st.number_input("Time Steps", min_value=1, max_value=100, value=3)
        
    with col3:
        model_version = st.selectbox("Model Version", ["v1", "v2", "v3", "v4", "v5", "v6", "v7"], index=6, help="v7 is the latest version using separate direction/magnitude models")
        forecast_horizon = st.number_input("Forecast Horizon", min_value=1, max_value=100, value=10)
        trend_window = st.number_input("Trend Window", min_value=1, max_value=200, value=60)
        
    # v3, v5, v7 are delta models, v6 is trend-residual model. These cannot be combined with Log Returns.
    is_delta_or_residual = model_version in ["v3", "v5", "v6", "v7"]
    use_returns = st.checkbox(
        "Use Log Returns", 
        value=False if is_delta_or_residual else True, 
        disabled=is_delta_or_residual,
        help="Transform price series to log returns before training (Disabled for delta/residual models like v3, v5, v6, v7)"
    )

    if st.button("🔥 Start Training Model", use_container_width=True):
        st.subheader("Training Console Logs")
        
        # Prepare command
        cmd = [
            "C:\\Users\\OM PRAKSH NANDA\\v\\Scripts\\python.exe",
            "stock_prediction_deep_learning.py",
            "-ticker", ticker,
            "-start_date", start_date.strftime("%Y-%m-%d"),
            "-validation_date", validation_date.strftime("%Y-%m-%d"),
            "-epochs", str(epochs),
            "-batch_size", str(batch_size),
            "-time_steps", str(time_steps),
            "-model_version", model_version,
            "-forecast_horizon", str(forecast_horizon),
            "-trend_window", str(trend_window),
            "-use_returns", "true" if use_returns else "false"
        ]
        
        # Execute training with live log streaming
        log_container = st.empty()
        log_text = ""
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read stdout line by line and update UI
            for line in iter(process.stdout.readline, ''):
                log_text += line
                log_container.code(log_text)
                
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                st.success("🎉 Training completed successfully!")
                
                # Find output folder
                # The folder name format in stock_prediction_deep_learning.py is Ticker_Date_Token
                # Let's search for the folder matching this pattern that was just created
                pattern = f"{ticker}_*"
                matching_dirs = glob.glob(pattern)
                
                if matching_dirs:
                    # Get the most recently created matching directory
                    target_dir = max(matching_dirs, key=os.path.getmtime)
                    st.info(f"Visualizing results from: `{target_dir}`")
                    
                    # Look for and display generated plots
                    plots_to_show = ["loss.png", "MSE.png", "DataHistogram.png"]
                    
                    # Display standard training plots
                    for plot_name in plots_to_show:
                        plot_path = os.path.join(target_dir, plot_name)
                        if os.path.exists(plot_path):
                            st.image(Image.open(plot_path), caption=f"{plot_name} ({target_dir})")
                            
                    # Display the final prediction plot (e.g. predictions.png or a custom name containing prediction)
                    # Let's list files in the directory and find any png file
                    for file in os.listdir(target_dir):
                        if file.endswith("prediction.png") or file.endswith("predictions.png"):
                            st.image(Image.open(os.path.join(target_dir, file)), caption=f"Prediction Fit: {file}")
            else:
                st.error(f"Training failed with exit code: {return_code}")
                
        except Exception as e:
            st.error(f"Error launching training subprocess: {e}")

with tab2:
    st.header("2. Run Inference and Future Forecast")
    
    # Refresh runs
    trained_runs = get_trained_runs()
    
    if not trained_runs:
        st.warning("⚠️ No trained model runs detected yet. Please train a model in Tab 1 first.")
    else:
        selected_run = st.selectbox("Select Trained Run Folder", trained_runs)
        
        # Load the configuration of the selected model run
        config_path = os.path.join(selected_run, "model_config.json")
        model_config = {}
        if os.path.exists(config_path):
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            st.info(f"Model Configuration: {model_config}")
            
        col1_inf, col2_inf = st.columns(2)
        
        with col1_inf:
            forecast_days = st.number_input("Forecast Days", min_value=1, max_value=365, value=30, help="Number of future days to forecast")
            use_business_days = st.checkbox("Use Business Days only", value=True)
            plot_history_days = st.number_input("Plot History Days", min_value=10, max_value=1000, value=200, help="Days of past stock history to show in the plot")
            
        with col2_inf:
            stochastic_paths = st.number_input("Stochastic Simulation Paths", min_value=0, max_value=500, value=50, help="Number of Monte Carlo simulations to generate P10-P90 range")
            stochastic_sigma_mult = st.slider("Stochastic Sigma Multiplier", min_value=0.0, max_value=3.0, value=0.6, step=0.1)
            stochastic_lookback = st.number_input("Stochastic Noise Est. Lookback", min_value=10, max_value=500, value=120)
            
        if st.button("🚀 Run Future Predictions", use_container_width=True):
            st.write("Running predictions...")
            
            # Extract variables from run folder name or config
            ticker_from_folder = model_config.get("ticker", selected_run.split("_")[0])
            start_date_val = pd.to_datetime(model_config.get("start_date", "2017-11-01"))
            validation_date_val = pd.to_datetime(model_config.get("validation_date", "2021-09-01"))
            time_steps_val = int(model_config.get("time_steps", 3))
            use_returns_val = bool(model_config.get("use_returns", True))
            use_deltas_val = bool(model_config.get("use_deltas", True))
            
            # Extract token from the selected folder name
            # Format: Ticker_Date_Token
            token_parts = selected_run.split("_")
            token_val = token_parts[-1] if len(token_parts) > 1 else "default"
            
            # Import and execute InferenceRunner
            try:
                from stock_prediction_deep_learning_inference import InferenceRunner
                
                runner = InferenceRunner(
                    run_folder=selected_run,
                    ticker=ticker_from_folder,
                    start_date=start_date_val,
                    validation_date=validation_date_val,
                    github_url="https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/",
                    epochs=int(model_config.get("epochs", 100)),
                    time_steps=time_steps_val,
                    token=token_val,
                    batch_size=int(model_config.get("batch_size", 10)),
                    forecast_days=int(forecast_days),
                    use_business_days=use_business_days,
                    plot_history_days=int(plot_history_days),
                    use_returns=use_returns_val,
                    use_deltas=use_deltas_val,
                    clip_negative=True,
                    blend_alpha=0.6,
                    direction_threshold=0.55,
                    mag_clip_pct=90,
                    stochastic_paths=int(stochastic_paths),
                    stochastic_seed=42,
                    stochastic_sigma_mult=stochastic_sigma_mult,
                    stochastic_lookback=int(stochastic_lookback)
                )
                
                # Execute in process (matplotlib is headless, so it won't block)
                with st.spinner("Calculating forecast paths and plotting..."):
                    runner.run()
                
                st.success("🎉 Future forecast calculated!")
                
                # Display output forecast graph
                forecast_image_name = f"{ticker_from_folder}_future_forecast.png"
                forecast_image_path = os.path.join(selected_run, forecast_image_name)
                
                if os.path.exists(forecast_image_path):
                    st.image(Image.open(forecast_image_path), caption="Future Forecast Chart")
                else:
                    st.warning("Forecast chart image could not be generated. Checking for forecast csv...")
                    
                # Look for generated prediction values
                csv_path = os.path.join(selected_run, "predictions.csv")
                if os.path.exists(csv_path):
                    df_preds = pd.read_csv(csv_path)
                    st.subheader("Historical Validation Predictions")
                    st.dataframe(df_preds.tail(20), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Inference execution failed: {e}")
                st.exception(e)
