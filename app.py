# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from datetime import datetime
now = datetime.now()
date_str = f"{now.year}年{now.month}月{now.day}日"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping as KerasEarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

# ===== SHAP関連ライブラリ =====
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Yu Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
logger = logging.getLogger(__name__)

# ===== XGBoost =====
from xgboost import XGBRegressor

# ===== PyTorch (Hybrid Transformer-LSTM用) =====
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===== stable-baselines3 (強化学習) =====
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ===== PyWavelets =====
import pywt

# ===== ARCHライブラリ（GARCH用）=====
from arch import arch_model

# ===== ロギング設定 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== シード固定 =====
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

# sklearn の Ridge 回帰
from sklearn.linear_model import Ridge, LinearRegression

# -----------------------------------------------------------
# クリップ関数（極端な値を制限する）
def clip_extreme_values(arr, threshold=1e8):
    return np.clip(arr, -threshold, threshold)

# -----------------------------------------------------------
# IQR方式の外れ値クリッピング関数
def clip_outliers_iqr(df: pd.DataFrame, columns, factor=1.5):
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

# -----------------------------------------------------------
# 1) Hybrid Transformer-LSTM モデル定義
class HybridTransformerLSTM(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128, nhead=8, num_layers=2):
        super(HybridTransformerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        trans_out = self.transformer(lstm_out)
        out = self.fc(trans_out[:, -1, :])
        return out

# -----------------------------------------------------------
# 2) 損失関数 (MSE + MAE)
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    def forward(self, pred, target):
        loss = self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.mae(pred, target)
        return loss

# -----------------------------------------------------------
# 3) XGBoost学習関数
def train_xgboost(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse"
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )
    return model

# -----------------------------------------------------------
# 4) Ridge学習関数
def train_ridge(X_train, y_train, X_val, y_val):
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

# -----------------------------------------------------------
# メインアプリ（Tkinter 部分を Streamlit に改修）
class AdvancedStockPredictorApp:
    def __init__(self):
        self.days = 730
        self.input_window = 30
        self.epochs = 50
        self.future_days = 1

        self.hybrid_model = None
        self.xgb_model = None
        self.ridge_model = None

        self.X_test_seq = None
        self.y_test_seq = None
        self.range_market_flags_test = None

        self.cf_hybrid = 1.0
        self.cf_xgb = 1.0
        self.cf_ridge = 1.0

        self.error_hybrid = None
        self.error_xgb = None
        self.error_ridge = None

        # センチメント用の最適補正係数
        self.sentiment_coeff = 0.0
        self.composite_sentiment = 0.0

        # ===== 残差補正用 =====
        self.residual_model = None  # 線形回帰などを格納
        self.features_for_correction = []  # 残差補正に使う特徴量
        self.residual_scaler = None  # 残差補正用のスケーラー
        # 補正係数算出に利用する各種指標（MAE, MBE, Directional Accuracy）
        self.mae_val = None
        self.mbe_val = None
        self.directional_accuracy = None

    # -----------------------------------------------------------
    # Streamlit を用いたメイン UI の実装
    def run(self):
        st.title("AI_LSTM with Deep Learning")
        st.write(f"実行日: {date_str}")
        ticker_input = st.sidebar.text_input("銘柄コード (4桁の数字)", value="")
        if st.sidebar.button("予測実行"):
            if not (ticker_input.isdigit() and len(ticker_input) == 4):
                st.error("銘柄コードは4桁の数字のみを入力してください。")
                return
            self.symbol = ticker_input + ".T"
            st.write("データを取得中...")
            try:
                raw_data = self.fetch_stock_data()
                processed_data = self.process_data(raw_data)
            except Exception as e:
                st.error(f"データ取得または前処理エラー: {e}")
                return
            st.write("データ前処理完了。モデル学習と予測を開始します...")
            avg_preds_base, avg_f5_base, avg_fp_base, avg_score_base, std_dev_base = self.multiple_run_prediction(processed_data, is_adjustment=False)
            try:
                backtest_mape_base, avg_pos_base, avg_neg_base = self.run_backtest(processed_data)
            except Exception as e:
                st.error(f"バックテストエラー: {e}")
                return
            current_price = float(processed_data["Close_orig"].iloc[-1])
            self.sentiment_coeff = self.optimize_sentiment_coefficient()
            self.composite_sentiment = self.fetch_composite_sentiment()
            if self.composite_sentiment > 0:
                avg_fp_base *= (1 + self.sentiment_coeff)
                avg_f5_base = avg_f5_base * (1 + self.sentiment_coeff)
            elif self.composite_sentiment < 0:
                avg_fp_base *= (1 - self.sentiment_coeff)
                avg_f5_base = avg_f5_base * (1 - self.sentiment_coeff)
            if backtest_mape_base < 10.0:
                signal = self.generate_single_signal(predicted_price=avg_fp_base, current_price=current_price, confidence_score=avg_score_base, threshold=60)
                st.write("MAPEが低いため補正は不要です。")
                fig = self.update_chart(processed_data, avg_preds_base, avg_f5_base)
                st.pyplot(fig)
                result_text = (f"銘柄名: {self.get_company_name()}\n"
                               f"最新株価: {current_price:.0f}円\n"
                               f"翌営業日予測: {avg_fp_base:.0f}円\n"
                               f"5日後予測: {avg_f5_base[-1]:.0f}円\n"
                               f"信頼度スコア: {avg_score_base:.1f} / 100\n"
                               f"MAPE: ±{backtest_mape_base:.1f}%\n"
                               f"シグナル: {signal}\n"
                               f"実行日: {date_str}")
                st.text(result_text)
            else:
                st.write(f"MAPE {backtest_mape_base:.1f}% → 残差補正試行")
                original_df = processed_data.copy()
                base_mape = backtest_mape_base
                try:
                    st.write("残差補正モデルを学習中...")
                    self.train_residual_model(processed_data)
                    avg_fp_corr_resid, avg_f5_corr_resid, mape_resid = self.apply_residual_correction(processed_data, avg_fp_base, avg_f5_base)
                    if mape_resid < base_mape:
                        pre_fp = avg_fp_base
                        pre_f5 = avg_f5_base
                        pre_score = avg_score_base
                        pre_mape = base_mape
                        pre_preds = avg_preds_base
                        avg_fp_base = avg_fp_corr_resid
                        avg_f5_base = avg_f5_corr_resid
                        fig = self.update_chart_with_preline(original_df, pre_preds, pre_f5, avg_preds_base, avg_f5_base)
                        st.pyplot(fig)
                        signal = self.generate_single_signal(predicted_price=avg_fp_base, current_price=current_price, confidence_score=pre_score, threshold=60)
                        result_text = (f"MAPE補正後\n\n"
                                       f"銘柄名: {self.get_company_name()}\n"
                                       f"最新株価: {current_price:.0f}円\n"
                                       f"翌営業日予測: {pre_fp:.0f}円 → {avg_fp_base:.0f}円\n"
                                       f"5日後予測: {pre_f5[-1]:.0f}円 → {avg_f5_base[-1]:.0f}円\n"
                                       f"信頼度スコア: {pre_score:.1f}/100 → {avg_score_base:.1f} / 100\n"
                                       f"MAPE: {pre_mape:.1f}% → {mape_resid:.1f}%\n"
                                       f"シグナル: {signal}\n"
                                       f"実行日: {date_str}")
                        st.text(result_text)
                    else:
                        st.write(f"残差補正は効果なし → Reverted (Baseline MAPE: {base_mape:.1f}%)")
                        signal = self.generate_single_signal(predicted_price=avg_fp_base, current_price=current_price, confidence_score=avg_score_base, threshold=60)
                        fig = self.update_chart(original_df, avg_preds_base, avg_f5_base)
                        st.pyplot(fig)
                        result_text = (f"銘柄名: {self.get_company_name()}\n"
                                       f"最新株価: {current_price:.0f}円\n"
                                       f"翌営業日予測: {avg_fp_base:.0f}円\n"
                                       f"5日後予測: {avg_f5_base[-1]:.0f}円\n"
                                       f"信頼度スコア: {avg_score_base:.1f} / 100\n"
                                       f"MAPE Reverted: {base_mape:.1f}%\n"
                                       f"シグナル: {signal}\n"
                                       f"実行日: {date_str}")
                        st.text(result_text)
                except Exception as e:
                    logger.error(f"残差補正失敗: {e}")
                    signal = self.generate_single_signal(predicted_price=avg_fp_base, current_price=current_price, confidence_score=avg_score_base, threshold=60)
                    fig = self.update_chart(original_df, avg_preds_base, avg_f5_base)
                    st.pyplot(fig)
                    result_text = (f"銘柄名: {self.get_company_name()}\n"
                                   f"最新株価: {current_price:.0f}円\n"
                                   f"翌営業日予測: {avg_fp_base:.0f}円\n"
                                   f"5日後予測: {avg_f5_base[-1]:.0f}円\n"
                                   f"補正エラー (MAPE元値: {base_mape:.1f}%)\n"
                                   f"シグナル: {signal}\n"
                                   f"実行日: {date_str}")
                    st.text(result_text)

            try:
                hybrid_model, xgb_model, ridge_model, X_test_seq, y_test_seq, range_flags_test = self.build_and_train_model(processed_data)
                last_seq = X_test_seq[-1]
                last_range_flag = range_flags_test[-1] if len(range_flags_test) > 0 else 0
                next_day_pred = self.predict_future_price(hybrid_model, xgb_model, ridge_model, last_seq, last_range_flag)
                future_5days = self.predict_future_5days(hybrid_model, xgb_model, ridge_model, last_seq, next_day_pred, last_range_flag)
                fig_future = self.create_future_chart(processed_data, future_5days)
                st.pyplot(fig_future)
            except Exception as e:
                st.error(f"未来5日予測エラー: {e}")

    # -----------------------------------------------------------
    # 補助関数：会社名取得
    def get_company_name(self):
        try:
            ticker = yf.Ticker(self.symbol)
            company_name = ticker.info.get("longName", "銘柄名不明")
            return company_name
        except Exception as e:
            logger.error(f"会社名取得エラー: {e}")
            return "銘柄名不明"

    # -----------------------------------------------------------
    # データ取得
    def fetch_stock_data(self):
        df = yf.download(self.symbol, period='1y', interval='1d', progress=False)
        if df.empty:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days)
            df = yf.download(
                self.symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                progress=False
            )
        if df.empty:
            raise ValueError(f"データが空: {self.symbol}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(tuple(map(str, c))) for c in df.columns]
        if 'Close' not in df.columns:
            possible_close = [c for c in df.columns if 'Close' in c]
            if len(possible_close) == 1:
                df.rename(columns={possible_close[0]: 'Close'}, inplace=True)
            elif len(possible_close) == 0:
                raise ValueError(f"Close列が見つかりません(列={df.columns})")
            else:
                raise ValueError(f"Close列が複数存在します(列={possible_close})")
        for colname in ["Open", "High", "Low", "Volume"]:
            if colname not in df.columns:
                possible_cols = [c for c in df.columns if colname in c]
                if len(possible_cols) == 1:
                    df.rename(columns={possible_cols[0]: colname}, inplace=True)
                elif len(possible_cols) == 0:
                    raise ValueError(f"{colname} 列が見つかりません(列={df.columns})")
                else:
                    raise ValueError(f"{colname} 列が複数存在します(列={possible_cols})")
        if len(df) < self.input_window * 2:
            raise ValueError(f"データ不足({len(df)})。最低{self.input_window * 2}件は必要")
        return df

    def process_data(self, df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df = clip_outliers_iqr(df, columns=["Open", "High", "Low", "Close"], factor=1.5)
        df['Close_denoised'] = self.denoise_wavelet(
            df['Close'].values,
            wavelet='db4',
            level=2,
            threshold_factor=1.0
        )
        df["Close_orig"] = df["Close_denoised"].copy()
        df['Close'] = np.log1p(df['Close_denoised'])
        df["Close"] = df["Close"].rolling(window=3, min_periods=1).mean()
        df["Open"] = df["Open"].rolling(window=3, min_periods=1).mean()
        df["High"] = df["High"].rolling(window=3, min_periods=1).mean()
        df["Low"] = df["Low"].rolling(window=3, min_periods=1).mean()
        df["Momentum_3d"] = df["Close"] - df["Close"].shift(3)
        df["Momentum_1d"] = df["Close"] - df["Close"].shift(1)
        df["log_return"] = df["Close"].diff().fillna(0)
        df["volatility_20"] = df["Close"].pct_change().rolling(20).std().fillna(0)
        df["HV_20"] = df["log_return"].rolling(20).std().fillna(0) * np.sqrt(252)
        returns = df["log_return"].dropna()
        try:
            garch = arch_model(returns, vol="Garch", p=1, q=1)
            res = garch.fit(disp="off")
            df["predicted_volatility"] = res.conditional_volatility
        except Exception as e:
            logger.error(f"GARCHモデル失敗: {e}")
            df["predicted_volatility"] = np.nan
        df = self.add_technical_indicators(df)
        df["trend_strength"] = np.abs(df["ADX"])
        df["pct_change"] = df["Close"].pct_change().fillna(0)
        ma_20 = df["Close"].rolling(20).mean()
        df["ma_deviation"] = (df["Close"] - ma_20) / (ma_20 + 1e-9)
        df["ma_deviation"].fillna(0, inplace=True)
        df["momentum"] = df["Close"] - df["Close"].shift(10)
        df["momentum"].fillna(0, inplace=True)
        df.fillna(0, inplace=True)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = np.nan_to_num(df[col], nan=0.0, posinf=1e10, neginf=-1e10)
                df[col] = clip_extreme_values(df[col], threshold=1e8)
        return df

    def denoise_wavelet(self, data, wavelet='db4', level=2, threshold_factor=1.0):
        coeff = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeff[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(data))) * threshold_factor
        coeff[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeff[1:]]
        denoised = pywt.waverec(coeff, wavelet)
        return denoised[:len(data)]

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Volume_Change_Rate"] = df["Volume"].pct_change().fillna(0) * 100
        df["Volume_MA5"] = df["Volume"].rolling(window=10).mean()
        df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA5"] + 1e-9)
        df.drop("Volume_MA5", axis=1, inplace=True, errors='ignore')
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean() + 1e-9
        avg_loss = loss.rolling(14).mean() + 1e-9
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100/(1+rs))
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MA5'] = df['Close'].rolling(5).mean()
        std20 = df['Close'].rolling(20).std()
        df['Upper'] = df['MA5'] + 2*std20
        df['Lower'] = df['MA5'] - 2*std20
        df['Typical Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (df['Typical Price'] * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-9)
        df.drop(['Typical Price'], axis=1, inplace=True, errors='ignore')
        df["Volume_Surge"] = (df["Volume"] - df["Volume"].rolling(window=10).mean()) / (df["Volume"].rolling(window=20).std() + 1e-9)
        df["VWAP_deviation"] = (df["Close"] - df["VWAP"]) / (df["VWAP"] + 1e-9)
        df['PreviousClose'] = df['Close'].shift(1)
        df['TR'] = df[['High','Low','PreviousClose']].apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - row['PreviousClose']),
                abs(row['Low'] - row['PreviousClose'])
            ),
            axis=1
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['H_diff'] = df['High'] - df['High'].shift(1)
        df['L_diff'] = df['Low'].shift(1) - df['Low']
        df['+DM'] = np.where((df['H_diff'] > df['L_diff']) & (df['H_diff'] > 0), df['H_diff'], 0)
        df['-DM'] = np.where((df['L_diff'] > df['H_diff']) & (df['L_diff'] > 0), df['L_diff'], 0)
        df['+DI'] = 100 * (df['+DM'].rolling(14).mean() / (df['ATR'] + 1e-9))
        df['-DI'] = 100 * (df['-DM'].rolling(14).mean() / (df['ATR'] + 1e-9))
        df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'] + 1e-9))
        df['ADX'] = df['DX'].rolling(14).mean()
        df['OBV'] = np.where(
            df['Close'] > df['Close'].shift(1),  df['Volume'],
            np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)
        ).cumsum()
        n = 14
        df["Low_n"] = df["Low"].rolling(window=n).min()
        df["High_n"] = df["High"].rolling(window=n).max()
        df["%K"] = (df["Close"] - df["Low_n"]) / (df["High_n"] - df["Low_n"] + 1e-9) * 100
        df["%D"] = df["%K"].rolling(window=3).mean()
        df.drop(["Low_n", "High_n"], axis=1, inplace=True, errors='ignore')
        df.fillna(0, inplace=True)
        df.drop(['PreviousClose','TR','H_diff','L_diff','+DM','-DM','+DI','-DI','DX'], axis=1, inplace=True, errors='ignore')
        np.random.seed(42)
        macro_indicator = np.random.normal(loc=100, scale=5, size=len(df))
        df["Macro_Indicator"] = macro_indicator
        df["Slope_MA5"] = df["MA5"].diff().fillna(0)
        df["BB_Width"] = (df["Upper"] - df["Lower"]) / (df["MA5"] + 1e-9) * 100
        range_threshold = 1.5
        slope_threshold = 0.02
        df["Range_Market"] = (
            (df["BB_Width"] < range_threshold) &
            (df["Slope_MA5"].abs() < slope_threshold)
        ).astype(int)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = np.nan_to_num(df[col], nan=0.0, posinf=1e10, neginf=-1e10)
                df[col] = clip_extreme_values(df[col], threshold=1e8)
        return df

    def build_and_train_model(self, df):
        features = [
            'Open','High','Low','Close','Volume',
            'Volume_Change_Rate','Volume_Ratio','Volume_Surge','VWAP_deviation','RSI','MACD','Signal',
            'MA5','Upper','Lower','VWAP','ATR','ADX','OBV',
            '%K','%D','Macro_Indicator',
            'log_return','volatility_20','HV_20','predicted_volatility','trend_strength',
            'pct_change','ma_deviation','momentum',
            'Slope_MA5','BB_Width','Range_Market'
        ]
        for f in features:
            if f not in df.columns:
                raise ValueError(f"必要な特徴量が不足: {f}")
        data = df[features].values.astype(np.float32)
        train_size = int(len(data) * 0.5)
        self.scaler = StandardScaler()
        train_data = self.scaler.fit_transform(data[:train_size])
        test_data = self.scaler.transform(data[train_size:])
        train_data = np.nan_to_num(train_data, nan=0.0, posinf=1e10, neginf=-1e10)
        test_data = np.nan_to_num(test_data, nan=0.0, posinf=1e10, neginf=-1e10)
        train_data = clip_extreme_values(train_data, threshold=1e8)
        test_data = clip_extreme_values(test_data, threshold=1e8)
        X_train_seq, y_train_seq = self.create_sequences(train_data)
        X_test_seq, y_test_seq = self.create_sequences(test_data)
        train_range_flags = train_data[:, -1]
        test_range_flags = test_data[:, -1]
        range_market_flags_test = test_range_flags[self.input_window:]
        hybrid_model = self.train_hybrid_model(X_train_seq, y_train_seq)
        X_train_xgb, y_train_xgb = self.convert_for_xgboost(X_train_seq, y_train_seq)
        X_val_xgb, y_val_xgb = self.convert_for_xgboost(X_test_seq, y_test_seq)
        X_train_xgb = np.nan_to_num(X_train_xgb, nan=0.0, posinf=1e10, neginf=-1e10)
        y_train_xgb = np.nan_to_num(y_train_xgb, nan=0.0, posinf=1e10, neginf=-1e10)
        X_val_xgb = np.nan_to_num(X_val_xgb, nan=0.0, posinf=1e10, neginf=-1e10)
        y_val_xgb = np.nan_to_num(y_val_xgb, nan=0.0, posinf=1e10, neginf=-1e10)
        xgb_model = train_xgboost(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
        ridge_model = train_ridge(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
        return hybrid_model, xgb_model, ridge_model, X_test_seq, y_test_seq, range_market_flags_test

    def train_hybrid_model(self, X_train, y_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train.shape[2]
        model = HybridTransformerLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            nhead=8,
            num_layers=2
        ).to(device)
        criterion = HybridLoss(alpha=0.5)
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        batch_size = 32
        for epoch in range(self.epochs):
            model.train()
            permutation = np.random.permutation(len(X_train_t))
            num_batches = len(X_train_t) // batch_size
            total_loss = 0
            for i in range(num_batches):
                indices = permutation[i * batch_size : (i + 1) * batch_size]
                batch_X = X_train_t[indices]
                batch_y = y_train_t[indices]
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred.view(-1), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        return model

    def create_sequences(self, scaled_data):
        scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=1e10, neginf=-1e10)
        X, y = [], []
        for i in range(self.input_window, len(scaled_data)):
            X.append(scaled_data[i-self.input_window:i])
            y.append(scaled_data[i, 3])
        return np.array(X), np.array(y)

    def convert_for_xgboost(self, X_seq, y_seq):
        X_out = X_seq[:, -1, :]
        y_out = y_seq
        return X_out, y_out

    def predict_and_evaluate(self, hybrid_model, xgb_model, ridge_model, X_test, y_test, range_flags_test):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hybrid_model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_hybrid = hybrid_model(X_test_t).view(-1).cpu().numpy()
        X_test_xgb, _ = self.convert_for_xgboost(X_test, y_test)
        pred_xgb = xgb_model.predict(X_test_xgb)
        pred_ridge = ridge_model.predict(X_test_xgb)
        inv_pred_hybrid = self.inverse_close(pred_hybrid)
        inv_pred_xgb = self.inverse_close(pred_xgb)
        inv_pred_ridge = self.inverse_close(pred_ridge)
        inv_y_test = self.inverse_close(y_test.reshape(-1, 1))
        self.cf_hybrid = np.mean(inv_y_test) / np.mean(inv_pred_hybrid)
        self.cf_xgb = np.mean(inv_y_test) / np.mean(inv_pred_xgb)
        self.cf_ridge = np.mean(inv_y_test) / np.mean(inv_pred_ridge)
        adj_pred_hybrid = inv_pred_hybrid * self.cf_hybrid
        adj_pred_xgb = inv_pred_xgb * self.cf_xgb
        adj_pred_ridge = inv_pred_ridge * self.cf_ridge
        error_hybrid = mean_absolute_percentage_error(inv_y_test, adj_pred_hybrid)
        error_xgb = mean_absolute_percentage_error(inv_y_test, adj_pred_xgb)
        error_ridge = mean_absolute_percentage_error(inv_y_test, adj_pred_ridge)
        self.error_hybrid = error_hybrid
        self.error_xgb = error_xgb
        self.error_ridge = error_ridge
        pred_ensemble = np.zeros_like(adj_pred_xgb)
        for i in range(len(pred_ensemble)):
            if range_flags_test[i] == 1:
                w_xgb = 1 / (error_xgb + 1e-6)
                w_ridge = 1 / (error_ridge + 1e-6)
                total = w_xgb + w_ridge
                weight_xgb = w_xgb / total
                weight_ridge = w_ridge / total
                pred_ensemble[i] = weight_xgb * adj_pred_xgb[i] + weight_ridge * adj_pred_ridge[i]
            else:
                w_hybrid = 1 / (error_hybrid + 1e-6)
                w_xgb = 1 / (error_xgb + 1e-6)
                total = w_hybrid + w_xgb
                weight_hybrid = w_hybrid / total
                weight_xgb = w_xgb / total
                pred_ensemble[i] = weight_hybrid * adj_pred_hybrid[i] + weight_xgb * adj_pred_xgb[i]
        mape = mean_absolute_percentage_error(inv_y_test, pred_ensemble)
        mae = mean_absolute_error(inv_y_test, pred_ensemble)
        r2 = r2_score(inv_y_test, pred_ensemble)
        residuals = inv_y_test - pred_ensemble
        std_dev = np.std(residuals)
        ci_text = f"±{2.0 * std_dev:.2f}円"
        score = self.calculate_reliability_score(inv_y_test, pred_ensemble, mape, mae, r2)
        print("\n==== SHAP 分析結果 ====")
        shap_values = calculate_shap_values_func(xgb_model, X_test_xgb[:10], feature_names=[
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MACD', 'VWAP', 'ATR', 'ADX'
        ])
        print("========================\n")
        return pred_ensemble, mape, mae, r2, ci_text, score

    def adjust_prediction(self, raw_pred, model_type):
        inv_pred = self.inverse_close(np.array([[raw_pred]]))[0]
        if model_type == "hybrid":
            return inv_pred * self.cf_hybrid
        elif model_type == "xgb":
            return inv_pred * self.cf_xgb
        elif model_type == "ridge":
            return inv_pred * self.cf_ridge
        else:
            return inv_pred

    def predict_future_price(self, hybrid_model, xgb_model, ridge_model, last_seq, last_range_flag):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hybrid_model.eval()
        inp_t = torch.tensor(last_seq[np.newaxis, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            raw_pred_hybrid = hybrid_model(inp_t).item()
        adj_pred_hybrid = self.adjust_prediction(raw_pred_hybrid, "hybrid")
        last_seq_xgb, _ = self.convert_for_xgboost(last_seq[np.newaxis, ...], np.array([0]))
        raw_pred_xgb = xgb_model.predict(last_seq_xgb)[0]
        adj_pred_xgb = self.adjust_prediction(raw_pred_xgb, "xgb")
        raw_pred_ridge = ridge_model.predict(last_seq_xgb)[0]
        adj_pred_ridge = self.adjust_prediction(raw_pred_ridge, "ridge")
        if hasattr(self, "error_hybrid") and hasattr(self, "error_xgb") and hasattr(self, "error_ridge"):
            if last_range_flag == 1:
                w_xgb = 1 / (self.error_xgb + 1e-6)
                w_ridge = 1 / (self.error_ridge + 1e-6)
                total = w_xgb + w_ridge
                weight_xgb = w_xgb / total
                weight_ridge = w_ridge / total
                ensemble_pred = weight_xgb * adj_pred_xgb + weight_ridge * adj_pred_ridge
            else:
                w_hybrid = 1 / (self.error_hybrid + 1e-6)
                w_xgb = 1 / (self.error_xgb + 1e-6)
                total = w_hybrid + w_xgb
                weight_hybrid = w_hybrid / total
                weight_xgb = w_xgb / total
                ensemble_pred = weight_hybrid * adj_pred_hybrid + weight_xgb * adj_pred_xgb
        else:
            if last_range_flag == 1:
                ensemble_pred = 0.7 * adj_pred_xgb + 0.3 * adj_pred_ridge
            else:
                ensemble_pred = 0.9 * adj_pred_hybrid + 0.1 * adj_pred_xgb
        return ensemble_pred

    def generate_single_signal(self, predicted_price, current_price, confidence_score, threshold=60):
        if confidence_score < threshold:
            return "Hold"
        return "Buy" if predicted_price > current_price else "Sell"

    def predict_future_5days(self, hybrid_model, xgb_model, ridge_model, last_seq, next_day_pred, last_range_flag):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq = np.copy(last_seq)
        future = [next_day_pred]
        dummy_for_scale = np.zeros((1, self.scaler.n_features_in_))
        dummy_for_scale[0, 3] = next_day_pred
        scaled_dummy = self.scaler.transform(dummy_for_scale)
        next_day_scaled_close = scaled_dummy[0, 3]
        seq = np.roll(seq, -1, axis=0)
        seq[-1, 3] = next_day_scaled_close
        for _ in range(4):
            inp_t = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32).to(device)
            hybrid_model.eval()
            with torch.no_grad():
                raw_pred_hybrid = hybrid_model(inp_t).item()
            adj_pred_hybrid = self.adjust_prediction(raw_pred_hybrid, "hybrid")
            seq_xgb, _ = self.convert_for_xgboost(seq[np.newaxis, ...], np.array([0]))
            raw_pred_xgb = xgb_model.predict(seq_xgb)[0]
            adj_pred_xgb = self.adjust_prediction(raw_pred_xgb, "xgb")
            raw_pred_ridge = ridge_model.predict(seq_xgb)[0]
            adj_pred_ridge = self.adjust_prediction(raw_pred_ridge, "ridge")
            if hasattr(self, "error_hybrid") and hasattr(self, "error_xgb") and hasattr(self, "error_ridge"):
                if last_range_flag == 1:
                    w_xgb = 1 / (self.error_xgb + 1e-6)
                    w_ridge = 1 / (self.error_ridge + 1e-6)
                    total = w_xgb + w_ridge
                    weight_xgb = w_xgb / total
                    weight_ridge = w_ridge / total
                    ensemble_pred = weight_xgb * adj_pred_xgb + weight_ridge * adj_pred_ridge
                else:
                    w_hybrid = 1 / (self.error_hybrid + 1e-6)
                    w_xgb = 1 / (self.error_xgb + 1e-6)
                    total = w_hybrid + w_xgb
                    weight_hybrid = w_hybrid / total
                    weight_xgb = w_xgb / total
                    ensemble_pred = weight_hybrid * adj_pred_hybrid + weight_xgb * adj_pred_xgb
            else:
                if last_range_flag == 1:
                    ensemble_pred = 0.7 * adj_pred_xgb + 0.3 * adj_pred_ridge
                else:
                    ensemble_pred = 0.9 * adj_pred_hybrid + 0.1 * adj_pred_xgb
            future.append(ensemble_pred)
            scaled_dummy_ = np.zeros((1, self.scaler.n_features_in_))
            scaled_dummy_[0, 3] = ensemble_pred
            scaled_dummy_ = self.scaler.transform(scaled_dummy_)
            seq = np.roll(seq, -1, axis=0)
            seq[-1, :] = scaled_dummy_[0]
        return np.array(future)

    def update_chart(self, df, preds, future_5days):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        ax.set_facecolor('#2E2E2E')
        fig.patch.set_facecolor('#2E2E2E')
        ax.tick_params(axis='both', colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.plot(df.index, df['Close_orig'], label='実際の株価', color='skyblue')
        test_start = int(len(df) * 0.5) + self.input_window
        pred_dates = df.index[test_start: test_start + len(preds)]
        ax.plot(pred_dates, preds, label='バックテスト', color='limegreen')
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
        ax.plot(future_dates, future_5days, label='5日後予測', color='red')
        ax.legend(loc='upper left')
        ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.axvline(x=mdates.date2num(last_date - timedelta(days=30)),
                   color='#8B7500', linestyle='-', linewidth=0.7)
        ax.axvline(x=mdates.date2num(last_date - timedelta(days=90)),
                   color='#8B7500', linestyle='-', linewidth=0.7)
        return fig

    def update_chart_with_preline(self, df, pre_preds, pre_future_5days, corrected_preds, corrected_future_5days):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        ax.set_facecolor('#2E2E2E')
        fig.patch.set_facecolor('#2E2E2E')
        ax.tick_params(axis='both', colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.plot(df.index, df['Close_orig'], label='実際の株価', color='skyblue')
        test_start = int(len(df) * 0.5) + self.input_window
        pred_dates = df.index[test_start: test_start + len(pre_preds)]
        ax.plot(pred_dates, pre_preds, color='limegreen', linestyle=':')
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
        ax.plot(future_dates, pre_future_5days, label='補正前予測', color='red', linestyle=':')
        ax.plot(pred_dates, corrected_preds, label='バックテスト', color='limegreen', linestyle='-')
        ax.plot(future_dates, corrected_future_5days, label='補正後予測', color='red', linestyle='-')
        ax.legend(loc='upper left')
        ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.axvline(x=mdates.date2num(last_date - timedelta(days=30)),
                   color='#8B7500', linestyle='-', linewidth=0.7)
        ax.axvline(x=mdates.date2num(last_date - timedelta(days=90)),
                   color='#8B7500', linestyle='-', linewidth=0.7)
        return fig

    def calculate_reliability_score(self, actual_values, pred_values, mape, mae, r2):
        mape_weight = 0.4
        mae_weight = 0.3
        r2_weight = 0.3
        mape_score = (1 - mape) * 100
        mae_score = (1 - mae / np.mean(np.abs(actual_values))) * 100
        r2_score_val = r2 * 100
        score = (mape_weight * mape_score +
                 mae_weight * mae_score +
                 r2_weight * r2_score_val)
        return score

    def inverse_close(self, arr):
        mean = self.scaler.mean_[3]
        scale = self.scaler.scale_[3]
        unscaled = arr.flatten() * scale + mean
        unscaled = np.clip(unscaled, -700, 700)
        unscaled = np.expm1(unscaled)
        unscaled = np.clip(unscaled, 0, 70000)
        return unscaled

    def train_rl_agent(self, df):
        env = StockTradingEnv(df)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        model.learn(total_timesteps=5000)
        obs = vec_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _info = vec_env.step(action)
            total_reward += reward[0]
        logger.info(f"RL最終報酬: {total_reward}")

    def save_model(self):
        try:
            if not all([self.hybrid_model, self.xgb_model, self.ridge_model]):
                raise ValueError("モデルが学習されていません。")
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"models/stock_predictor_{now_str}.pkl"
            import pickle
            with open(filename, 'wb') as f:
                models = {
                    'hybrid_model_state_dict': self.hybrid_model.state_dict(),
                    'xgb_model': self.xgb_model,
                    'ridge_model': self.ridge_model,
                    'scaler': self.scaler
                }
                pickle.dump(models, f)
            st.success(f"モデルを {filename} に保存しました。")
        except Exception as e:
            self.handle_error(e)

    def run_backtest(self, df):
        if len(df) < 120:
            raise ValueError(f"バックテスト用データが不足（最低120日必要）。現在: {len(df)}")
        backtest_df = df.iloc[-120:].copy()
        train_df = backtest_df.iloc[:90]
        test_df = backtest_df.iloc[90:]
        features = [
            'Open','High','Low','Close','Volume',
            'Volume_Change_Rate','Volume_Ratio','Volume_Surge','VWAP_deviation','RSI','MACD','Signal',
            'MA5','Upper','Lower','VWAP','ATR','ADX','OBV',
            '%K','%D','Macro_Indicator',
            'log_return','volatility_20','HV_20','predicted_volatility','trend_strength',
            'pct_change','ma_deviation','momentum',
            'Slope_MA5','BB_Width','Range_Market'
        ]
        for f in features:
            if f not in train_df.columns:
                raise ValueError(f"必要な特徴量が不足: {f}")
        train_data = train_df[features].values.astype(np.float32)
        test_data = test_df[features].values.astype(np.float32)
        scaler_bt = StandardScaler()
        train_scaled = scaler_bt.fit_transform(train_data)
        test_scaled = scaler_bt.transform(test_data)
        train_scaled = np.nan_to_num(train_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        test_scaled = np.nan_to_num(test_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        train_scaled = clip_extreme_values(train_scaled, threshold=1e8)
        test_scaled = clip_extreme_values(test_scaled, threshold=1e8)
        X_train_seq, y_train_seq = self.create_sequences(train_scaled)
        if len(X_train_seq) == 0:
            raise ValueError("バックテスト用のトレーニングデータが不足しています。")
        hybrid_model_bt = self.train_hybrid_model(X_train_seq, y_train_seq)
        X_train_xgb, y_train_xgb = self.convert_for_xgboost(X_train_seq, y_train_seq)
        xgb_model_bt = train_xgboost(X_train_xgb, y_train_xgb, X_train_xgb, y_train_xgb)
        ridge_model_bt = train_ridge(X_train_xgb, y_train_xgb, X_train_xgb, y_train_xgb)
        current_seq = train_scaled[-self.input_window:].copy()
        forecasts = []
        actuals = []
        for i in range(len(test_scaled)):
            pred_list = []
            for _ in range(6):
                old_scaler = self.scaler
                self.scaler = scaler_bt
                pred = self.predict_future_price(hybrid_model_bt, xgb_model_bt, ridge_model_bt, current_seq, 0)
                self.scaler = old_scaler
                pred_list.append(pred)
            forecast = np.median(pred_list)
            forecasts.append(forecast)
            if "Close_orig" in test_df.columns and (i < len(test_df["Close_orig"])):
                actual_close = test_df["Close_orig"].iloc[i]
            elif "Close_denoised" in test_df.columns and (i < len(test_df["Close_denoised"])):
                actual_close = test_df["Close_denoised"].iloc[i]
            else:
                actual_close = np.nan
            actuals.append(actual_close)
            current_seq = np.roll(current_seq, -1, axis=0)
            current_seq[-1, :] = test_scaled[i]
        forecasts = np.array(forecasts).flatten()
        actuals = np.array(actuals).flatten()
        valid_mask = (actuals != 0)
        if not any(valid_mask):
            backtest_mape = 999
            avg_pos = 0
            avg_neg = 0
        else:
            ape = np.abs((forecasts[valid_mask] - actuals[valid_mask]) / actuals[valid_mask]) * 100
            backtest_mape = np.mean(ape)
            diffs = (forecasts[valid_mask] - actuals[valid_mask]) / actuals[valid_mask] * 100
            pos_diffs = diffs[diffs > 0]
            neg_diffs = diffs[diffs < 0]
            avg_pos = np.mean(pos_diffs) if pos_diffs.size > 0 else 0
            avg_neg = np.mean(neg_diffs) if neg_diffs.size > 0 else 0
        return backtest_mape, avg_pos, avg_neg

    def fetch_composite_sentiment(self):
        tickers = ["^DJI", "^IXIC"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        sentiments = []
        for ticker in tickers:
            df_index = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                progress=False
            )
            df_index = df_index.sort_index()
            if len(df_index) < 2:
                sentiments.append(0.0)
                continue
            closes = df_index["Close"].values
            d0, d1 = closes[-2], closes[-1]
            if d0 == 0:
                sentiments.append(0.0)
                continue
            rate = (d1 - d0) / d0
            sentiments.append(rate)
        if len(sentiments) == 2:
            composite = 0.5 * sentiments[0] + 0.5 * sentiments[1]
        else:
            composite = 0.0
        return composite

    def optimize_sentiment_coefficient(self):
        best_c = 0.008
        return best_c

    def multiple_run_prediction(self, processed_data, is_adjustment=False):
        even_preds = []
        even_future_5days = []
        even_future_prices = []
        even_scores = []
        residuals_list = []
        for i in range(1, 7):
            # UI更新部分はStreamlitの場合は省略
            hybrid_model, xgb_model, ridge_model, X_test, y_test, range_flags_test = self.build_and_train_model(processed_data)
            preds, mape, mae, r2, ci_text, score = self.predict_and_evaluate(
                hybrid_model, xgb_model, ridge_model,
                X_test, y_test, range_flags_test
            )
            if len(X_test) == 0:
                raise ValueError("テストデータが存在しないため、予測ができません。")
            future_price = self.predict_future_price(
                hybrid_model, xgb_model, ridge_model,
                X_test[-1],
                range_flags_test[-1] if len(range_flags_test) > 0 else 0
            )
            future_5days = self.predict_future_5days(
                hybrid_model, xgb_model, ridge_model,
                X_test[-1], future_price,
                range_flags_test[-1] if len(range_flags_test) > 0 else 0
            )
            inv_y_test = self.inverse_close(y_test.reshape(-1, 1))
            inv_preds = self.inverse_close(preds)
            residuals = inv_y_test - inv_preds
            residuals_list.append(residuals)
            if i % 2 == 0:
                even_preds.append(preds)
                even_future_5days.append(future_5days)
                even_future_prices.append(future_price)
                even_scores.append(score)
        avg_preds = np.median(even_preds, axis=0)
        avg_future_5days = np.median(even_future_5days, axis=0)
        avg_future_price = np.median(even_future_prices)
        avg_score = np.median(even_scores)
        all_residuals = np.concatenate(residuals_list)
        std_dev = np.std(all_residuals) if len(all_residuals) > 0 else 0
        return avg_preds, avg_future_5days, avg_future_price, avg_score, std_dev

    def train_residual_model(self, df_corrected):
        if len(df_corrected) < 120:
            raise ValueError("補正モデル学習用のデータが不足")
        backtest_df = df_corrected.iloc[-120:].copy()
        train_df = backtest_df.iloc[:90]
        test_df = backtest_df.iloc[90:]
        all_columns = list(df_corrected.columns)
        numeric_cols = []
        for c in all_columns:
            if pd.api.types.is_numeric_dtype(df_corrected[c]):
                numeric_cols.append(c)
        self.features_for_correction = numeric_cols
        train_data = train_df[self.features_for_correction].values.astype(np.float32)
        test_data = test_df[self.features_for_correction].values.astype(np.float32)
        self.residual_scaler = StandardScaler()
        train_scaled = self.residual_scaler.fit_transform(train_data)
        test_scaled = self.residual_scaler.transform(test_data)
        actual_close_train = train_df["Close_orig"].values
        pred_close_train = np.expm1(train_df["Close"].values)
        residual_train = actual_close_train - pred_close_train
        self.mae_val = np.mean(np.abs(residual_train))
        self.mbe_val = np.mean(residual_train)
        predicted_diff = np.diff(pred_close_train)
        actual_diff = np.diff(actual_close_train)
        self.directional_accuracy = np.mean(np.sign(predicted_diff) == np.sign(actual_diff))
        X_train_resid = train_scaled
        y_train_resid = residual_train
        self.residual_model = LinearRegression()
        self.residual_model.fit(X_train_resid, y_train_resid)

    def apply_residual_correction(self, df_corrected, next_day_pred, future_5days):
        latest_row = df_corrected.iloc[-1][self.features_for_correction].values.astype(np.float32).reshape(1, -1)
        latest_scaled = self.residual_scaler.transform(latest_row)
        correction_value_next_day = self.residual_model.predict(latest_scaled)[0]
        optimal_correction_factor = 1 + (self.mbe_val / (self.mae_val + 1e-6)) * ((self.directional_accuracy - 0.5) * 2)
        optimal_correction_factor = np.clip(optimal_correction_factor, 0.5, 2.0)
        corrected_next_day = next_day_pred + optimal_correction_factor * correction_value_next_day
        corrected_f5 = future_5days + optimal_correction_factor * correction_value_next_day
        backtest_mape_resid, _, _ = self.run_backtest(df_corrected)
        mape_resid = backtest_mape_resid
        return corrected_next_day, corrected_f5, mape_resid

    def create_future_chart(self, df, future_5days):
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        ax.set_facecolor('#2E2E2E')
        fig.patch.set_facecolor('#2E2E2E')
        ax.tick_params(axis='both', colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.plot(df.index, df['Close_orig'], label='実際の株価', color='skyblue')
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
        
        ax.legend(loc='upper left')
        ax.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        ax.axvline(x=mdates.date2num(last_date - timedelta(days=30)),
                   color='#8B7500', linestyle='-', linewidth=0.7)
        ax.axvline(x=mdates.date2num(last_date - timedelta(days=90)),
                   color='#8B7500', linestyle='-', linewidth=0.7)
        return fig

    def calculate_shap_values(self, model, X_sample, feature_names):
        try:
            if X_sample is None or len(X_sample) == 0:
                logger.error("SHAP計算エラー: X_sample が空です。")
                return None
            logger.info(f"SHAP計算対象データの形状: {X_sample.shape}")
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample)
            if shap_values is None or shap_values.values is None:
                logger.error("SHAP計算エラー: shap_values が None です。")
                return None
            importance = np.abs(shap_values.values).mean(axis=0)
            sorted_indices = np.argsort(importance)[::-1]
            num_features_to_show = min(5, len(feature_names))
            print("\n===== SHAP 重要特徴量（TOP {}） =====".format(num_features_to_show))
            for i in range(num_features_to_show):
                idx = sorted_indices[i]
                print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.6f}")
            return shap_values
        except Exception as e:
            logger.error(f"SHAP計算エラー: {e}")
            return None

def TrainingProgressCallback(update_fn):
    class CallbackImpl(Callback):
        def __init__(self, update_fn):
            super().__init__()
            self.update_fn = update_fn
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / self.params['epochs']
            self.update_fn(progress * 0.5 + 0.2)
    return CallbackImpl(update_fn)

import gym
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_cash=1000000):
        super(StockTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.nrows = len(df)
        self.initial_cash = initial_cash
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        self._reset_internal_state()
    def _reset_internal_state(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.done = False
    def reset(self):
        self._reset_internal_state()
        return self._get_observation()
    def step(self, action):
        reward = 0
        prev_asset = self._get_asset_value()
        current_price = self.df.loc[self.current_step, 'Close']
        if action == 1:
            shares_to_buy = int(self.cash // current_price)
            self.shares += shares_to_buy
            self.cash -= shares_to_buy * current_price
        elif action == 2:
            if self.shares > 0:
                self.cash += self.shares * current_price
                self.shares = 0
        self.current_step += 1
        if self.current_step >= self.nrows - 1:
            self.done = True
        current_asset = self._get_asset_value()
        reward = current_asset - prev_asset
        obs = self._get_observation()
        return obs, reward, self.done, {}
    def _get_observation(self):
        current_price = self.df.loc[self.current_step, 'Close']
        log_return = self.df.loc[self.current_step, 'log_return']
        vol20 = self.df.loc[self.current_step, 'volatility_20']
        return np.array([current_price, log_return, vol20, self.cash, self.shares], dtype=np.float32)
    def _get_asset_value(self):
        current_price = self.df.loc[self.current_step, 'Close']
        return self.cash + (self.shares * current_price)

def calculate_shap_values_func(model, X_sample, feature_names):
    try:
        if X_sample is None or len(X_sample) == 0:
            logger.error("SHAP計算エラー: X_sample が空です。")
            return None
        logger.info(f"SHAP計算対象データの形状: {X_sample.shape}")
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        if shap_values is None or shap_values.values is None:
            logger.error("SHAP計算エラー: shap_values が None です。")
            return None
        importance = np.abs(shap_values.values).mean(axis=0)
        sorted_indices = np.argsort(importance)[::-1]
        num_features_to_show = min(5, len(feature_names))
        print("\n===== SHAP 重要特徴量（TOP {}） =====".format(num_features_to_show))
        for i in range(num_features_to_show):
            idx = sorted_indices[i]
            print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.6f}")
        return shap_values
    except Exception as e:
        logger.error(f"SHAP計算エラー: {e}")
        return None

if __name__ == "__main__":
    app = AdvancedStockPredictorApp()
    app.run()
