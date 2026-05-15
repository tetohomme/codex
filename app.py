import os
import traceback
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional XGBoost import (graceful fallback)
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


TARGET_PNN50 = "PNN50"
TARGET_RATIO = "beta_alpha"


def build_model(model_name: str, random_state: int = 42):
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost がインストールされていません。requirements を確認してください。")
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=4,
        )
    if model_name == "MLP":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=800,
            random_state=random_state,
        )
    raise ValueError(f"未知のモデル: {model_name}")


def train_for_target(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float], np.ndarray, np.ndarray]:
    train_df = df.dropna(subset=[target_col]).copy()
    if len(train_df) < 10:
        raise ValueError(f"{target_col} の学習データが不足しています（10行未満）。")

    X = train_df[feature_cols]
    y = train_df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(model_name)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)

    val_score = pipeline.score(X_valid, y_valid)

    pred_all = pipeline.predict(df[feature_cols])

    p_importance = permutation_importance(
        pipeline,
        X_valid,
        y_valid,
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    imp_df = pd.DataFrame(
        {"feature": feature_cols, "importance": p_importance.importances_mean}
    ).sort_values("importance", ascending=False)

    metrics = {
        "R2(valid)": float(val_score),
        "train_rows": float(len(train_df)),
        "valid_rows": float(len(X_valid)),
    }
    return pipeline, imp_df, metrics, pred_all, train_df.index.values


def make_russell_ring_plot(pred_df: pd.DataFrame) -> go.Figure:
    theta = np.linspace(0, 2 * np.pi, 400)
    outer_r = 1.0
    inner_r = 0.7

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=outer_r * np.cos(theta),
            y=outer_r * np.sin(theta),
            mode="lines",
            line=dict(color="lightgray"),
            name="outer",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=inner_r * np.cos(theta),
            y=inner_r * np.sin(theta),
            mode="lines",
            line=dict(color="lightgray"),
            name="inner",
            showlegend=False,
        )
    )

    ba = pred_df[TARGET_RATIO].values
    pnn = pred_df[TARGET_PNN50].values

    ba_scaled = (ba - np.nanmin(ba)) / (np.nanmax(ba) - np.nanmin(ba) + 1e-9)
    pnn_scaled = (pnn - np.nanmin(pnn)) / (np.nanmax(pnn) - np.nanmin(pnn) + 1e-9)

    radius = inner_r + (outer_r - inner_r) * pnn_scaled
    angle = 2 * np.pi * ba_scaled

    x = radius * np.cos(angle)
    y = radius * np.sin(angle)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=9,
                color=pnn,
                colorscale="Viridis",
                colorbar=dict(title="Pred PNN50"),
                opacity=0.9,
            ),
            text=[f"row={i}<br>β/α={b:.3f}<br>PNN50={p:.3f}" for i, b, p in zip(pred_df.index, ba, pnn)],
            hoverinfo="text",
            name="predictions",
        )
    )

    fig.update_layout(
        title="Russell円環上の予測マップ",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        width=700,
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def main():
    st.set_page_config(page_title="PNN50 / βα 推定アプリ", layout="wide")
    st.title("空間画像特徴量から PNN50 と β/α を推定")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    with st.expander("環境設定（.env）"):
        if api_key:
            st.success("OPENAI_API_KEY を .env から読み込みました。")
        else:
            st.warning("OPENAI_API_KEY が未設定です。必要なら .env を作成してください。")
            st.code("OPENAI_API_KEY=your_api_key_here")

    st.markdown(
        "CSVには目的変数列として `PNN50` と `beta_alpha` を含めることを想定しています。"
    )

    uploaded = st.file_uploader("CSVをアップロード", type=["csv"])
    if uploaded is None:
        st.info("CSVをアップロードすると学習・推定を開始します。")
        return

    try:
        df = pd.read_csv(uploaded)
        if df.empty:
            st.error("CSVが空です。内容を確認してください。")
            return

        missing_targets = [c for c in [TARGET_PNN50, TARGET_RATIO] if c not in df.columns]
        if missing_targets:
            st.error(f"必要な列がありません: {missing_targets}")
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in [TARGET_PNN50, TARGET_RATIO]]
        if len(feature_cols) < 2:
            st.error("数値特徴量列が不足しています（最低2列必要）。")
            return

        model_options = ["RandomForest", "MLP"] + (["XGBoost"] if XGBOOST_AVAILABLE else [])
        model_name = st.selectbox("学習モデルを選択", model_options)

        st.write("### 入力データ概要")
        st.dataframe(df.head(10))
        st.write(f"行数: {len(df)} / 特徴量数: {len(feature_cols)}")

        _, imp_pnn, m_pnn, pred_pnn, _ = train_for_target(df, TARGET_PNN50, feature_cols, model_name)
        _, imp_ratio, m_ratio, pred_ratio, _ = train_for_target(df, TARGET_RATIO, feature_cols, model_name)

        result_df = df.copy()
        result_df[TARGET_PNN50] = pred_pnn
        result_df[TARGET_RATIO] = pred_ratio

        st.write("### 予測メトリクス")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("PNN50 R2(valid)", f"{m_pnn['R2(valid)']:.3f}")
        with c2:
            st.metric("β/α R2(valid)", f"{m_ratio['R2(valid)']:.3f}")

        st.write("### Russell円環可視化")
        ring_fig = make_russell_ring_plot(result_df[[TARGET_PNN50, TARGET_RATIO]])
        st.plotly_chart(ring_fig, use_container_width=True)

        st.write("### 重要特徴量（Permutation Importance）")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("PNN50")
            st.bar_chart(imp_pnn.set_index("feature").head(15))
        with col2:
            st.subheader("β/α")
            st.bar_chart(imp_ratio.set_index("feature").head(15))

        st.write("### 予測結果（先頭20行）")
        st.dataframe(result_df[[TARGET_PNN50, TARGET_RATIO]].head(20))

        csv_bytes = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "予測結果CSVをダウンロード",
            data=csv_bytes,
            file_name="predicted_results.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error("処理中にエラーが発生しました。CSV形式や列名を確認してください。")
        st.exception(e)
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
