# PNN50 / βα 推定 Streamlit アプリ

空間画像特徴量CSVから **PNN50** と **β/α (beta_alpha)** をそれぞれ学習・予測し、Russell円環上に可視化する初心者向けアプリです。

---

## 1. ファイル構成

```text
.
├── app.py              # Streamlitアプリ本体
├── requirements.txt    # 依存ライブラリ
├── .env.example        # 環境変数サンプル
└── README.md           # セットアップ手順
```

---

## 2. セットアップ手順（初心者向け）

### 2.1 Python仮想環境を作成

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2.2 ライブラリをインストール

```bash
pip install -r requirements.txt
```

### 2.3 .env を作成

```bash
cp .env.example .env
```

`.env` を開いて必要に応じてAPIキーを設定します。

```env
OPENAI_API_KEY=your_api_key_here
```

> このアプリでは主にローカル推論を行いますが、キー管理の練習として `.env` 読み込みを実装しています。

---

## 3. 実行方法

```bash
streamlit run app.py
```

ブラウザで表示された画面からCSVをアップロードしてください。

---

## 4. CSV仕様

必須列:
- `PNN50`（目的変数1）
- `beta_alpha`（目的変数2）

その他の**数値列**は特徴量として自動利用されます。

---

## 5. アプリの機能

- CSVアップロード
- 欠損値処理（中央値補完）
- 特徴量の標準化（StandardScaler）
- モデル選択（RandomForest / XGBoost / MLP）
- PNN50とβ/αを別々に学習・予測
- Russell円環上への予測プロット
- 重要特徴量（Permutation Importance）の可視化
- 予測結果CSVのダウンロード
- 例外処理とエラーメッセージ表示

---

## 6. トラブルシュート

- `必要な列がありません` と表示される
  - CSVに `PNN50` と `beta_alpha` があるか確認してください。
- XGBoostが選択肢に出ない
  - `pip install -r requirements.txt` を再実行してください。
- データ行が少ない
  - 各目的変数ごとに最低10行以上の教師データが必要です。
