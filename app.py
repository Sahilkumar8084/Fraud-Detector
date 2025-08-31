import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify
import inflection
import xgboost as xgb
import numpy as np
import shap
import plotly
import plotly.express as px
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import traceback

# --- Fix / Compatibility for possible xgboost label encoder pickles ---
class XGBoostLabelEncoder:
    def __init__(self, *args, **kwargs):
        self.classes_ = None
        if args:
            try:
                self.classes_ = args[0]
            except:
                self.classes_ = None

    def fit(self, y):
        self.classes_ = sorted(set(y))
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("Encoder not fitted yet")
        return [self.classes_.index(val) if val in self.classes_ else -1 for val in y]

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        if self.classes_ is None:
            raise ValueError("Encoder not fitted yet")
        return [self.classes_[idx] if 0 <= idx < len(self.classes_) else None for idx in y]


# Monkey patch into xgboost.compat if available
try:
    import xgboost.compat as _xgb_compat
    _xgb_compat.XGBoostLabelEncoder = XGBoostLabelEncoder
except Exception:
    pass

# --- Paths (update if your filenames are different) ---
MODEL_PATH = "model_cycle1 (1).joblib"
SCALER_PATH = "minmaxscaler_cycle1.joblib"
OHE_PATH = "onehotencoder_cycle1.joblib"

model = None
explainer = None

# Load model and try to initialize SHAP if possible
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully:", type(model))
    try:
        # Use TreeExplainer for tree models
        if hasattr(model, "get_booster") or "XGB" in str(type(model)).upper():
            explainer = shap.TreeExplainer(model)
            print("✅ SHAP TreeExplainer initialized")
        else:
            explainer = None
            print("⚠️ SHAP not initialized (model type not tree-like)")
    except Exception as e:
        print("⚠️ Could not initialize SHAP:", e)
        explainer = None
except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    explainer = None

app = Flask(__name__)


class Fraud:
    def __init__(self):
        # load scalers safely
        try:
            self.minmaxscaler = joblib.load(SCALER_PATH)
            print("✅ MinMax scaler loaded")
        except Exception as e:
            print("⚠️ Could not load minmax scaler:", e)
            self.minmaxscaler = None

        try:
            self.onehotencoder = joblib.load(OHE_PATH)
            print("✅ OneHot encoder loaded")
        except Exception as e:
            print("⚠️ Could not load one-hot encoder:", e)
            self.onehotencoder = None

    def data_cleaning(self, df1: pd.DataFrame) -> pd.DataFrame:
        snakecase = lambda i: inflection.underscore(i)
        df1 = df1.copy()
        df1.columns = [snakecase(c) for c in df1.columns]
        return df1

    def feature_engineering(self, df2: pd.DataFrame) -> pd.DataFrame:
        df2 = df2.copy()
        # safe balance diffs
        if 'newbalance_orig' in df2.columns and 'oldbalance_org' in df2.columns:
            df2['diff_new_old_balance'] = df2['newbalance_orig'] - df2['oldbalance_org']
        else:
            df2['diff_new_old_balance'] = 0.0

        if 'newbalance_dest' in df2.columns and 'oldbalance_dest' in df2.columns:
            df2['diff_new_old_destiny'] = df2['newbalance_dest'] - df2['oldbalance_dest']
        else:
            df2['diff_new_old_destiny'] = 0.0

        # Extract first char of names
        for col in ['name_orig', 'name_dest']:
            if col in df2.columns:
                df2[col] = df2[col].apply(lambda i: i[0] if isinstance(i, str) and len(i) > 0 else 'U')

        columns_to_keep = ['step', 'amount', 'oldbalance_org', 'newbalance_orig',
                           'oldbalance_dest', 'newbalance_dest', 'type',
                           'diff_new_old_balance', 'diff_new_old_destiny']

        available_columns = [c for c in columns_to_keep if c in df2.columns]
        return df2[available_columns].copy()

    def data_preparation(self, df3: pd.DataFrame) -> pd.DataFrame:
        df_proc = df3.copy().reset_index(drop=True)

        # One-hot encode 'type' if encoder available
        if self.onehotencoder is not None and 'type' in df_proc.columns:
            try:
                enc = self.onehotencoder.transform(df_proc[['type']])
                if hasattr(enc, "toarray"):
                    enc_df = pd.DataFrame(enc.toarray(), columns=self.onehotencoder.get_feature_names_out())
                else:
                    enc_df = pd.DataFrame(enc, columns=self.onehotencoder.get_feature_names_out())
                df_proc = df_proc.drop(columns=['type']).reset_index(drop=True)
                df_proc = pd.concat([df_proc.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)
            except Exception as e:
                print("⚠️ OHE transform failed:", e)

        # Scale numeric columns if scaler available
        num_columns = ['amount', 'oldbalance_org', 'newbalance_orig', 'oldbalance_dest',
                       'newbalance_dest', 'diff_new_old_balance', 'diff_new_old_destiny']
        available_num = [c for c in num_columns if c in df_proc.columns]
        if self.minmaxscaler is not None and len(available_num) > 0:
            try:
                scaled = self.minmaxscaler.transform(df_proc[available_num])
                df_proc.loc[:, available_num] = scaled
            except Exception as e:
                print("⚠️ Scaling failed:", e)

        # Final expected columns (adapt to your model if different)
        expected_columns = ['step', 'oldbalance_org', 'newbalance_orig', 'newbalance_dest',
                            'diff_new_old_balance', 'diff_new_old_destiny', 'type_TRANSFER']

        for col in expected_columns:
            if col not in df_proc.columns:
                df_proc[col] = 0.0

        # Ensure column order and types
        return df_proc[expected_columns].astype(np.float32)

    def get_prediction(self, model_obj, original_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns: DataFrame (copy of original_data with appended columns:
                 prediction, fraud_probability, legit_probability, feature_importance)
        """
        od = original_data.copy().reset_index(drop=True)

        if model_obj is None:
            od['prediction'] = 0
            od['fraud_probability'] = 0.0
            od['legit_probability'] = 1.0
            od['model_error'] = "Model not loaded"
            od['feature_importance'] = [None]  # one cell containing None
            return od

        # Prepare feature array
        try:
            X = np.asarray(test_data.values)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        except Exception:
            X = np.atleast_2d(test_data)

        # Predict label
        try:
            pred = model_obj.predict(X)
            pred_label = int(pred[0]) if hasattr(pred, '__len__') else int(pred)
        except Exception as e:
            print("Prediction error:", e)
            pred_label = 0

        # Predict probabilities (safe)
        fraud_prob = None
        legit_prob = None
        try:
            if hasattr(model_obj, "predict_proba"):
                proba = model_obj.predict_proba(X)
                if proba.shape[1] >= 2:
                    fraud_prob = float(proba[0, 1])
                    legit_prob = float(proba[0, 0])
                else:
                    fraud_prob = float(proba[0, 0])
                    legit_prob = 1.0 - fraud_prob
            else:
                fraud_prob = 0.8 if pred_label == 1 else 0.2
                legit_prob = 1.0 - fraud_prob
        except Exception as e:
            print("Predict_proba error:", e)
            fraud_prob = 0.5
            legit_prob = 0.5

        # SHAP / feature importance (best-effort)
        feature_importance = None
        try:
            if explainer is not None:
                shap_values = explainer.shap_values(X)
                # shap_values can be array or list; handle both
                if isinstance(shap_values, (list, tuple)):
                    sv = shap_values[0]
                else:
                    sv = shap_values
                # if sv is 2D (n_samples x n_features), take first row
                if isinstance(sv, np.ndarray) and sv.ndim == 2:
                    values = sv[0]
                else:
                    values = np.array(sv).ravel()

                feature_names = test_data.columns.tolist()
                # align length: if mismatch, cut or pad with zeros
                if len(values) != len(feature_names):
                    # pad or cut
                    min_len = min(len(values), len(feature_names))
                    feature_names = feature_names[:min_len]
                    values = values[:min_len]

                shap_df = pd.DataFrame({
                    "feature": feature_names,
                    "shap_value": np.array(values)
                })
                shap_df["importance"] = shap_df["shap_value"].abs()
                shap_df = shap_df.sort_values("importance", ascending=False).reset_index(drop=True)
                feature_importance = shap_df.to_dict("records")
        except Exception as e:
            print("SHAP computation error:", e)
            feature_importance = None

        # Attach results to od: make sure lengths match (one row)
        od['prediction'] = pred_label
        od['fraud_probability'] = fraud_prob
        od['legit_probability'] = legit_prob
        # Put feature_importance in a single cell as a list so pandas accepts it
        od['feature_importance'] = [feature_importance]

        return od

    def generate_shap_plot(self, test_data: pd.DataFrame):
        """Generate SHAP plot as base64 PNG (best-effort)."""
        if explainer is None or test_data is None:
            return None
        try:
            X = np.asarray(test_data.values)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            shap_values = explainer.shap_values(X)

            plt.figure(figsize=(8, 4))
            try:
                shap.force_plot(explainer.expected_value, shap_values[0], X[0],
                                feature_names=test_data.columns, matplotlib=True, show=False)
            except Exception:
                # fallback to summary plot if force_plot fails
                shap.summary_plot(shap_values, X, feature_names=test_data.columns, show=False)

            buffer = BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            img_bytes = buffer.getvalue()
            buffer.close()
            plt.close()
            return "data:image/png;base64," + base64.b64encode(img_bytes).decode("utf-8")
        except Exception as e:
            print("Error generating SHAP plot:", e)
            return None


# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/dashboard')
def dashboard():
    df = pd.read_csv("transactions.csv")
    df.columns = [c.lower() for c in df.columns]  # normalize column names

    # --- Stats ---
    total_transactions = len(df)
    total_fraud = df['isfraud'].sum()
    fraud_rate_overall = (total_fraud / total_transactions) * 100 if total_transactions > 0 else 0

    # --- Interactive Charts ---

    # 1. Fraud Rate by Transaction Type
    fraud_by_type = df.groupby('type')['isfraud'].mean().reset_index()
    fraud_by_type['fraud_rate_percent'] = fraud_by_type['isfraud']*100
    fig1 = px.bar(fraud_by_type, x='type', y='fraud_rate_percent',
                  title='Fraud Rate by Transaction Type',
                  labels={'type':'Transaction Type','fraud_rate_percent':'Fraud Rate (%)'},
                  text='fraud_rate_percent')

    # 2. Fraud vs Legit Count per Type (stacked bar)
    counts = df.groupby(['type','isfraud']).size().reset_index(name='count')
    fig2 = px.bar(counts, x='type', y='count', color='isfraud',
                  title='Fraud vs Legit Count per Transaction Type',
                  labels={'isfraud':'Is Fraud','count':'Number of Transactions'},
                  color_discrete_map={0:'#4a6fc0',1:'#e74c3c'})

    # 3. Top 10 Accounts with Most Fraudulent Transactions
    top_accounts = df[df['isfraud']==1].groupby('nameorig')['isfraud'].sum().sort_values(ascending=False).head(10).reset_index()
    fig3 = px.bar(top_accounts, y='nameorig', x='isfraud', orientation='h',
                  title='Top 10 Accounts with Most Fraudulent Transactions',
                  labels={'nameorig':'Account','isfraud':'Number of Fraud Transactions'},
                  text='isfraud')

    interactive_graphs = [fig1, fig2, fig3]
    graphsJSON = [json.dumps(g, cls=plotly.utils.PlotlyJSONEncoder) for g in interactive_graphs]

    # --- Static Charts ---

    # 4. Fraud vs Legit over Step
    step_counts = df.groupby(['step','isfraud']).size().reset_index(name='count')
    fig4 = px.line(step_counts, x='step', y='count', color='isfraud',
                   title='Fraud vs Legit Transactions Over Steps',
                   labels={'step':'Time Step','count':'Number of Transactions','isfraud':'Is Fraud'},
                   color_discrete_map={0:'#4a6fc0', 1:'#e74c3c'})

    # 5. Fraud Amount Distribution
    fraud_amounts = df[df['isfraud']==1]
    fig5 = px.histogram(fraud_amounts, x='amount', nbins=20,
                        title='Fraudulent Transaction Amount Distribution',
                        labels={'amount':'Transaction Amount','count':'Count'},
                        color_discrete_sequence=['#e74c3c'])

    # 6. Pie Chart: Fraud vs Legit
    fig6 = px.pie(df, names='isfraud', title='Overall Fraud vs Legit Transactions',
                  color='isfraud', color_discrete_map={0:'#4a6fc0',1:'#e74c3c'},
                  labels={'isfraud':'Is Fraud'})

    # 7. Transaction Amount Distribution by Type (Box Plot)
    fig7 = px.box(df, x='type', y='amount', color='isfraud',
                  title='Transaction Amount Distribution by Type',
                  labels={'amount':'Amount','type':'Transaction Type','isfraud':'Is Fraud'},
                  color_discrete_map={0:'#4a6fc0',1:'#e74c3c'})

    # 8. Cumulative Fraud Over Steps
    cumulative = df[df['isfraud']==1].groupby('step')['isfraud'].sum().cumsum().reset_index()
    fig8 = px.line(cumulative, x='step', y='isfraud',
                   title='Cumulative Fraud Over Steps',
                   labels={'step':'Time Step','isfraud':'Cumulative Fraud'},
                   line_shape='spline')

    # # 9. Top 10 Destinations Targeted by Fraud
    # top_dest = df[df['isfraud']==1].groupby('namedest')['isfraud'].sum().sort_values(ascending=False).head(10).reset_index()
    # fig9 = px.bar(top_dest, y='namedest', x='isfraud', orientation='h',
    #               title='Top 10 Destinations Targeted by Fraud',
    #               labels={'namedest':'Account','isfraud':'Fraud Count'},
    #               text='isfraud', color_discrete_sequence=['#e74c3c'])

    # 10. Transaction Type Distribution (Pie)
    fig10 = px.pie(df, names='type',
                   title='Transaction Type Distribution',
                   labels={'type':'Transaction Type'},
                   hole=0.3)

    # Convert static charts to base64 PNG
    static_graphs = []
    for g in [fig4, fig5, fig6, fig7, fig8, fig10]:
        buf = BytesIO()
        g.write_image(buf, format='png', width=700, height=450)
        buf.seek(0)
        img_bytes = buf.getvalue()
        buf.close()
        img_b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode('utf-8')
        static_graphs.append(img_b64)

    return render_template('dashboard.html',
                           graphsJSON=graphsJSON,
                           staticGraphs=static_graphs,
                           total_transactions=total_transactions,
                           total_fraud=total_fraud,
                           fraud_rates=fraud_rate_overall)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        form_data = {
            'step': float(form.get('step', 1)),
            'amount': float(form.get('amount', 0.0)),
            'oldbalance_org': float(form.get('oldbalance_org', 0.0)),
            'newbalance_orig': float(form.get('newbalance_orig', 0.0)),
            'oldbalance_dest': float(form.get('oldbalance_dest', 0.0)),
            'newbalance_dest': float(form.get('newbalance_dest', 0.0)),
            'type': form.get('type', 'TRANSFER'),
            'name_orig': form.get('name_orig', 'U'),
            'name_dest': form.get('name_dest', 'U')
        }

        test_raw = pd.DataFrame(form_data, index=[0])
        pipeline = Fraud()
        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)

        # Prediction and SHAP
        result = pipeline.get_prediction(model, test_raw, df3)
        # result guaranteed to be DataFrame
        if result is None or result.empty:
            return render_template('index.html', error="Prediction failed: no result returned", form_data=form_data)

        # Extract info from first row
        row = result.iloc[0].to_dict()
        shap_plot = pipeline.generate_shap_plot(df3)

        response_data = {
            'is_fraud': bool(int(row.get('prediction', 0))),
            'prediction': int(row.get('prediction', 0)),
            'fraud_probability': float(row.get('fraud_probability', 0.0)),
            'legit_probability': float(row.get('legit_probability', 1.0)),
            'feature_importance': row.get('feature_importance', []),
            'shap_plot': shap_plot
        }

        return render_template('index.html', result=response_data, form_data=form_data)
    except Exception as e:
        tb = traceback.format_exc()
        print("Error in /predict:", tb)
        return render_template('index.html', error=f"Error processing request: {str(e)}", traceback=tb)


@app.route('/api/fraud/predict', methods=['POST'])
def fraud_predict_api():
    try:
        test_json = request.get_json()
        if not test_json:
            return jsonify({"error": "No JSON body provided"}), 400

        if isinstance(test_json, dict):
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            test_raw = pd.DataFrame(test_json)

        pipeline = Fraud()
        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.feature_engineering(df1)
        df3 = pipeline.data_preparation(df2)
        result = pipeline.get_prediction(model, test_raw, df3)

        return result.to_json(orient="records")
    except Exception as e:
        tb = traceback.format_exc()
        print("API error:", tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route('/health', methods=['GET'])
def health_check():
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": str(type(model)) if model is not None else "None",
        "shap_available": explainer is not None
    }
    return jsonify(status)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
