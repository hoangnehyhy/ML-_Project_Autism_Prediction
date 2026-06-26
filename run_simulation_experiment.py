import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# --- CUSTOM LOGISTIC REGRESSION ---
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.05, num_iterations=2000):
        self.lr = learning_rate
        self.iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in probs])

# --- SIMULATION FUNCTION WITH EXTREME NOISE INJECTION ---
def simulate_jama_features(df, target_col='Class/ASD', noisy=False, random_seed=42):
    np.random.seed(random_seed)
    n_samples = len(df)
    targets = df[target_col].values
    
    age_first_smile = np.zeros(n_samples)
    age_first_sentence = np.zeros(n_samples)
    eating_difficulty = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        is_asd = targets[i]
        
        if noisy:
            # EXTREME NOISY CASE: Milestones are almost identical in distribution (massive overlap)
            if is_asd == 1:
                # ASD: slight delay but huge overlap with typical children
                age_first_smile[i] = np.clip(np.random.normal(loc=2.2, scale=1.8), 0.5, 12.0)
                age_first_sentence[i] = np.clip(np.random.normal(loc=24.0, scale=10.0), 10.0, 60.0)
                # Eating difficulty has only 40% vs 30% separation
                eating_difficulty[i] = np.random.choice([0, 1], p=[0.60, 0.40])
            else:
                # Non-ASD
                age_first_smile[i] = np.clip(np.random.normal(loc=1.8, scale=1.2), 0.5, 6.0)
                age_first_sentence[i] = np.clip(np.random.normal(loc=20.0, scale=6.0), 10.0, 40.0)
                eating_difficulty[i] = np.random.choice([0, 1], p=[0.70, 0.30])
        else:
            # PERFECT CLEAN CASE: Clear separation
            if is_asd == 1:
                age_first_smile[i] = np.clip(np.random.normal(loc=3.2, scale=1.2), 0.5, 12.0)
                age_first_sentence[i] = np.clip(np.random.normal(loc=32.0, scale=8.0), 10.0, 60.0)
                eating_difficulty[i] = np.random.choice([0, 1], p=[0.35, 0.65])
            else:
                age_first_smile[i] = np.clip(np.random.normal(loc=1.5, scale=0.4), 0.5, 4.0)
                age_first_sentence[i] = np.clip(np.random.normal(loc=18.5, scale=2.5), 10.0, 30.0)
                eating_difficulty[i] = np.random.choice([0, 1], p=[0.85, 0.15])
            
    df_sim = df.copy()
    df_sim['age_first_smile'] = np.round(age_first_smile, 1)
    df_sim['age_first_sentence'] = np.round(age_first_sentence, 1)
    df_sim['eating_difficulty'] = eating_difficulty
    
    return df_sim

# --- EXPERIMENT PIPELINE ---
def run_experiment():
    print("Loading cleaned datasets...")
    train = pd.read_csv("data/processed/train_cleaned.csv")
    test = pd.read_csv("data/processed/test_cleaned.csv")
    
    target_col = 'Class/ASD'
    
    # 1. Simulate perfect & noisy features
    print("Generating simulated datasets with extreme noise...")
    train_clean = simulate_jama_features(train, target_col, noisy=False, random_seed=42)
    test_clean = simulate_jama_features(test, target_col, noisy=False, random_seed=43)
    
    train_noisy = simulate_jama_features(train, target_col, noisy=True, random_seed=44)
    test_noisy = simulate_jama_features(test, target_col, noisy=True, random_seed=45)
    
    # Save simulated datasets
    train_clean.to_csv("data/processed/train_simulated.csv", index=False)
    test_clean.to_csv("data/processed/test_simulated.csv", index=False)
    train_noisy.to_csv("data/processed/train_simulated_noisy.csv", index=False)
    test_noisy.to_csv("data/processed/test_simulated_noisy.csv", index=False)
    print("All simulated datasets saved to 'data/processed/'.")
    
    # 2. Extract arrays
    y_train = train[target_col].values
    y_test = test[target_col].values
    
    X_train_orig = train.drop(columns=[target_col]).values
    X_test_orig = test.drop(columns=[target_col]).values
    
    X_train_clean = train_clean.drop(columns=[target_col]).values
    X_test_clean = test_clean.drop(columns=[target_col]).values
    
    X_train_noisy = train_noisy.drop(columns=[target_col]).values
    X_test_noisy = test_noisy.drop(columns=[target_col]).values
    
    # 4. Only Milestones (No AQ-10) Case
    milestone_cols = ['age_first_smile', 'age_first_sentence', 'eating_difficulty']
    X_train_only_milestones = train_noisy[milestone_cols].values
    X_test_only_milestones = test_noisy[milestone_cols].values
    
    # 3. Model definitions
    def get_models():
        return {
            "Custom Logistic Regression": CustomLogisticRegression(learning_rate=0.05, num_iterations=2000),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, class_weight='balanced', random_state=42),
            "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
        }
    
    results = []
    
    datasets = {
        "1. Original (AQ-10)": (X_train_orig, X_test_orig),
        "2. Clean Simulated (+ Perfect Milestones)": (X_train_clean, X_test_clean),
        "3. High Noise Simulated (+ Extreme Noise)": (X_train_noisy, X_test_noisy),
        "4. Only Milestones (Extreme Noise / No AQ-10)": (X_train_only_milestones, X_test_only_milestones)
    }
    
    for ds_name, (X_tr, X_te) in datasets.items():
        print(f"\n--- Training on {ds_name} ---")
        models = get_models()
        for name, model in models.items():
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_te)
                if len(y_prob.shape) == 2: y_prob = y_prob[:, 1]
            else:
                y_prob = y_pred
                
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_prob)
            
            results.append({
                "Model": name,
                "Dataset": ds_name,
                "Accuracy": acc,
                "F1-Score": f1,
                "ROC-AUC": roc
            })
            print(f"[{name}] -> ROC-AUC: {roc:.4f}")
            
    # 6. Comparative Summary
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY LEADERBOARD WITH HIGH NOISE INJECTION")
    print("="*80)
    df_res = pd.DataFrame(results)
    df_pivot = df_res.pivot(index="Model", columns="Dataset", values=["Accuracy", "F1-Score", "ROC-AUC"])
    print(df_pivot.round(4).to_string())
    print("="*70)

if __name__ == "__main__":
    run_experiment()
