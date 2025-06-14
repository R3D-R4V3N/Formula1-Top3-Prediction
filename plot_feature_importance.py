import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

# Create output directory
output_dir = "feature_imporance"
os.makedirs(output_dir, exist_ok=True)

# 1. Load the dataset
csv_path = "f1_data_2022_to_present.csv"
df = pd.read_csv(csv_path)

# 2. Create target
df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)

# 3. Prepare features and target
X = df.drop(columns=["finishing_position", "top3_flag"])
y = df["top3_flag"].values

# Identify categorical features
cat_features = ["circuit_id", "driver_id", "constructor_id"]
cat_feature_indices = [X.columns.get_loc(col) for col in cat_features]

# 4. Initialize CatBoost with tuned params
from model_catboost_final import MODEL_PARAMS
model = CatBoostClassifier(**MODEL_PARAMS)
model.fit(X, y, cat_features=cat_feature_indices)

# 5. Define importance types to plot
importance_types = ["FeatureImportance", "PredictionValuesChange", "LossFunctionChange", "ShapValues", "Interaction"]

# 6. Plot each importance type
for imp_type in importance_types:
    if imp_type == "ShapValues":
        shap_vals = model.get_feature_importance(type=imp_type, data=Pool(X, label=y), thread_count=-1)
        importances = np.mean(np.abs(shap_vals), axis=0)
    else:
        importances = model.get_feature_importance(type=imp_type)

    feat_names = X.columns.tolist()
    df_imp = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(df_imp['feature'], df_imp['importance'])
    plt.xlabel(f"Importance ({imp_type})")
    plt.title(f"CatBoost {imp_type} Feature Importances")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"feature_importances_{imp_type}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")
