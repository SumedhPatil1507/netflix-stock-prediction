import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs"


def shap_analysis(model, X, max_samples=300):
    """
    SHAP on the XGB base learner inside ManualStackingRegressor.
    """
    xgb_est = None
    for name, est in model.fitted_learners_:
        if 'xgb' in name.lower():
            xgb_est = est
            break

    if xgb_est is None:
        print("  SHAP: XGB estimator not found, skipping.")
        return

    # Data is already scaled inside the model; re-scale here for SHAP
    X_scaled = model.scaler.transform(np.array(X))
    n        = min(max_samples, len(X_scaled))
    idx      = np.random.choice(len(X_scaled), n, replace=False)
    X_sample = X_scaled[idx]

    explainer   = shap.TreeExplainer(xgb_est)
    shap_values = explainer.shap_values(X_sample)
    feat_names  = list(X.columns)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_summary.png", dpi=120, bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feat_names,
                      plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_bar.png", dpi=120, bbox_inches='tight')
    plt.close()

    print("  SHAP plots saved → outputs/shap_summary.png, shap_bar.png")
