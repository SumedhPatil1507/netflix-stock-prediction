import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "outputs"


def shap_analysis(pipeline, X, max_samples=500):
    """
    Run SHAP analysis on the GBR sub-model inside the VotingRegressor pipeline.
    Saves summary and bar plots to outputs/.
    """
    # Scale X the same way the pipeline does
    scaler = pipeline.named_steps['scaler']
    X_scaled = scaler.transform(X)

    # Pull the GradientBoosting estimator from the VotingRegressor
    gbr = pipeline.named_steps['model'].estimators_[0]

    # Subsample for speed
    n = min(max_samples, len(X_scaled))
    idx = np.random.choice(len(X_scaled), n, replace=False)
    X_sample = X_scaled[idx]

    explainer   = shap.TreeExplainer(gbr)
    shap_values = explainer.shap_values(X_sample)

    # Summary dot plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample,
                      feature_names=list(X.columns),
                      show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_summary.png", dpi=120, bbox_inches='tight')
    plt.close()

    # Bar plot (mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_values, X_sample,
                      feature_names=list(X.columns),
                      plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT}/shap_bar.png", dpi=120, bbox_inches='tight')
    plt.close()

    print("  SHAP plots saved → outputs/shap_summary.png, shap_bar.png")
