"""Generate a plain, reproducible PNG of the classical ML workflow."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT = Path("results/figures/ml_workflow_plain.png")


STEPS = [
    ("1. INPUT", "Sleep-EDF: 100 subjects, 457,652 epochs\n96 features | Wake, N1, N2, N3, REM"),
    ("2. SUBJECT-WISE SPLIT", "Train: 70 subjects | Calibration: 15 | Test: 15\nNo subject leakage"),
    ("3. TRAIN-ONLY GROUPKFOLD", "5 folds on the 70 training subjects\nAbout 56 train + 14 CV-validation subjects per fold"),
    ("4. MODELS + CLASS WEIGHT", "Random Forest | XGBoost | Logistic Regression\nFold-local imbalance handling"),
    ("5. CONSTRAINED TUNING", "Gap = train Macro-F1 - CV Macro-F1\nAccept configuration only when gap <= 0.10"),
    ("6. MODEL SELECTION", "RF: gap 0.185 -> rejected\nXGBoost: gap 0.067 -> selected over Logistic Regression"),
    ("7. IMBALANCE ABLATION", "XGBoost class weight vs XGBoost half-SMOTE\nHalf-SMOTE is applied inside training folds only"),
    ("8. SELECTED VARIANT", "Half-SMOTE CV Macro-F1: 0.755 | gap: 0.072\nClass-weight CV Macro-F1: 0.732 | gap: 0.067"),
    ("9. FINAL TRAINING", "Fit selected XGBoost on all 70 training subjects\nValidation and test remain untouched"),
    ("10. CALIBRATION", "Sigmoid / Platt scaling on 15 validation subjects\nStore raw and calibrated probabilities"),
    ("11. HELD-OUT TEST", "Evaluate once on 15 unseen test subjects\nMacro-F1 | Balanced accuracy | Kappa | ECE | Brier"),
    ("12. EXTERNAL VALIDATION", "Frozen model evaluated on HMC\nNo retuning and no recalibration"),
    ("13. REPORTING", "Confusion matrix | Per-class F1 | ROC | PR\nReliability diagram | SHAP"),
    ("14. FINAL ROBUSTNESS", "LOSO: 99 subjects train -> 1 subject test, repeated 100 times\nReport aggregate and subject-level stability"),
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 22))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_title("Classical ML Sleep-Staging Workflow", fontsize=22, weight="bold", pad=22)

    box_x, box_w, box_h = 0.08, 0.84, 0.052
    top, spacing = 0.94, 0.066
    for index, (title, body) in enumerate(STEPS):
        y = top - index * spacing
        box = FancyBboxPatch(
            (box_x, y - box_h), box_w, box_h,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            linewidth=1.4, edgecolor="#243b53", facecolor="#f7f9fc",
        )
        ax.add_patch(box)
        ax.text(box_x + 0.02, y - 0.014, title, fontsize=11.5, weight="bold",
                color="#102a43", va="center")
        ax.text(box_x + 0.02, y - 0.038, body, fontsize=9.8,
                color="#334e68", va="center")
        if index < len(STEPS) - 1:
            ax.annotate(
                "", xy=(0.5, y - spacing + 0.006), xytext=(0.5, y - box_h - 0.005),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#486581"},
            )

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
