#!/usr/bin/env python3
import logging

import signalplot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

"""
Generate visualizations for Flow Assurance Risk Prediction blog post.
Uses minimalist styling with serif fonts, clean axes, and high-quality output.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def generate_pipeline_telemetry(n_segments=4000, seed=77):
    """Generate realistic pipeline segment telemetry data."""
    rng = np.random.default_rng(seed)

    crude = rng.choice(
        ["light_sweet", "medium", "heavy_waxy"], n_segments, p=[0.5, 0.35, 0.15]
    )
    wat_celsius = np.where(
        crude == "heavy_waxy",
        rng.normal(35, 2, n_segments),
        rng.normal(27, 2, n_segments),
    )
    temp_in_celsius = rng.normal(32, 4, n_segments)
    temp_out_celsius = temp_in_celsius - rng.normal(3, 2, n_segments).clip(0.5, 10)
    pressure_bar = rng.normal(55, 6, n_segments)
    flow_ksm3h = rng.normal(2.0, 0.6, n_segments).clip(0.3, 4.0)
    shear_proxy = 0.3 * flow_ksm3h / (pressure_bar / 50)
    inhibitor_active = rng.choice([0, 1], n_segments, p=[0.7, 0.3])
    water_cut = rng.beta(2, 10, n_segments)

    avg_temp = 0.5 * (temp_in_celsius + temp_out_celsius)
    thermal_margin = wat_celsius - avg_temp

    base_logit = (
        0.8 * (thermal_margin > 0)
        + 0.4 * (thermal_margin > 2)
        + 0.3 * (water_cut > 0.2)
        + 0.3 * (crude == "heavy_waxy")
        - 0.4 * inhibitor_active
        - 0.3 * (shear_proxy > 0.9)
    )

    risk_probability = 1 / (1 + np.exp(-(base_logit - 0.4)))
    risk_observed = (rng.random(n_segments) < risk_probability).astype(int)

    return pd.DataFrame(
        {
            "crude_type": crude,
            "wat_celsius": wat_celsius,
            "temp_in_celsius": temp_in_celsius,
            "temp_out_celsius": temp_out_celsius,
            "pressure_bar": pressure_bar,
            "flow_ksm3h": flow_ksm3h,
            "shear_proxy": shear_proxy,
            "inhibitor_active": inhibitor_active,
            "water_cut": water_cut,
            "risk_observed": risk_observed,
        }
    )


def train_simple_model(data):
    """Train simplified model for visualization."""
    X = data.drop(columns=["risk_observed"])
    y = data["risk_observed"]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(drop="first", sparse_output=False),
                ["crude_type"],
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=400, random_state=3, n_jobs=-1),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    return X_test, y_test, y_proba


def create_main_visualization(plot: bool = False):
    """Create main flow assurance risk prediction visualization."""
    np.random.seed(77)

    # Generate data and train model
    data = generate_pipeline_telemetry(n_segments=4000)
    X_test, y_test, y_proba = train_simple_model(data)

    # Calculate thermal margin
    thermal_margin = X_test["wat_celsius"] - 0.5 * (
        X_test["temp_in_celsius"] + X_test["temp_out_celsius"]
    )

    # Create figure with three panels
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

        # Panel 1: Risk vs Thermal Margin
        ax1.scatter(
            thermal_margin,
            y_proba,
            s=20,
            alpha=0.5,
            color="white",
            edgecolors="black",
            linewidths=0.5,
        )

        # Add risk threshold line
        ax1.axhline(
            y=0.5, color="gray", linestyle="--", linewidth=1, label="50% Risk Threshold"
        )

        # Add zero margin line
        ax1.axvline(
            x=0, color="gray", linestyle=":", linewidth=1, label="Zero Thermal Margin"
        )

        # Apply minimalist style
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_position(("outward", 5))
        ax1.spines["bottom"].set_position(("outward", 5))
        ax1.set_title(
            "Risk Probability vs Thermal Margin",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax1.set_xlabel("Thermal Margin (WAT - Avg Temp, °C)", fontsize=10)
        ax1.set_ylabel("Risk Probability", fontsize=10)
        ax1.legend(loc="upper right", frameon=False, fontsize=9)
        ax1.set_ylim(-0.05, 1.05)

        # Panel 2: Risk vs Flow Rate
        ax2.scatter(
            X_test["flow_ksm3h"],
            y_proba,
            s=20,
            alpha=0.5,
            color="white",
            edgecolors="black",
            linewidths=0.5,
        )

        # Add risk threshold
        ax2.axhline(
            y=0.5, color="gray", linestyle="--", linewidth=1, label="50% Risk Threshold"
        )

        # Apply minimalist style
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_position(("outward", 5))
        ax2.spines["bottom"].set_position(("outward", 5))
        ax2.set_title(
            "Risk Probability vs Flow Rate", fontsize=12, fontweight="bold", loc="left"
        )
        ax2.set_xlabel("Flow Rate (kSm³/h)", fontsize=10)
        ax2.set_ylabel("Risk Probability", fontsize=10)
        ax2.legend(loc="upper right", frameon=False, fontsize=9)
        ax2.set_ylim(-0.05, 1.05)

        # Panel 3: Risk vs Water Cut
        ax3.scatter(
            X_test["water_cut"],
            y_proba,
            s=20,
            alpha=0.5,
            color="white",
            edgecolors="black",
            linewidths=0.5,
        )

        # Add risk threshold
        ax3.axhline(
            y=0.5, color="gray", linestyle="--", linewidth=1, label="50% Risk Threshold"
        )

        # Add high water cut line
        ax3.axvline(
            x=0.2, color="gray", linestyle=":", linewidth=1, label="High Water Cut"
        )

        # Apply minimalist style
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_position(("outward", 5))
        ax3.spines["bottom"].set_position(("outward", 5))
        ax3.set_title(
            "Risk Probability vs Water Cut", fontsize=12, fontweight="bold", loc="left"
        )
        ax3.set_xlabel("Water Cut (fraction)", fontsize=10)
        ax3.set_ylabel("Risk Probability", fontsize=10)
        ax3.legend(loc="upper right", frameon=False, fontsize=9)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlim(-0.05, 1.05)

        # Save
        signalplot.save("09_flow_assurance_main.png")
    logger.info("✓ Created: 09_flow_assurance_main.png")


def create_accuracy_visualization(plot: bool = False):
    """Create model performance visualization."""
    np.random.seed(77)

    # Generate data and train model
    data = generate_pipeline_telemetry(n_segments=4000)
    X_test, y_test, y_proba = train_simple_model(data)

    # Create figure with two panels
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Panel 1: ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        ax1.plot(
            fpr,
            tpr,
            color="black",
            linewidth=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})",
        )
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

        # Add optimal threshold point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ax1.scatter(
            [fpr[optimal_idx]],
            [tpr[optimal_idx]],
            s=100,
            color="white",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label=f"Optimal Threshold ({optimal_threshold:.2f})",
        )

        # Apply minimalist style
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_position(("outward", 5))
        ax1.spines["bottom"].set_position(("outward", 5))
        ax1.set_title(
            "ROC Curve: Model Discrimination",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax1.set_xlabel("False Positive Rate", fontsize=10)
        ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
        ax1.legend(loc="lower right", frameon=False, fontsize=9)
        ax1.set_xlim(-0.02, 1.02)
        ax1.set_ylim(-0.02, 1.02)
        ax1.set_aspect("equal")

        # Panel 2: Predicted Probability Distribution
        risk_events = y_proba[y_test == 1]
        safe_segments = y_proba[y_test == 0]

        bins = np.linspace(0, 1, 25)

        ax2.hist(
            safe_segments,
            bins=bins,
            alpha=0.5,
            color="white",
            edgecolor="black",
            linewidth=1.5,
            label="Safe Segments",
        )
        ax2.hist(
            risk_events,
            bins=bins,
            alpha=0.5,
            color="gray",
            edgecolor="black",
            linewidth=1.5,
            label="Risk Events",
        )

        # Add threshold line
        ax2.axvline(
            x=0.5,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Decision Threshold",
        )

        # Apply minimalist style
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_position(("outward", 5))
        ax2.spines["bottom"].set_position(("outward", 5))
        ax2.set_title(
            "Predicted Probability Distribution",
            fontsize=12,
            fontweight="bold",
            loc="left",
        )
        ax2.set_xlabel("Predicted Risk Probability", fontsize=10)
        ax2.set_ylabel("Frequency", fontsize=10)
        ax2.legend(loc="upper right", frameon=False, fontsize=9)

        # Save
        signalplot.save("09_flow_assurance_accuracy.png")
    logger.info("✓ Created: 09_flow_assurance_accuracy.png")


def main():
    """Generate all visualizations."""
    signalplot.apply(font_family="serif")
    logger.info("FLOW ASSURANCE RISK PREDICTION - VISUALIZATION GENERATION")
    logger.info()

    # Set serif font globally

    logger.info("Creating visualizations...")
    create_main_visualization()
    create_accuracy_visualization()

    logger.info()
    logger.info("All visualizations created successfully!")


if __name__ == "__main__":
    main()
