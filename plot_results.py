import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


def plot_results(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    y_pred,
    baseline_feature_name,
    outcome_name,
):
    # Font size for all plots
    fs = 24
    xcolour = "grey"

    # Global settings for grey colour
    plt.rcParams["axes.edgecolor"] = xcolour
    plt.rcParams["axes.labelcolor"] = xcolour
    plt.rcParams["xtick.color"] = xcolour
    plt.rcParams["ytick.color"] = xcolour
    plt.rcParams["text.color"] = xcolour

    # Create a figure and a grid of subplots with the same dimensions
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

    # Scatter plot will be on the first subplot at (0, 0)
    ax[0].scatter(y_pred, y_test, s=600, alpha=0.5, c="pink")

    variance = np.var(y_pred)
    min_val = np.min(y_test)
    max_val = np.max(y_test)
    x_vals = np.linspace(min_val, max_val, 50)

    ax[0].plot(x_vals, x_vals, "-", c="#666666", alpha=0.4, lw=2)
    ax[0].plot(x_vals, x_vals + np.sqrt(variance), "-", c="#999999", alpha=0.3, lw=2)
    ax[0].plot(x_vals, x_vals - np.sqrt(variance), "-", c="#999999", alpha=0.3, lw=2)
    ax[0].set_title(r"Variance of Predictions: {:.2f}".format(variance), fontsize=fs)
    ax[0].set_ylabel(f"True {outcome_name}", fontsize=fs)
    ax[0].set_xlabel(f"Predicted {outcome_name}", fontsize=fs)
    ax[0].tick_params(axis="both", which="major", labelsize=fs)

    # Generate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X_train, y_train, cv=5, scoring="r2"
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax[1].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="lightblue",
    )
    ax[1].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="lightgreen",
    )
    ax[1].plot(
        train_sizes,
        train_scores_mean,
        "o-",
        ms=25,
        label="Training score",
        color="lightblue",
    )
    ax[1].plot(
        train_sizes,
        test_scores_mean,
        "o-",
        ms=25,
        label="Cross-validation score",
        color="lightgreen",
    )
    ax[1].set_xlabel("Training Size", fontsize=fs)
    ax[1].set_ylabel("Score", fontsize=fs)
    ax[1].tick_params(axis="both", which="major", labelsize=fs)
    ax[1].legend(loc="best", fontsize=fs)

    # Add labels to the subplots
    ax[0].text(
        -0.0, 1.1, "A", transform=ax[0].transAxes, fontsize=fs + 4, fontweight="bold"
    )
    ax[1].text(
        -0.0, 1.1, "B", transform=ax[1].transAxes, fontsize=fs + 4, fontweight="bold"
    )

    # Remove the frames, keeping only x and y axes
    for ax_sub in ax.flat:
        ax_sub.spines["top"].set_visible(False)
        ax_sub.spines["right"].set_visible(False)
        ax_sub.spines["left"].set_visible(True)
        ax_sub.spines["bottom"].set_visible(True)
        ax_sub.yaxis.set_ticks_position("left")
        ax_sub.xaxis.set_ticks_position("bottom")
        ax_sub.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax_sub.xaxis.set_major_locator(plt.MaxNLocator(4))
        # Adjust layout
        plt.tight_layout()

    # Save the whole figure as a PDF
    plt.savefig(f"paper/img/results_{outcome_name}.pdf", format="pdf")

    # Show the figure
    plt.show()
