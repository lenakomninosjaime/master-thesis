import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from typing import Union

# Constants for data quality thresholds
CORRELATION_THRESHOLD = 0.7
VIF_PROBLEMATIC = 10
VIF_CONCERNING = 5
LIKERT_MIN = 1
LIKERT_MAX = 5
LABEL_MAX_CHARS = 30
SIGNIFICANCE_LEVEL = 0.05


def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess survey data from CSV.

    Drops comment columns and reverses Likert scale from 1-5 to 5-1.

    Args:
        file_path: Path to CSV file.

    Returns:
        Preprocessed DataFrame with inverted numeric columns.
    """
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Weitere Anmerkungen:"], errors="ignore")
    numeric_cols = [col for col in df.columns if col not in ["Grade", "Topic"]]
    df[numeric_cols] = df[numeric_cols] * -1 + 6
    return df


def join_datasets(
    df1: pd.DataFrame, df2: pd.DataFrame, on: Union[str, list], how: str = "inner"
) -> pd.DataFrame:
    """Join two DataFrames on specified columns.

    Args:
        df1: First DataFrame.
        df2: Second DataFrame.
        on: Column name(s) to join on.
        how: Join type ('inner', 'outer', 'left', 'right'). Defaults to 'inner'.

    Returns:
        Merged DataFrame.
    """
    return pd.merge(df1, df2, on=on, how=how)


def _prepare_regression_data(
    dataset: pd.DataFrame,
    independent_vars: Union[str, list[str]],
    dependent_var: str,
    add_constant: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Prepare and clean data for regression analysis.

    Args:
        dataset: Raw DataFrame.
        independent_vars: Variable name(s) to use as predictors.
        dependent_var: Target variable name.
        add_constant: Whether to add intercept (constant) term.

    Returns:
        Tuple of (X, y, sample_size) where X is design matrix and y is target vector.
    """
    vars_to_keep = (
        [independent_vars] if isinstance(independent_vars, str) else independent_vars
    )
    df = dataset.dropna(subset=vars_to_keep + [dependent_var]).reset_index(drop=True)

    x = df[vars_to_keep]
    if add_constant:
        x = sm.add_constant(x)
    y = df[dependent_var]

    return x, y, len(y)


def _print_regression_header(
    title: str, sample_size: int, outcome: str, predictors: list[str]
) -> None:
    """Print formatted regression analysis header."""
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"Sample size: {sample_size}")
    print(f"Outcome: {outcome}")
    print(f"Predictors: {', '.join(predictors)}\n")


def likert_log_regression(
    dataset: pd.DataFrame,
    independent_vars: Union[str, list[str]],
    dependent_vars: Union[str, list[str]],
):
    """Run ordinal logistic regression on Likert scale data.

    Fits proportional odds model. Falls back to binary logit if OrderedModel fails.
    For multiple dependent variables, recursively calls itself.

    Args:
        dataset: DataFrame containing variables.
        independent_vars: Predictor variable name(s).
        dependent_vars: Target variable name(s). Can be list or string.

    Returns:
        Fitted model object (OrderedModel or Logit) when dependent_vars is string,
        None when dependent_vars is list (recursive calls).
    """
    independent_vars = (
        [independent_vars] if isinstance(independent_vars, str) else independent_vars
    )

    if isinstance(dependent_vars, list):
        for dep_var in dependent_vars:
            likert_log_regression(dataset, independent_vars, dep_var)
        return None
    else:
        x, y, _ = _prepare_regression_data(
            dataset, independent_vars, dependent_vars, add_constant=False
        )

        try:
            res = OrderedModel(y, x, distr="logit").fit(disp=False)
            return res
        except Exception as e:
            print(f"OrderedModel failed ({e}). Falling back to binary logit...")
            x_const = sm.add_constant(x)
            res = sm.Logit(y, x_const).fit(disp=False)
            return res


def _build_color_and_label_maps(
    items: list[str],
    palette: str,
    y_labels: dict = None,
    color_indices: list = None,
) -> tuple[dict, dict]:
    """Build color and label maps for visualization.

    Maps items to colors from a palette and to display labels.

    Args:
        items: List of item names to map.
        palette: Seaborn palette name.
        y_labels: Dict mapping items to tuples of (display_label, color_index).
        color_indices: List of color indices overriding y_labels indices.

    Returns:
        Tuple of (color_map, label_map) dicts.
    """
    # Determine max color index needed
    max_idx = 0
    if color_indices:
        max_idx = max(color_indices)
    elif y_labels:
        for item in items:
            if (
                item in y_labels
                and len(y_labels[item]) > 1
                and isinstance(y_labels[item][1], int)
            ):
                max_idx = max(max_idx, y_labels[item][1])

    # Get palette colors
    palette_colors = sns.color_palette(palette, max(max_idx + 1, len(items)))

    # Build maps
    color_map = {}
    label_map = {}
    for i, item in enumerate(items):
        # Determine color index
        if color_indices and i < len(color_indices):
            color_idx = color_indices[i]
        elif (
            y_labels
            and item in y_labels
            and len(y_labels[item]) > 1
            and isinstance(y_labels[item][1], int)
        ):
            color_idx = y_labels[item][1]
        else:
            color_idx = i

        color_map[item] = palette_colors[color_idx]
        label_map[item] = y_labels.get(item, (item,))[0] if y_labels else item

    return color_map, label_map


def create_boxplot(
    dataset: pd.DataFrame,
    y: Union[str, list[str]] = None,
    hue: str = None,
    title: str = None,
    palette: str = "Set2",
    x_labels: dict = None,
    y_labels: dict = None,
    x_order: list = None,
    size=None,
) -> None:
    """Create boxplot(s) for Likert scale responses.

    Multiple y columns are melted into a single plot. Without x_labels, creates
    individual boxplots for each y-label showing the entire distribution.

    Args:
        dataset: DataFrame containing response data.
        y: Column name(s) to plot. If None, uses keys from y_labels.
        hue: Column name for grouping colors (optional).
        title: Plot title.
        palette: Seaborn palette name (default: "Set2").
        x_labels: Dict like {'Klasse': 'Class'} defining x-axis column and label.
        y_labels: Dict like {'col': ('Display Label', color_idx)} for custom labels.
        x_order: Order of x-axis categories (optional).
    """
    if y is None:
        y = list(y_labels.keys()) if y_labels else []

    # Normalize y to list
    y = [y] if isinstance(y, str) else y

    # Build color and label maps
    color_map, label_map = _build_color_and_label_maps(y, palette, y_labels)

    # Case 1: With x-axis grouping
    if x_labels:
        x = list(x_labels.keys())[0]
        # Melt data
        id_vars = [x] + ([hue] if hue else [])
        melted = dataset[id_vars + y].melt(
            id_vars=id_vars, var_name="Question", value_name="Response"
        )

        # Apply display labels
        melted["Question"] = melted["Question"].map(lambda q: label_map.get(q, q))

        # Plot
        x_order = x_order or sorted(melted[x].unique())
        figsize = (10, 5) if len(y) > 1 else (10, 6)
        figsize = size if size is not None else figsize
        plt.figure(figsize=figsize)
        sns.boxplot(
            x=x,
            y="Response",
            data=melted,
            hue="Question",
            palette=color_map,
            order=x_order,
            showmeans=True,
            medianprops=dict(color="red", linewidth=2),
            meanprops={
                "marker": "d",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )

        plt.title(title)
        plt.xlabel(x_labels[x])
        plt.yticks([1, 2, 3, 4, 5])
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, frameon=True)
        plt.tight_layout()
        plt.show()

    # Case 2: Without x-axis grouping - one boxplot per y-label
    else:
        fig, axes = plt.subplots(1, len(y), figsize=(4 * len(y), 5))
        if len(y) == 1:
            axes = [axes]

        for idx, col in enumerate(y):
            data_to_plot = dataset[col].dropna()
            box_color = color_map[col]

            # Create boxplot
            axes[idx].boxplot(
                data_to_plot,
                widths=0.5,
                patch_artist=True,
                showmeans=True,
                boxprops=dict(facecolor=box_color),
                medianprops=dict(color="red", linewidth=2),
                meanprops={
                    "marker": "d",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                    "markeredgewidth": 1.5,
                    "markersize": 8,
                },
            )

            axes[idx].set_ylabel("Response")
            axes[idx].set_xticklabels([""])
            axes[idx].set_xticks([])
            axes[idx].set_ylim(1, 5)
            axes[idx].set_yticks([1, 2, 3, 4, 5])

        fig.suptitle(title or "Response Distributions", fontsize=14, y=1.02)
        # Create legend with y labels
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=color_map[col],
                markersize=8,
                label=label_map[col],
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
            for col in y
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=1,
            frameon=True,
        )
        plt.tight_layout()
        plt.show()


def separate_points_plot(
    dataset: pd.DataFrame,
    questions: list[str],
    title: str = None,
    palette: str = "Set2",
    y_labels: dict = None,
    color_indices: list = None,
    size=None,
) -> None:
    """Plot individual respondent responses across multiple questions.

    Each respondent gets an x-position. Questions are plotted with different colors
    and connected by arrows showing response progression. Respondents are sorted by
    the first and second question for meaningful ordering.

    Args:
        dataset: DataFrame containing response data.
        questions: Column names (questions) to plot.
        title: Plot title.
        palette: Seaborn palette name (default: "Set2").
        y_labels: Dict like {'col': ('Display Label', color_idx)} for custom labels.
        color_indices: List of color indices overriding y_labels indices.
    """
    # Sort by first and second question
    sort_cols = questions[:2] if len(questions) > 1 else questions[:1]
    df_sorted = dataset.sort_values(by=sort_cols, na_position="last").reset_index(
        drop=True
    )

    # Build color and label maps
    color_map, label_map = _build_color_and_label_maps(
        questions, palette, y_labels, color_indices
    )

    plt.figure(figsize=(10, 4) if size is None else size)

    # For each row (teacher/respondent)
    for row_idx, (_, row) in enumerate(df_sorted.iterrows()):
        responses = [(q, row[q]) for q in questions if pd.notna(row[q])]
        if not responses:
            continue

        x_pos = row_idx + 1
        x_coords, y_coords = [], []

        # Plot points
        for q, value in responses:
            x_coords.append(x_pos)
            y_coords.append(value)
            plt.scatter(
                x_pos,
                value,
                s=100,
                color=color_map[q],
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
                zorder=2,
            )

        # Draw arrows connecting points
        for i in range(len(x_coords) - 1):
            arrow = FancyArrowPatch(
                (x_coords[i], y_coords[i]),
                (x_coords[i + 1], y_coords[i + 1]),
                arrowstyle="-",
                mutation_scale=25,
                linewidth=1.2,
                color="black",
                alpha=0.4,
                zorder=1,
            )
            plt.gca().add_patch(arrow)

    # Configure plot
    plt.ylim(0.5, 5.5)
    plt.ylabel("Responses")
    plt.yticks([1, 2, 3, 4, 5])
    plt.xticks([])

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[q],
            markersize=8,
            label=label_map[q],
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for q in questions
    ]
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=1,
        frameon=True,
    )

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def _shorten_label(label: str, max_chars: int = LABEL_MAX_CHARS) -> str:
    """Add line breaks to long labels for better readability.

    Args:
        label: Text to shorten.
        max_chars: Maximum characters per line.

    Returns:
        Shortened label with newline breaks.
    """
    if len(label) <= max_chars:
        return label
    words = label.split()
    lines = []
    current_line = []
    for word in words:
        if sum(len(w) for w in current_line) + len(word) + len(current_line) > max_chars:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    lines.append(" ".join(current_line))
    return "\n".join(lines)


def assess_data_quality(
    dataset: pd.DataFrame,
    columns: Union[str, list[str]] = None,
    show_correlation_heatmap: bool = True,
) -> None:
    """Comprehensive data quality assessment for regression analysis.

    Evaluates multicollinearity through correlation matrix and VIF (Variance
    Inflation Factor). High correlations (|r| > 0.7) and VIF > 10 indicate problems.

    Args:
        dataset: DataFrame to assess.
        columns: Numeric columns to assess. If None, auto-detects numeric columns.
        show_correlation_heatmap: Whether to display correlation heatmap.
    """
    # Select columns
    if columns is None:
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [columns] if isinstance(columns, str) else columns

    df = dataset[numeric_cols].copy()
    df_clean = df.dropna()
    corr_matrix = df_clean.corr()

    # CORRELATION MATRIX REPORT
    print(f"\n{'-'*80}")
    print("CORRELATION MATRIX - High Correlations (|r| > 0.7)")
    print(f"{'-'*80}")

    # Apply shortened labels to correlation matrix for heatmap
    shortened_cols = [_shorten_label(col) for col in corr_matrix.columns]
    corr_matrix_display = corr_matrix.copy()
    corr_matrix_display.columns = shortened_cols
    corr_matrix_display.index = shortened_cols

    # Create heatmap
    set2_colors = sns.color_palette("Set2")
    cmap = sns.light_palette(set2_colors[1], as_cmap=True)

    if show_correlation_heatmap:
        plt.figure(figsize=(9, 8))
        sns.heatmap(
            corr_matrix_display,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            square=True,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Correlation Matrix Heatmap")
        plt.tight_layout()
        plt.show()

    # Find and report high correlations
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > CORRELATION_THRESHOLD:
                high_corr.append({
                    "Variable 1": corr_matrix.columns[i],
                    "Variable 2": corr_matrix.columns[j],
                    "Correlation": corr_matrix.iloc[i, j],
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        print(high_corr_df.to_string(index=False))
    else:
        print("None found - Good! Low multicollinearity ✓")

    # VARIANCE INFLATION FACTOR (VIF)
    print(f"\n{'-'*80}")
    print("MULTICOLLINEARITY: VARIANCE INFLATION FACTOR (VIF)")
    print(f"{'-'*80}")
    print(f"VIF > {VIF_PROBLEMATIC}: Problematic ✗")
    print(f"VIF {VIF_CONCERNING}-{VIF_PROBLEMATIC}: Concerning ⚠")
    print(f"VIF < {VIF_CONCERNING}: Acceptable ✓\n")

    X = df_clean[numeric_cols]
    vif_data = pd.DataFrame({
        "Column": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    vif_data = vif_data.sort_values("VIF", ascending=False)
    vif_data["Status"] = vif_data["VIF"].apply(
        lambda x: (
            "✓ OK"
            if x < VIF_CONCERNING
            else ("⚠ Concerning" if x < VIF_PROBLEMATIC else "✗ Problematic")
        )
    )
    print(vif_data.to_string(index=False))


def test_proportional_odds_assumption(
    dataset: pd.DataFrame, predictor: Union[str, list[str]], outcome: str
) -> None:
    """Test the Proportional Odds (PO) Assumption using Brant Test.

    Tests whether the effect of predictors is consistent across all outcome
    category thresholds. Violation suggests using multinomial logistic regression
    instead of ordinal logistic regression.

    The test compares:
    - Ordinal Logistic Regression (constrained, assumes PO)
    - Multinomial Logistic Regression (unconstrained, allows different effects)

    Args:
        dataset: DataFrame containing predictors and outcome.
        predictor: Predictor variable name(s) to test.
        outcome: Ordinal outcome variable name.
    """
    predictor_list = [predictor] if isinstance(predictor, str) else predictor

    # Clean data
    df = dataset.dropna(subset=predictor_list + [outcome]).reset_index(drop=True)

    print(f"\n{'='*80}")
    print("PROPORTIONAL ODDS (PO) ASSUMPTION TEST - Brant Test")
    print(f"{'='*80}")
    print(f"Outcome: {outcome}")
    print(f"Predictor(s): {', '.join(predictor_list)}")
    print(f"Sample size: {len(df)}\n")

    X = df[predictor_list]
    y = df[outcome]

    # Fit ordinal logistic regression (constrained model - assumes PO)
    print(f"{'-'*80}")
    print("1. ORDINAL LOGISTIC REGRESSION (Constrained - PO Assumption)")
    print(f"{'-'*80}")

    try:
        ordinal_model = OrderedModel(y, X, distr="logit").fit(disp=False)
        print(f"Log-Likelihood: {ordinal_model.llf:.4f}")
        print(f"AIC: {ordinal_model.aic:.4f}")
        print(f"BIC: {ordinal_model.bic:.4f}\n")
        ordinal_ll = ordinal_model.llf
    except Exception as e:
        print(f"Error fitting ordinal model: {e}")
        return

    # Fit multinomial logistic regression (unconstrained model)
    print(f"{'-'*80}")
    print("2. MULTINOMIAL LOGISTIC REGRESSION (Unconstrained - No PO)")
    print(f"{'-'*80}")

    X_const = sm.add_constant(X)
    try:
        multinomial_model = sm.MNLogit(y, X_const).fit(disp=False)
        print(f"Log-Likelihood: {multinomial_model.llf:.4f}")
        print(f"AIC: {multinomial_model.aic:.4f}")
        print(f"BIC: {multinomial_model.bic:.4f}\n")
        multinomial_ll = multinomial_model.llf
    except Exception as e:
        print(f"Error fitting multinomial model: {e}")
        return

    # Likelihood Ratio Test (Brant Test)
    print(f"{'-'*80}")
    print("3. LIKELIHOOD RATIO TEST (Brant Test)")
    print(f"{'-'*80}")

    lr_statistic = -2 * (ordinal_ll - multinomial_ll)

    # Calculate degrees of freedom: p * (K - 2)
    # where p = number of predictors, K = number of outcome categories
    n_categories = len(np.unique(y))
    n_predictors = len(predictor_list)
    df_test = n_predictors * (n_categories - 2)

    # Calculate p-value from chi-square distribution
    if df_test > 0 and lr_statistic >= 0:
        p_value = 1 - chi2.cdf(lr_statistic, df_test)
    else:
        p_value = 1.0

    print(f"Null Hypothesis (H₀): Proportional Odds Assumption HOLDS")
    print(f"Alternative (H₁): Proportional Odds Assumption is VIOLATED\n")
    print(f"Likelihood Ratio Statistic: {lr_statistic:.4f}")
    print(f"Degrees of Freedom: {df_test}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significance Level: α = {SIGNIFICANCE_LEVEL}\n")

    # INTERPRETATION
    print(f"{'-'*80}")
    print("INTERPRETATION:")
    print(f"{'-'*80}")

    if p_value > SIGNIFICANCE_LEVEL:
        print(f"✓ P-value ({p_value:.4f}) > {SIGNIFICANCE_LEVEL}")
        print(f"✓ FAIL TO REJECT null hypothesis")
        print(f"✓ The PROPORTIONAL ODDS ASSUMPTION HOLDS")
        print(f"✓ Ordinal logistic regression is appropriate ✓")
        print(f"\nRecommendation: Use ordinal logistic regression results.")
    else:
        print(f"✗ P-value ({p_value:.4f}) < {SIGNIFICANCE_LEVEL}")
        print(f"✗ REJECT null hypothesis")
        print(f"✗ The PROPORTIONAL ODDS ASSUMPTION is VIOLATED")
        print(f"✗ Coefficients may differ across outcome categories")
        print(f"\nRecommendation: Consider multinomial logistic regression instead,")
        print(f"                or report ordinal results with caution.")

    print(f"\n{'='*80}")
