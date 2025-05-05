import os
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt

from medaidml import NO_DATALEAK_NAME, TEST_NAME, RESULTS_DIR
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score

def get_args():
    parser = argparse.ArgumentParser(description="Train a model using encoder-based methods.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mdeberta-v3-base",
                 "xlm-roberta-base",
                 "bert-base-multilingual-cased",
                 "baseline",
                 "gpt2",
                 "fast_detect_gpt",
                 "Medical-mT5-large"],
        help="Name of the model from with the results were generated."
    )
    return parser.parse_args()

def load_data_in_dir(dir_path: str) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    dataframes = []
    for dir_name in os.listdir(dir_path):
        test_df = pd.read_csv(os.path.join(dir_path, dir_name, TEST_NAME))
        no_dataleak_df = pd.read_csv(os.path.join(dir_path, dir_name, NO_DATALEAK_NAME))
        dataframes.append((dir_name, test_df, no_dataleak_df))
    return dataframes

def calculate_metrics(dataframes: List[Tuple[str, pd.DataFrame, pd.DataFrame]]) -> List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]:
    metrics = []
    for seed, test_df, no_dataleak_df in dataframes:
        test_accuracy = accuracy_score(test_df["Ground Truth"], test_df["Prediction"])
        test_precision = precision_score(test_df["Ground Truth"], test_df["Prediction"], average='weighted', zero_division=0)
        test_f1 = f1_score(test_df["Ground Truth"], test_df["Prediction"], average='weighted', zero_division=0)

        no_dataleak_accuracy = accuracy_score(no_dataleak_df["Ground Truth"], no_dataleak_df["Prediction"])
        no_dataleak_precision = precision_score(no_dataleak_df["Ground Truth"], no_dataleak_df["Prediction"], average='weighted', zero_division=0)
        no_dataleak_f1 = f1_score(no_dataleak_df["Ground Truth"], no_dataleak_df["Prediction"], average='weighted', zero_division=0)

        metrics.append((seed, (test_accuracy, test_precision, test_f1), (no_dataleak_accuracy, no_dataleak_precision, no_dataleak_f1)))
    return metrics

def plot_boxplot_metrics(metrics: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]) -> None:
    test_values = [list(m[1]) for m in metrics]
    leakfree_values = [list(m[2]) for m in metrics]

    test_per_metric = list(zip(*test_values))
    leakfree_per_metric = list(zip(*leakfree_values))
    
    fig, axes = plt.subplots(3, 2, figsize=(5, 12), sharey='row')
    if MODEL == "baseline":
        title = "Baseline"
    elif MODEL == "gpt2":
        title = "GPT-2"
    elif MODEL == "fast_detect_gpt":
        title = "Fast-DetectGPT"
    elif MODEL == "mdeberta-v3-base":
        title = "mDeBERTa v3"
    elif MODEL == "xlm-roberta-base":
        title = "XLM-RoBERTa"
    elif MODEL == "bert-base-multilingual-cased":
        title = "Multilingual BERT"
    elif MODEL == "Medical-mT5-large":
        title = "Medical mT5"
    else:
        title = MODEL
    fig.suptitle(title, fontsize=27, y=0.99)
    
    metric_names = ["Accuracy", "Precision", "F1 Score"]
    box_colors = ["#1f77b4", "#ff7f0e"]

    for i in range(3):
        combined_values = test_per_metric[i] + leakfree_per_metric[i]
        min_val = min(combined_values)
        max_val = max(combined_values)
        padding = 0.0125
        y_min = max(0.0, min_val - padding)
        y_max = min(1.0, max_val + padding)

        ax_test = axes[i, 0]
        ax_test.boxplot(
            test_per_metric[i],
            patch_artist=True,
            widths=0.8,
            boxprops=dict(facecolor=box_colors[0], linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(color='black', linewidth=1.5),
        )
        if i == 0:
            ax_test.set_title("Validation", fontsize=22)
        ax_test.set_ylim(y_min, y_max)
        ax_test.set_ylabel(metric_names[i], fontsize=18)

        # No Data Leak plot
        ax_noleak = axes[i, 1]
        ax_noleak.boxplot(
            leakfree_per_metric[i],
            patch_artist=True,
            widths=0.7,
            boxprops=dict(facecolor=box_colors[1], linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(color='black', linewidth=1.5),
        )
        if i == 0:
            ax_noleak.set_title("Test", fontsize=22)
            ax_noleak.set_ylim(y_min, y_max)

        # Clean look
        for ax in (ax_test, ax_noleak):
            ax.tick_params(axis='y', labelsize=18)
            ax.tick_params(axis='x', bottom=False, labelbottom=False)
            #ax.set_yticklabels([f"{tick:.3f}" for tick in ax.get_yticks()])

    plt.tight_layout()
    plt.show()
    
def calculate_metrics_for_attribute(dataframes: List[Tuple[str, pd.DataFrame, pd.DataFrame]],
                                    attribute: str,
                                    attribute_name: str,
                                    *,
                                    concat = False) -> List[Tuple[str, str, float]]:
    metrics = []
    for seed, test_df, no_dataleak_df in dataframes:
        if concat:
            combined_df = pd.concat([test_df, no_dataleak_df], axis=0)
        else:
            combined_df = test_df
        if attribute == "language":
            restricted_df = combined_df[combined_df["language"] == attribute_name]
        elif attribute == "source": # for models
            restricted_df = combined_df[combined_df["source"] == attribute_name]
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
        
        overall_accuracy = accuracy_score(restricted_df["Ground Truth"], restricted_df["Prediction"])

        metrics.append((seed, attribute, overall_accuracy))
    return metrics
    
def print_metrics(metrics: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]]) -> None:
    for seed, test_metrics, no_dataleak_metrics in metrics:
        print(f"Seed: {seed}")
        print(f"Test Metrics: Accuracy: {test_metrics[0]:.4f}, Precision: {test_metrics[1]:.4f}, F1 Score: {test_metrics[2]:.4f}")
        print(f"No Data Leak Metrics: Accuracy: {no_dataleak_metrics[0]:.4f}, Precision: {no_dataleak_metrics[1]:.4f}, F1 Score: {no_dataleak_metrics[2]:.4f}")
        print("-" * 50)
    print("Overall Metrics:")
    print(f"Test Accuracy: {sum(m[1][0] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[1][0] for m in metrics]).std():.4f}")
    print(f"Test Precision: {sum(m[1][1] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[1][1] for m in metrics]).std():.4f}")
    print(f"Test Recall: {sum(m[1][2] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[1][2] for m in metrics]).std():.4f}")
    print(f"Test F1 Score: {sum(m[1][2] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[1][2] for m in metrics]).std():.4f}")
    print(f"No Data Leak Accuracy: {sum(m[2][0] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[2][0] for m in metrics]).std():.4f}")
    print(f"No Data Leak Precision: {sum(m[2][1] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[2][1] for m in metrics]).std():.4f}")
    print(f"No Data Leak Recall: {sum(m[2][2] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[2][2] for m in metrics]).std():.4f}")
    print(f"No Data Leak F1 Score: {sum(m[2][2] for m in metrics) / len(metrics):.4f}, sd: {pd.Series([m[2][2] for m in metrics]).std():.4f}")

def print_metrics_for_attribute(metrics: List[Tuple[str, str, float]]) -> None:
    for seed, _, metric in metrics:
        print(f"Seed: {seed}")
        print(f"Accuracy: {metric:.4f}")
        print("-" * 50)
        
def plot_barchart_metrics_for_language(metrics: List[Tuple[str, List[Tuple[str, str, float]]]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    languages = [lang for lang, _ in metrics]
    means = [sum(m[2] for m in lang_metrics) / len(lang_metrics) for _, lang_metrics in metrics]
    std_devs = [pd.Series([m[2] for m in lang_metrics]).std() for _, lang_metrics in metrics]

    ax.bar(languages, means, yerr=std_devs, capsize=5, color="#1f77b4", alpha=0.7, error_kw=dict(ecolor='black', elinewidth=1.5, capthick=1.5))
    ax.set_xlabel("Language", fontsize=22)
    ax.set_ylabel("Accuracy", fontsize=22)
    if MODEL == "baseline":
        title = "Baseline"
    elif MODEL == "gpt2":
        title = "GPT-2"
    elif MODEL == "fast_detect_gpt":
        title = "Fast-DetectGPT"
    elif MODEL == "mdeberta-v3-base":
        title = "mDeBERTa v3"
    elif MODEL == "xlm-roberta-base":
        title = "XLM-RoBERTa"
    elif MODEL == "bert-base-multilingual-cased":
        title = "BERT Multilingual"
    elif MODEL == "Medical-mT5-large":
        title = "Medical mT5"
    else:
        title = MODEL
    ax.set_title(title, fontsize=27)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0.5, 0.9)
    plt.tight_layout()
    plt.show()

def plot_barchart_metrics_for_llm(metrics: List[Tuple[str, str, float]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    llms = [llm for llm, _ in metrics]
    means = [sum(m[2] for m in llm_metrics) / len(llm_metrics) for _, llm_metrics in metrics]
    std_devs = [pd.Series([m[2] for m in llm_metrics]).std() for _, llm_metrics in metrics]

    ax.bar(llms, means, yerr=std_devs, capsize=5, color="#ff6347", alpha=0.7, error_kw=dict(ecolor='black', elinewidth=1.5, capthick=1.5))
    ax.set_xlabel("Source", fontsize=22)
    ax.set_ylabel("Accuracy", fontsize=22)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0.2, 1.0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    args = get_args()
    
    MODEL = args.model
    MAIN_PATH = os.path.join(RESULTS_DIR, MODEL)
    LANGUAGES = ["en", "de", "es", "fr"]
    LLM_NAMES = ["HUMAN", "gpt4o", "llama", "mistral"]
    print(f"Looking at data in {MAIN_PATH}...")
    
    dataframes = load_data_in_dir(MAIN_PATH)
    
    metrics = calculate_metrics(dataframes)
    print("Overall Metrics:")
    print_metrics(metrics)
    print("-" * 50)
    plot_boxplot_metrics(metrics)
    
    lang_metrics = []
    for lang in LANGUAGES:
        metrics = calculate_metrics_for_attribute(dataframes, "language", lang)
        lang_metrics.append((lang, metrics))
        print(f"Metrics for {lang}:")
        print_metrics_for_attribute(metrics)
        print(f"-" * 50)
    
    plot_barchart_metrics_for_language(lang_metrics)
        
    llm_metrics = []
    for llm in LLM_NAMES:
        metrics = calculate_metrics_for_attribute(dataframes, "source", llm)
        llm_metrics.append((llm, metrics))
        print(f"Metrics for {llm}:")
        print_metrics_for_attribute(metrics)
        print(f"-" * 50)
    
    plot_barchart_metrics_for_llm(llm_metrics)
    
    print("Done.")
    
    