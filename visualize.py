import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

def generate_comparisons():
    os.makedirs("results/plots", exist_ok=True)
    
    # Load all results
    all_files = glob.glob("results/*_results.csv")
    if not all_files:
        print("No result CSVs found in results/ directory.")
        return
        
    df_list = [pd.read_csv(f) for f in all_files]
    df_all = pd.concat(df_list, ignore_index=True)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # 1-4. Metric comparisons across models for all datasets
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_all, x='Dataset', y=metric, hue='Model')
        plt.title(f'{metric} Comparison across Models and Datasets')
        plt.ylim(0, 100)
        plt.ylabel(f'{metric} (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'results/plots/{metric.lower()}_comparison.png')
        plt.close()
        
    # 5. Dataset-wise comparison (Radar chart or grouped bar chart)
    # We will do a grouped bar chart per dataset showing all metrics for all models
    for dataset in df_all['Dataset'].unique():
        df_ds = df_all[df_all['Dataset'] == dataset]
        df_melt = pd.melt(df_ds, id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model')
        plt.title(f'Performance Metrics for {dataset}')
        plt.ylim(0, 100)
        plt.ylabel('Score (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'results/plots/dataset_wise_{dataset.lower()}.png')
        plt.close()

def plot_confusion_matrix(dataset, model_name="hybrid"):
    cm_path = f"results/{dataset}_{model_name}_cm.csv"
    if not os.path.exists(cm_path):
        print(f"Confusion matrix for {dataset} {model_name} not found.")
        return
        
    cm = pd.read_csv(cm_path).values
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset.upper()} ({model_name.upper()})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'results/plots/cm_{dataset}_{model_name}.png')
    plt.close()

if __name__ == "__main__":
    generate_comparisons()
    plot_confusion_matrix("hcv", "hybrid")
    plot_confusion_matrix("dermatology", "hybrid")
    print("Visualizations generated in results/plots/")
