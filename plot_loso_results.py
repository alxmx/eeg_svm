import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loso_reports.csv')
df = df[df['Label'].isin(['angry','calm','excited','sad'])]
metrics = ['precision','recall','f1-score']
for metric in metrics:
    plt.figure(figsize=(14,8))
    sns.barplot(data=df, x='Fold', y=metric, hue='Label')
    plt.title(f'Classification {metric.capitalize()} per Fold')
    plt.ylim(0,1.05)
    plt.legend(title='Emotion', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'classification_{metric}_per_fold_big.png')
    plt.close()
print('Plots saved: classification_precision_per_fold_big.png, classification_recall_per_fold_big.png, classification_f1-score_per_fold_big.png')
