import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score


# unpack files and read in meta learner X data
df = pd.read_csv('../results/preds.csv')
probs = df['probs']
actuals = df['actuals']


# calculate metrics
precision, recall, thresholds_pr = precision_recall_curve(actuals, probs)
fpr, tpr, thresholds_roc = roc_curve(actuals, probs)
pr_auc = average_precision_score(actuals, probs)
roc_auc = roc_auc_score(actuals, probs)


# calculate min(precision, recall) for each threshold in PR curve
min_precision_recall = [min(p, r) for p, r in zip(precision, recall)]
best_threshold_index = min_precision_recall.index(max(min_precision_recall))
best_threshold = thresholds_pr[best_threshold_index]


# extract precision and recall at the best threshold
best_precision = precision[best_threshold_index]
best_recall = recall[best_threshold_index]


# plot PR-AUC curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.3f})', color='b')
plt.scatter(best_recall, best_precision, color='red', label=f'Best Threshold: {best_threshold:.3f}')
plt.axhline(best_precision, color='orange', linestyle='--', alpha=0.7, label=f'Precision: {best_precision:.3f}')
plt.axvline(best_recall, color='orange', linestyle='--', alpha=0.7, label=f'Recall: {best_recall:.3f}')
plt.axhline(sum(actuals)/len(actuals), color='gray', linestyle='--', alpha=0.7, label=f'Baseline Precision: {sum(actuals)/len(actuals):.3f}')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.annotate(f'({best_recall:.3f}, {best_precision:.3f})',
               xy=(best_recall, best_precision),
               xytext=(best_recall + 0.05, best_precision + 0.05),
               arrowprops=dict(facecolor='black', arrowstyle='-'),
               fontsize=12)
plt.tight_layout()
plt.savefig('../results/pr_curve.png')
plt.close()


# plot PR-AUC and ROC-AUC curves side-by-side
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc:.3f})', color='b')
ax[0].scatter(best_recall, best_precision, color='red', label=f'Best Threshold: {best_threshold:.3f}')
ax[0].axhline(best_precision, color='orange', linestyle='--', alpha=0.7, label=f'Precision: {best_precision:.3f}')
ax[0].axvline(best_recall, color='orange', linestyle='--', alpha=0.7, label=f'Recall: {best_recall:.3f}')
ax[0].axhline(sum(actuals)/len(actuals), color='gray', linestyle='--', alpha=0.7, label=f'Baseline Precision: {sum(actuals)/len(actuals):.3f}')
ax[0].set_title('Precision-Recall Curve')
ax[0].set_xlabel('Recall')
ax[0].set_ylabel('Precision')
ax[0].legend(loc='best')
ax[0].grid(True, alpha=0.3)
ax[0].annotate(f'({best_recall:.3f}, {best_precision:.3f})',
               xy=(best_recall, best_precision),
               xytext=(best_recall + 0.05, best_precision + 0.05),
               arrowprops=dict(facecolor='black', arrowstyle='-'),
               fontsize=12)

ax[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', color='b')
ax[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].legend(loc='best')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/pr_roc_curves.png')
plt.close()


# convert probs to preds and extract confusion matrix
preds = (probs >= best_threshold).astype(int)

cm = confusion_matrix(actuals, preds)
cm = cm[[1, 0], :][:, [1, 0]]

confusion = pd.DataFrame(cm, index=['P+', 'P-'], columns=['D+', 'D-'])
confusion.to_csv('../results/confusion_matrix.csv', index=True)


# calculate metrics
sens = recall_score(actuals, preds)
spec = confusion.iloc[1,1] / confusion.iloc[:,1].sum()
ppv = precision_score(actuals, preds)
npv = confusion.iloc[1,1] / confusion.iloc[1,:].sum()
acc = accuracy_score(actuals, preds)
f1 = f1_score(actuals, preds)


# write stats to datadrame
stats = pd.DataFrame({'stat': ['sens', 'spec', 'ppv', 'npv', 'acc', 'f1'], 'value':[sens, spec, ppv, npv, acc, f1]})
stats.to_csv('../results/stats.csv', index=False)
