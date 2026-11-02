CELL_SOURCES = [
    r'''
import matplotlib.pyplot as plt
import seaborn as sns

X = ds["X"].copy()
y = ds["y"].copy()
target = ds["target_name"]

print("Shape:", X.shape)
print()
print("Dtypes:\n", X.dtypes)
print()
print("Missing values:\n", X.isna().sum())
print()
print("Class balance (target):")
print(y.value_counts())
print("Default rate:", format(y.mean(), ".4f"))
print()
print("Describe:")
print(X.describe())

fig_dir = "reports/figures"

num_cols = X.select_dtypes(include=np.number).columns
if len(num_cols) > 1:
    plt.figure(figsize=(9, 7))
    sns.heatmap(X[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_correlation_heatmap.png"), dpi=150)
    plt.close()

plt.figure(figsize=(5, 4))
sns.countplot(x=y)
plt.title("Class Balance (Target)")
plt.xlabel(target)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "eda_class_balance.png"), dpi=150)
plt.close()

for c in list(num_cols)[:6]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=y, y=X[c])
    plt.title(str(c) + " by Target")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "eda_boxplot_" + str(c) + ".png"), dpi=150)
    plt.close()

print("EDA plots saved to reports/figures/")
''',
]