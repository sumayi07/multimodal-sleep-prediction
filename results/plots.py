import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read improvement data
df = pd.read_csv("improvement_data.csv")

# Plotting aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 20,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.title_fontsize': 14
})

predictor_order = [
    'phone', 'exercise', 'mood', 
    'phone+exercise', 'phone+mood', 
    'exercise+mood', 'all_combined'
]

fig, ax = plt.subplots(figsize=(12, 6))
colors = {'efficiency': '#1f77b4', 'minutesAsleep': '#ff7f0e', 'restfulness': '#2ca02c'}

for target in df['Target'].unique():
    target_df = df[df['Target'] == target].copy()
    target_df['Predictor Group'] = pd.Categorical(
        target_df['Predictor Group'],
        categories=predictor_order,
        ordered=True
    )
    target_df = target_df.sort_values('Predictor Group')
    
    sns.lineplot(
        data=target_df,
        x='Predictor Group',
        y='Percentage Improvement',
        marker='o',
        markersize=8,
        linewidth=2.5,
        color=colors[target],
        label=target.capitalize(),
        ax=ax
    )

ax.set_title('Percentage Improvement from Baseline', pad=20, fontweight='bold')
ax.set_ylabel('Percentage Improvement (%)', labelpad=15, fontweight='bold')
ax.set_xlabel('Predictor Group', labelpad=15, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

legend = ax.legend(title='Target Metric', framealpha=1)
legend.get_title().set_fontweight('bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig('improvement_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated 'improvement_comparison.png'")