import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Load the dataset
df = pd.read_csv('Zara_sales_EDA.csv')

# Display basic information
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# Create a comprehensive visualization layout
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Distribution of numerical columns
ax1 = fig.add_subplot(gs[0, 0])
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 0:
    df[numerical_cols[0]].hist(bins=30, ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_title(f'Distribution of {numerical_cols[0]}')
    ax1.set_xlabel(numerical_cols[0])
    ax1.set_ylabel('Frequency')

# 2. Box plot for outlier detection
ax2 = fig.add_subplot(gs[0, 1])
if len(numerical_cols) > 0:
    ax2.boxplot([df[col].dropna() for col in numerical_cols[:3]], 
                labels=numerical_cols[:3])
    ax2.set_title('Box Plot - Outlier Detection')
    ax2.set_ylabel('Values')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 3. Missing values visualization
ax3 = fig.add_subplot(gs[0, 2])
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    ax3.barh(range(len(missing)), missing.values, color='coral')
    ax3.set_yticks(range(len(missing)))
    ax3.set_yticklabels(missing.index)
    ax3.set_xlabel('Missing Count')
    ax3.set_title('Missing Values by Column')
else:
    ax3.text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
    ax3.set_title('Missing Values')

# 4. Correlation heatmap (simplified)
ax4 = fig.add_subplot(gs[1, :])
if len(numerical_cols) > 1:
    corr = df[numerical_cols].corr()
    im = ax4.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_yticks(range(len(corr.columns)))
    ax4.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax4.set_yticklabels(corr.columns)
    
    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax4.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='black', fontsize=8)
    
    plt.colorbar(im, ax=ax4)
    ax4.set_title('Correlation Matrix')

# 5. Categorical column distribution (if exists)
ax5 = fig.add_subplot(gs[2, 0])
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    value_counts = df[categorical_cols[0]].value_counts().head(10)
    ax5.bar(range(len(value_counts)), value_counts.values, color='teal')
    ax5.set_xticks(range(len(value_counts)))
    ax5.set_xticklabels(value_counts.index, rotation=45, ha='right')
    ax5.set_title(f'Top 10 {categorical_cols[0]}')
    ax5.set_ylabel('Count')

# 6. Line plot for trends (if applicable)
ax6 = fig.add_subplot(gs[2, 1])
if len(numerical_cols) > 1:
    ax6.plot(df.index[:100], df[numerical_cols[0]].iloc[:100], 
            marker='o', linestyle='-', linewidth=1, markersize=3, color='purple')
    ax6.set_title(f'{numerical_cols[0]} Trend (First 100 records)')
    ax6.set_xlabel('Index')
    ax6.set_ylabel(numerical_cols[0])
    ax6.grid(True, alpha=0.3)

# 7. Scatter plot (if multiple numerical columns exist)
ax7 = fig.add_subplot(gs[2, 2])
if len(numerical_cols) >= 2:
    ax7.scatter(df[numerical_cols[0]], df[numerical_cols[1]], 
               alpha=0.5, color='green', edgecolors='black', linewidth=0.5)
    ax7.set_xlabel(numerical_cols[0])
    ax7.set_ylabel(numerical_cols[1])
    ax7.set_title(f'{numerical_cols[0]} vs {numerical_cols[1]}')
    ax7.grid(True, alpha=0.3)

plt.suptitle('Zara Sales - Exploratory Data Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Additional analysis - Top insights
print("\n" + "="*50)
print("KEY INSIGHTS")
print("="*50)
print(f"Total Records: {len(df)}")
print(f"Total Features: {len(df.columns)}")
print(f"Numerical Features: {len(numerical_cols)}")
print(f"Categorical Features: {len(categorical_cols)}")

if len(numerical_cols) > 0:
    print(f"\nMean of {numerical_cols[0]}: {df[numerical_cols[0]].mean():.2f}")
    print(f"Median of {numerical_cols[0]}: {df[numerical_cols[0]].median():.2f}")
    print(f"Std Dev of {numerical_cols[0]}: {df[numerical_cols[0]].std():.2f}")
