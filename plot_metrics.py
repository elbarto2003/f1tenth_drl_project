import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
df = pd.read_csv("models/metrics.csv")


# Ensure 'algo' column exists
if 'algo' not in df.columns:
    df['algo'] = df['model'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else 'unknown')

# Group by algorithm and compute the best model per metric
best_reward = df.loc[df.groupby('algo')['mean_reward'].idxmax()]
best_crash = df.loc[df.groupby('algo')['eps_with_crash'].idxmin()]
best_length = df.loc[df.groupby('algo')['avg_length'].idxmax()]
best_laps = df.loc[df.groupby('algo')['avg_lap_count'].idxmax()]
best_lap_time = df.loc[df.groupby('algo')['mean_lap_time'].idxmin()]

# Plotting each metric
def plot_bar(data, metric, ylabel, title, filename):
    plt.figure(figsize=(4, 4))  # Slightly wider for longer titles
    bars = plt.bar(data['algo'], data[metric])

    # Bold title and axis labels
    plt.title(title, fontweight='bold', fontsize=12, pad=15)  # Add padding to separate from plot
    plt.ylabel(ylabel, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    # Extend y-limit to make space for labels
    max_val = max(data[metric])
    plt.ylim(0, max_val * 1.2)  # Adjusted from 1.15 to 1.2 for better spacing

    # Annotate bars slightly below the top of the bar
    for i, val in enumerate(data[metric]):
        plt.text(i, val + max_val * 0.02, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space at top for title
    plt.savefig(f"models/{filename}")
    plt.close()


plot_bar(best_reward, 'mean_reward', 'Mean Reward', 'Best Mean Reward by Algorithm', 'mean_reward.png')
plot_bar(best_crash, 'eps_with_crash', 'Episodes with Crash', 'Fewest Crashes by Algorithm', 'crashes.png')
plot_bar(best_length, 'avg_length', 'Average Episode Length', 'Longest Episode Length', 'avg_length.png')
plot_bar(best_laps, 'avg_lap_count', 'Average Lap Count', 'Most Laps Completed by Algorithm', 'avg_laps.png')
plot_bar(best_lap_time, 'mean_lap_time', 'Mean Lap Time (steps)', 'Fastest Lap Time by Algorithm', 'mean_lap_time.png')
