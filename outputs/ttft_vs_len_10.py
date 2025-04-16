import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
data = pd.read_csv("test_local_cpu_experimental.csv")

# Filter out rows where context_len equals 25000
data = data[data['context_len'] != 25000]

# Verify the unique engine IDs for debugging purposes
unique_engine_ids = data['engine_id'].unique()
print("Unique engine_id values:", unique_engine_ids)

# Group the data by 'engine_id' and 'context_len'
grouped = data.groupby(['engine_id', 'context_len'])

# Compute three TTFT metrics for each group:
# 1. avg_all: average TTFT over all rows in the group
# 2. avg_except_first: average TTFT excluding the first row (if more than one row exists)
# 3. first_value: TTFT value from the first row
results = grouped['TTFT'].agg(
    avg_all='mean',
    avg_except_first=lambda x: x.iloc[1:].mean() if len(x) > 1 else x.iloc[0],
    first_value=lambda x: x.iloc[0]
).reset_index()

# Create a figure with three subplots (stacked vertically)
fig, axs = plt.subplots(3, 1, figsize=(10, 18), sharex=True)

# List all engine_id values in the aggregated results
engine_ids = results['engine_id'].unique()

# Plot each metric for every engine_id on the corresponding subplot.
for engine in engine_ids:
    engine_data = results[results['engine_id'] == engine]
    
    # Plot average of all rows (typically 10 rows)
    axs[0].plot(engine_data['context_len'], engine_data['avg_all'],
                marker='o', linestyle='-', label=f"engine_id {engine}")
    
    # Plot average of rows excluding the first
    axs[1].plot(engine_data['context_len'], engine_data['avg_except_first'],
                marker='o', linestyle='-', label=f"engine_id {engine}")
    
    # Plot the first row TTFT value
    axs[2].plot(engine_data['context_len'], engine_data['first_value'],
                marker='o', linestyle='-', label=f"engine_id {engine}")

# Add labels, titles, grid, and legend to each subplot.
axs[0].set_title("TTFT vs. Context Length - Average of All Rows")
axs[0].set_ylabel("TTFT (all rows)")
axs[0].grid(True)
axs[0].legend(title="engine_id")

axs[1].set_title("TTFT vs. Context Length - Average Excluding First Row")
axs[1].set_ylabel("TTFT (rows 2-10)")
axs[1].grid(True)
axs[1].legend(title="engine_id")

axs[2].set_title("TTFT vs. Context Length - First Row")
axs[2].set_xlabel("Context Length")
axs[2].set_ylabel("TTFT (first row)")
axs[2].grid(True)
axs[2].legend(title="engine_id")

# -----------------------------------------------------------
# Annotate overhead percentage of engine1 relative to engine0 on each graph
# -----------------------------------------------------------
# Make sure we have at least two engine_ids
engine_ids_sorted = sorted(results['engine_id'].unique())
if len(engine_ids_sorted) >= 2:
    # Assume engine0 is the first and engine1 is the second in sorted order.
    engine0_id = engine_ids_sorted[0]
    engine1_id = engine_ids_sorted[1]
    
    # Create separate DataFrames for engine0 and engine1
    engine0_df = results[results['engine_id'] == engine0_id]
    engine1_df = results[results['engine_id'] == engine1_id]
    
    # Merge the two DataFrames on context_len so that we can compare metrics
    merged = pd.merge(engine0_df, engine1_df, on='context_len', suffixes=('_0', '_1'))
    
    # Define the list of metric columns in the order corresponding to the subplots
    metric_columns = ['avg_all', 'avg_except_first', 'first_value']
    
    for i, metric in enumerate(metric_columns):
        # For each matching context_len, calculate overhead and annotate on the subplot
        for k, row in merged.iterrows():
            # Get the x coordinate (context_len) and the y coordinate from engine1's metric
            x_val = row['context_len']
            y_val = row[f"{metric}_1"]
            # Calculate the overhead percentage
            overhead = - (row[f"{metric}_1"] - row[f"{metric}_0"]) / row[f"{metric}_0"] * 100
            
            # Annotate the overhead percentage on the corresponding subplot, with a small offset
            axs[i].annotate(f"{overhead:.1f}%",
                            (x_val, y_val),
                            textcoords="offset points",
                            xytext=(0, -18),
                            ha='center')
else:
    print("Not enough engine_ids to compute overhead annotations.")

# Adjust layout and save the figure to a file
plt.tight_layout()
plt.savefig("ttft_vs_context_length_three_graphs.png")
plt.close()

print("Plot saved to 'ttft_vs_context_length_three_graphs.png'.")
