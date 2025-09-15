import json

# Load the JSON file
with open("profiling_results.json", "r") as f:
    profiling_data = json.load(f)

# Get the number of runs (assuming all event lists have the same length)
num_runs = len(next(iter(profiling_data.values()))["times"])

# Dictionary to store total execution time per run
total_times = {}

# Iterate over each run index
for run_index in range(num_runs):
    total_time = 0.0  # Initialize total time for this run

    # Sum the times of all events for this run
    for event in profiling_data:  # Loop over events (ffma, fadd, etc.)
        total_time += profiling_data[event]["times"][run_index]  # Add time from this event

    # Store the total time for this run
    total_times[f"Run_{run_index+1}"] = total_time  # 1-based indexing for readability

# Save the total execution times to a new JSON file
with open("total_execution_times.json", "w") as f:
    json.dump(total_times, f, indent=4)

print("Total execution times per run have been saved in 'total_execution_times.json'.")
