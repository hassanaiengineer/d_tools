import pandas as pd

# Load the CSV file
df = pd.read_csv('sample_data.csv') 

# Display the first few rows to check if it's loaded correctly
print(df.head())

# Check data structure
print(df.info())

##

# Filter accounts that moved from Growth to Strategic
growth_to_strategic = df[df['SEGMENT_MOVEMENT'] == 'Growth to Strategic']

# Filter accounts that moved from Strategic to Growth
strategic_to_growth = df[df['SEGMENT_MOVEMENT'] == 'Strategic to Growth']

# Display the number of records for each transition
print(f"Growth to Strategic: {growth_to_strategic.shape[0]} accounts")
print(f"Strategic to Growth: {strategic_to_growth.shape[0]} accounts")

# Check a sample of the filtered data
print(growth_to_strategic.head())
print(strategic_to_growth.head())

##

# Create control group for Growth to Strategic (accounts that stayed in Growth)
control_growth_to_strategic = df[(df['FY24Q2_SEGMENT'] == 'Growth') & (df['FY24Q3_SEGMENT'] == 'Growth')]

# Create control group for Strategic to Growth (accounts that stayed in Strategic)
control_strategic_to_growth = df[(df['FY24Q2_SEGMENT'] == 'Strategic') & (df['FY24Q3_SEGMENT'] == 'Strategic')]

# Display the number of records in each control group
print(f"Control Group for Growth to Strategic: {control_growth_to_strategic.shape[0]} accounts")
print(f"Control Group for Strategic to Growth: {control_strategic_to_growth.shape[0]} accounts")

# Check a sample of the control groups
print(control_growth_to_strategic.head())
print(control_strategic_to_growth.head())

##

# Calculate the average revenue for Growth to Strategic (treated and control groups)
avg_rev_growth_to_strategic_treated = growth_to_strategic['REV'].mean()
avg_rev_growth_to_strategic_control = control_growth_to_strategic['REV'].mean()

# Calculate the average revenue for Strategic to Growth (treated and control groups)
avg_rev_strategic_to_growth_treated = strategic_to_growth['REV'].mean()
avg_rev_strategic_to_growth_control = control_strategic_to_growth['REV'].mean()

# Print the results
print(f"Average Revenue for Growth to Strategic (Treated): {avg_rev_growth_to_strategic_treated}")
print(f"Average Revenue for Growth to Strategic (Control): {avg_rev_growth_to_strategic_control}")

print(f"Average Revenue for Strategic to Growth (Treated): {avg_rev_strategic_to_growth_treated}")
print(f"Average Revenue for Strategic to Growth (Control): {avg_rev_strategic_to_growth_control}")
