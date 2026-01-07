import pandas as pd
from scipy.stats import proportions_ztest

# Load data
df = pd.read_csv("../data/experiment_data.csv")

# Separate groups
group_a = df[df["variant"] == "A"]
group_b = df[df["variant"] == "B"]

# Conversions and sample sizes
conversions = [
    group_a["converted"].sum(),
    group_b["converted"].sum()
]

samples = [
    len(group_a),
    len(group_b)
]

# Conversion rates
rate_a = conversions[0] / samples[0]
rate_b = conversions[1] / samples[1]

# Hypothesis test
stat, p_value = proportions_ztest(conversions, samples)

print("Conversion Rate A:", round(rate_a, 3))
print("Conversion Rate B:", round(rate_b, 3))
print("P-value:", round(p_value, 4))

# Decision
alpha = 0.05
if p_value < alpha:
    print("Result: Statistically significant difference. Consider Version B.")
else:
    print("Result: No statistically significant difference. Do not roll out Version B.")
