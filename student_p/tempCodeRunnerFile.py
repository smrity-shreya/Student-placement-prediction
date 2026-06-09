lse)
print(f"Dataset saved → placement_data.csv  |  Shape: {df.shape}")
print(df["Placed"].value_counts().rename({1: "Placed", 0: "Not Placed"}))
print(df.head())
