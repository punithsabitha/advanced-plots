import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Iris.csv")

# Display first rows
print(df.head())

# Set style
sns.set(style="whitegrid")

# 1. Barplot: Species vs SepalLengthCm
plt.figure(figsize=(6,4))
sns.barplot(x="Species", y="SepalLengthCm", data=df)
plt.title("Barplot of Species vs SepalLengthCm")
plt.show()

# 2. Countplot: Count of different species
plt.figure(figsize=(6,4))
sns.countplot(x="Species", data=df)
plt.title("Count of Different Species")
plt.show()

# 3. Boxplot: Species vs SepalWidthCm
plt.figure(figsize=(6,4))
sns.boxplot(x="Species", y="SepalWidthCm", data=df)
plt.title("Boxplot of Species vs SepalWidthCm")
plt.show()

# 4. Swarmplot: Species vs SepalWidthCm
plt.figure(figsize=(6,4))
sns.swarmplot(x="Species", y="SepalWidthCm", data=df)
plt.title("Swarmplot of Species vs SepalWidthCm")
plt.show()

# 5. Distribution plot of SepalWidthCm
sns.displot(df["SepalWidthCm"], kde=True)
plt.title("Distribution of SepalWidthCm")
plt.show()

# 6. Jointplot of SepalWidthCm and SepalLengthCm
sns.jointplot(x="SepalWidthCm", y="SepalLengthCm", data=df)
plt.show()

# 7. Pairplot with Species as hue
sns.pairplot(df, hue="Species")
plt.show()