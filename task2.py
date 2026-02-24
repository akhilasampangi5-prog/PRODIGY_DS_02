import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

print("First 5 rows:")
print(df.head())
print("\nShape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin column (too many missing values)
df.drop("Cabin", axis=1, inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()