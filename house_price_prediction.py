import pandas as pd  #use pandas

#---------------------------Dataset Load-------------------------------
data = pd.read_csv("data/train.csv")  #Read the train dataset in python

#--------------------------Basic Check------------------------------
print(data.head()) #Show first 5 rows
print(data.shape) #Size(row-column)
print(data.columns) #Show column names
data.info() #column data type and missing value info
print(data.describe()) # Show avg,max,min,sd


#-----------------------------missing value analysis--------------------------------
print(data.isnull().sum()[data.isnull().sum() > 0]) #show missing value column only

# Fill missing value in LotFrontage
data["LotFrontage"] = data["LotFrontage"].fillna(data["LotFrontage"].median())

# Fill missing categorical values with "None"
data["Alley"] = data["Alley"].fillna("None")
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["BsmtQual"] = data["BsmtQual"].fillna("None")
data["BsmtCond"] = data["BsmtCond"].fillna("None")
data["BsmtExposure"] = data["BsmtExposure"].fillna("None")
data["BsmtFinType1"] = data["BsmtFinType1"].fillna("None")
data["BsmtFinType2"] = data["BsmtFinType2"].fillna("None")
data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
data["GarageType"] = data["GarageType"].fillna("None")
data["GarageFinish"] = data["GarageFinish"].fillna("None")
data["GarageQual"] = data["GarageQual"].fillna("None")
data["GarageCond"] = data["GarageCond"].fillna("None")
data["PoolQC"] = data["PoolQC"].fillna("None")
data["Fence"] = data["Fence"].fillna("None")
data["MiscFeature"] = data["MiscFeature"].fillna("None")

# Fill missing numerical values with median
data["MasVnrArea"] = data["MasVnrArea"].fillna(data["MasVnrArea"].median())
data["GarageYrBlt"] = data["GarageYrBlt"].fillna(data["GarageYrBlt"].median())

# Fill the missing categorical value with the most common value
most_common = data["Electrical"].mode()[0]
data["Electrical"] = data["Electrical"].fillna(most_common)

# Check final missing values
print(data.isnull().sum()[data.isnull().sum() > 0])

# Fill remaining numerical missing values with median
numerical_columns = data.select_dtypes(include="number").columns

for column in numerical_columns:
    data[column] = data[column].fillna(data[column].median())

# Check again
print(data.isnull().sum()[data.isnull().sum() > 0])

#---------------------------------------------EDA---------------------------------------
import matplotlib.pyplot as plt

# 1.Show house price distribution(ktogula house ei price range er moddhe porse)
plt.hist(data["SalePrice"], bins=30)
plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.title("House Price Distribution")
plt.show()

# 2.Living area vs house price
plt.figure(figsize=(8, 5))
plt.scatter(
    data["GrLivArea"],
    data["SalePrice"],
    alpha=0.5, # for transparent dot
    s=30 #dot size
)
plt.xlabel("Living Area (sq ft)")
plt.ylabel("House Price ($)")
plt.title("Living Area vs House Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check correlation between living area and house price
correlation = data["GrLivArea"].corr(data["SalePrice"])
print("Correlation:", correlation)

# 3. Overall Quality vs House Price
plt.figure(figsize=(8, 5))
plt.scatter(
    data["OverallQual"],
    data["SalePrice"],
    alpha=0.5,
    s=30
)
plt.xlabel("Overall Quality")
plt.ylabel("House Price ($)")
plt.title("Overall Quality vs House Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check correlation between quality and house price
correlation = data["OverallQual"].corr(data["SalePrice"])
print("Correlation:", correlation)

# 4. Year Built vs House Price(bari joto notun hy dam toto bshi hy)
plt.figure(figsize=(8, 5))
plt.scatter(
    data["YearBuilt"],
    data["SalePrice"],
    alpha=0.5,
    s=30
)
plt.xlabel("Year Built")
plt.ylabel("House Price ($)")
plt.title("Year Built vs House Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check correlation(moderate positive)
correlation = data["YearBuilt"].corr(data["SalePrice"])
print("Correlation:", correlation)

# 5. Basement Area vs House Price
plt.figure(figsize=(8, 5))
plt.scatter(
    data["TotalBsmtSF"],
    data["SalePrice"],
    alpha=0.5,
    s=30
)
plt.xlabel("Basement Area (sq ft)")
plt.ylabel("House Price ($)")
plt.title("Basement Area vs House Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check correlation(ans e positive ashse)
correlation = data["TotalBsmtSF"].corr(data["SalePrice"])
print("Correlation:", correlation)

# 6. Garage Capacity vs House Price
plt.figure(figsize=(8, 5))
plt.scatter(
    data["GarageCars"],
    data["SalePrice"],
    alpha=0.5,
    s=30
)
plt.xlabel("Garage Capacity (Cars)")
plt.ylabel("House Price ($)")
plt.title("Garage Capacity vs House Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check correlation
correlation = data["GarageCars"].corr(data["SalePrice"])
print("Correlation:", correlation)

# 7. First Floor Area vs House Price
plt.figure(figsize=(8, 5))
plt.scatter(
    data["1stFlrSF"],
    data["SalePrice"],
    alpha=0.5,
    s=30
)
plt.xlabel("First Floor Area (sq ft)")
plt.ylabel("House Price ($)")
plt.title("First Floor Area vs House Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check correlation
correlation = data["1stFlrSF"].corr(data["SalePrice"])
print("Correlation:", correlation)

# 8. 2nd Floor Area vs House Price
plt.figure(figsize=(8, 5))
plt.scatter(
    data["2ndFlrSF"],
    data["SalePrice"],
    alpha=0.5,
    s=30
)
plt.xlabel("2nd Floor Area (sq ft)")
plt.ylabel("House Price ($)")
plt.title("2nd Floor Area vs House Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Check correlation
correlation = data["2ndFlrSF"].corr(data["SalePrice"])
print("Correlation:", correlation)

# 9. Correlation of Important Features with House Price
features = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "1stFlrSF",
    "YearBuilt",
    "2ndFlrSF"
]
correlations = data[features + ["SalePrice"]].corr()["SalePrice"].sort_values(ascending=False)
print(correlations)

# 10. Correlation Heatmap
import seaborn as sns
plt.figure(figsize=(10, 8))
correlation_matrix = data[
    features + ["SalePrice"]
].corr()
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#------------------------Feature Engineering-----------------------
# 1. Create Total Area
data["TotalSF"] = (
    data["TotalBsmtSF"] +
    data["1stFlrSF"] +
    data["2ndFlrSF"]
)

# 2. Create Total Bathrooms
data["TotalBathrooms"] = (
    data["FullBath"] +
    0.5 * data["HalfBath"] +
    data["BsmtFullBath"] +
    0.5 * data["BsmtHalfBath"]
)

# 3. Create Total Porch Area
data["TotalPorchSF"] = (
    data["OpenPorchSF"] +
    data["3SsnPorch"] +
    data["EnclosedPorch"] +
    data["ScreenPorch"] +
    data["WoodDeckSF"]
)

# 4. Create House Age
data["HouseAge"] = data["YrSold"] - data["YearBuilt"]

# Check the new features
print(data[[
    "TotalSF",
    "TotalBathrooms",
    "TotalPorchSF",
    "HouseAge"
]].head())

#-----------------------------categorical to numerical conversion---------------------------------
# Remove Id column
data = data.drop("Id", axis=1)
print(data.shape)

# Find numerical and categorical columns
numerical_columns = data.select_dtypes(include="number").columns
categorical_columns = data.select_dtypes(include="object").columns
print("Numerical columns:")
print(numerical_columns)   #37
print("\nCategorical columns:")
print(categorical_columns)  #43

# Convert categorical columns into numbers(ONE-hot Encoding)
data = pd.get_dummies(data, columns=categorical_columns)
print(data.shape)

# separate features and target
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)


#------------------------Prepare Test Data-----------------------

# Load test dataset
test_data = pd.read_csv("data/test.csv")
print("Test data shape:", test_data.shape)

#------------------------Missing value analysis in test data------------------------
# Check missing values in test data
print(test_data.isnull().sum()[test_data.isnull().sum() > 0])


#--------------------Fill Missing Values in Test Data--------------------
# Fill missing values in test data
test_data["LotFrontage"] = test_data["LotFrontage"].fillna(data["LotFrontage"].median())
test_data["MasVnrArea"] = test_data["MasVnrArea"].fillna(data["MasVnrArea"].median())
test_data["GarageYrBlt"] = test_data["GarageYrBlt"].fillna(data["GarageYrBlt"].median())

# Fill categorical missing values with "None"
test_data["Alley"] = test_data["Alley"].fillna("None")
test_data["MasVnrType"] = test_data["MasVnrType"].fillna("None")
test_data["BsmtQual"] = test_data["BsmtQual"].fillna("None")
test_data["BsmtCond"] = test_data["BsmtCond"].fillna("None")
test_data["BsmtExposure"] = test_data["BsmtExposure"].fillna("None")
test_data["BsmtFinType1"] = test_data["BsmtFinType1"].fillna("None")
test_data["BsmtFinType2"] = test_data["BsmtFinType2"].fillna("None")
test_data["FireplaceQu"] = test_data["FireplaceQu"].fillna("None")
test_data["GarageType"] = test_data["GarageType"].fillna("None")
test_data["GarageFinish"] = test_data["GarageFinish"].fillna("None")
test_data["GarageQual"] = test_data["GarageQual"].fillna("None")
test_data["GarageCond"] = test_data["GarageCond"].fillna("None")
test_data["PoolQC"] = test_data["PoolQC"].fillna("None")
test_data["Fence"] = test_data["Fence"].fillna("None")
test_data["MiscFeature"] = test_data["MiscFeature"].fillna("None")

# Fill remaining missing values with the most common value
test_data = test_data.fillna(data.mode().iloc[0])

# Check remaining missing values
print(test_data.isnull().sum()[test_data.isnull().sum() > 0])

# Fill remaining missing values
test_data["MSZoning"] = test_data["MSZoning"].fillna(test_data["MSZoning"].mode()[0])
test_data["Utilities"] = test_data["Utilities"].fillna(test_data["Utilities"].mode()[0])
test_data["Exterior1st"] = test_data["Exterior1st"].fillna(test_data["Exterior1st"].mode()[0])
test_data["Exterior2nd"] = test_data["Exterior2nd"].fillna(test_data["Exterior2nd"].mode()[0])
test_data["KitchenQual"] = test_data["KitchenQual"].fillna(test_data["KitchenQual"].mode()[0])
test_data["Functional"] = test_data["Functional"].fillna(test_data["Functional"].mode()[0])
test_data["SaleType"] = test_data["SaleType"].fillna(test_data["SaleType"].mode()[0])

# Check missing values
print(test_data.isnull().sum()[test_data.isnull().sum() > 0])


