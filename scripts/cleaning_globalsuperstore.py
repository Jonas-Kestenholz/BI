from data_loader import load_excel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import os 
from scipy.stats import yeojohnson

# --- 1. DATA COLLECTION AND INITIAL CLEANING ---

data = load_excel("~/Desktop/BIexam/data/Global Superstore.xls")

colnames = data.columns.tolist()

#Here i look at the data at first glance
print(data.head())
print(colnames)
print(data.shape)

#We can see on the data that some of the columns are not relevant for our analysis, so we remove them
data = data.drop(columns=['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Product Name'])

print(data.info())
print(data.isnull().sum())

#It can be seen that there are missing alot of postal codes in the data sheet, so we are also removing that column
data = data.drop(columns=['Postal Code'])

#Here days months and years are extracted from the order date and ship date columns
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

data['Order Year'] = data['Order Date'].dt.year
data['Order Month'] = data['Order Date'].dt.month
data['Order Day'] = data['Order Date'].dt.day

# --- 2. EXPLORATION AND OUTLIER IDENTIFICATION ---

#I make some basic statistics and visualizations to get an idea of the data
print("\nNumerical Features:")
num_cols = data.select_dtypes(include=np.number).columns
print(data[num_cols].describe().transpose())

print("\nTop 5 Categories (Value Counts):")
print(data['Category'].value_counts())
print("\nTop 5 Regions (Value Counts):")
print(data['Region'].value_counts())

#I make some boxplots to visualize the numerical data
for col in ['Profit', 'Sales', 'Shipping Cost']:
    plot.figure(figsize=(6, 4))
    sns.boxplot(y=data[col])
    plot.title(f'Boxplot of {col} (Raw Data)')
    plot.ylabel(col)
    plot.show()

# --- 3. OUTLIER HANDLING (Capping using IQR method) ---

#Then as i can see that there are some outliers in the data, i make a function to cap them using the IQR method
#I also make some new boxplots to visualize the data after capping
def cap_outliers(df, column):
    print(f"\nHandling Outliers for: {column} ---")
    

    #Calculate IQR bounds
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    #Apply capping
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    print(f"  Lower Bound: {lower_bound:.2f}")
    print(f"  Upper Bound: {upper_bound:.2f}")

    #Visual check AFTER capping
    plot.figure(figsize=(6, 4))
    sns.boxplot(y=df[column])
    plot.title(f'Boxplot of {column} (After Capping)')
    plot.ylabel(f'{column} (Capped)')
    plot.show()
    return df

data = cap_outliers(data, 'Profit')
data = cap_outliers(data, 'Sales')
data = cap_outliers(data, 'Shipping Cost')

# --- 4. SKEWNESS HANDLING (TRANSFORMATION) ---

plot.figure(figsize=(8, 6))
sns.histplot(data['Profit'], kde=True)
plot.title('Distribution of Profit (Capped, BEFORE Transformation)')
plot.xlabel('Profit (Capped)')
plot.show()

plot.figure(figsize=(8, 6))
sns.histplot(data['Sales'], kde=True)
plot.title('Distribution of Sales (Capped, BEFORE Transformation)')
plot.xlabel('Sales (Capped)')
plot.show()

plot.figure(figsize=(8, 6))
sns.histplot(data['Shipping Cost'], kde=True)
plot.title('Distribution of Shipping Cost (Capped, BEFORE Transformation)')
plot.xlabel('Shipping Cost (Capped)')
plot.show()

#I look at a histogram of the capped numerical data.
#Since we have with negative data in profits we apply Yeo-Johnson transformation
data['Profit_YJ'], lmbda = yeojohnson(data['Profit'])
print(f"\nProfit Yeo-Johnson Lambda: {lmbda:.3f}")

#And then to sales and shipping cost we can apply log1p transformation
data['Sales_Log'] = np.log1p(data['Sales'])
data['ShippingCost_Log'] = np.log1p(data['Shipping Cost'])

#Visualize the transformed data
plot.figure(figsize=(8, 6))
sns.histplot(data['Profit_YJ'], kde=True)
plot.title('Distribution of Profit (After Yeo-Johnson Transformation)')
plot.xlabel('Profit (Yeo-Johnson Transformed)')
plot.show()

plot.figure(figsize=(8, 6))
sns.histplot(data['Sales_Log'], kde=True)
plot.title('Distribution of Sales (After Log1p Transformation)')
plot.xlabel('Sales (Log Transformed)')
plot.show()

plot.figure(figsize=(8, 6))
sns.histplot(data['ShippingCost_Log'], kde=True)
plot.title('Distribution of ShippingCost (After Log1p Transformation)')
plot.xlabel('Shipping (Log Transformed)')
plot.show()

#I remove the untransformed columns to avoid multicollinearity
data = data.drop(columns=['Profit', 'Sales', 'Shipping Cost'])

# --- 5. CATEGORICAL DATA PREPARATION (One-Hot Encoding) ---

#We check for binary columns and can see there arent any
unique_counts = data.nunique()
binary_columns = unique_counts[unique_counts == 2].index

object_columns = data.select_dtypes(include=['object']).columns

#We check the columns for unique values and can see that some of them have alot of unique values that wouldnt make sense to do one-hot encoding on
print(data[object_columns].nunique().sort_values(ascending=False))

#We remove the columns with too many unique values
data = data.drop(columns=['City', 'State', 'Country'])

object_columns = data.select_dtypes(include=['object']).columns

#Then some one-hot encoding
data_final = pd.get_dummies(data, columns=object_columns, drop_first=True)

# --- 6. FINAL DATAFRAME SAVING ---
#I save the final dataframe as a pickle file
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, '..', 'data', 'processed')

output_path = os.path.join(output_dir, 'global_superstore_cleaned.pkl')
data_final.to_pickle(output_path)

csv_path = os.path.join(output_dir, 'global_superstore_cleaned.csv')
data_final.to_csv(csv_path, index=False) 

print(data_final.head())
print(data_final.shape)
