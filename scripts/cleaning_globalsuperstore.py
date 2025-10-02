from data_loader import load_excel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

data = load_excel("~/Desktop/BIexam/data/Global Superstore.xls")

colnames = data.columns.tolist()

print(data.head())
print(colnames)
print(data.shape)

#We can see on the data that some of the columns are not relevant for our analysis, so we remove them
customer_names = data['Customer Name']
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

#I make some basic statistics and visualizations to get an idea of the data
print("\nNumerical Features:")
numeriske_kolonner = data.select_dtypes(include=np.number).columns
print(data[numeriske_kolonner].describe().transpose())

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

#Then as i can see that there are some outliers in the data, i make a function to cap them using the IQR method
def cap_outliers_iqr(df, column):
    """Calculates IQR bounds and caps outliers in the specified column."""
    print(f"\n--- Handling Outliers for: {column} ---")
    


    # Calculate IQR bounds
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply capping
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

    print(f"  Lower Bound: {lower_bound:.2f}")
    print(f"  Upper Bound: {upper_bound:.2f}")

    # Visual check AFTER capping
    plot.figure(figsize=(6, 4))
    sns.boxplot(y=df[column])
    plot.title(f'Boxplot of {column} (After Capping)')
    plot.ylabel(f'{column} (Capped)')
    plot.show()
    return df

#I look at af histograms of the numerical data
#We can see that the data is very skewed, so we will log-transform it
sns.histplot(data['Profit'], kde=True)
plot.title('Distribution of Capped Profit')
plot.show()

data['Profit_Log'] = np.log1p(data['Profit'])
data['Sales_Log'] = np.log1p(data['Sales'])
data['ShippingCost_Log'] = np.log1p(data['Shipping Cost'])

plot.figure(figsize=(8, 6))
sns.histplot(data['Profit_Log'], kde=True)
plot.title('Distribution of Profit (After Log-Transformation)')
plot.xlabel('Log(1 + Profit)')
plot.show()

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

print(data_final.head())

