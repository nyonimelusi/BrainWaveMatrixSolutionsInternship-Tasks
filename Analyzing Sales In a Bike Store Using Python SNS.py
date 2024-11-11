#!/usr/bin/env python
# coding: utf-8

# # Data Analysis With Seaborn: Analyzing Data Using Visualizations.

# In[2]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sales_data.csv")
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[6]:


df.info()


# Data Cleaning:

# We will check for Missing Values:

# In[7]:


df.isnull().sum()


# Data Visualization will be next:

# In[9]:


Histogram: df['Product_Category'].hist()


# Will will look at Product Profitability:

# To analyze the sales and profitability of the categories in that data, we can use the bar plot from the Seaborn library. We are going to combine sales and profitability in one bar plot:

# In[10]:


# Create a new column 'Sales' by multiplying 'Order_Quantity' by 'Unit_Price'
df['Sales'] = df['Order_Quantity'] * df['Unit_Price']

# Verify if the column was created correctly
print(df[['Order_Quantity', 'Unit_Price', 'Sales']].head())


# In[11]:


df.info()


# # We want to Analyze Sales by Category:

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot Sales vs. Profit by Category
def plot_sales_profit(data):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Barplot for Sales
    sns.barplot(x='Product_Category', y='Sales', data=data, color='blue', ax=ax1)
    ax1.set_ylabel('Sales', color='blue', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Barplot for Profit
    ax2 = ax1.twinx()
    sns.barplot(x='Product_Category', y='Profit', data=data, hue='Product_Category', ax=ax2)
    ax2.set_ylabel('Profit', color='Green', labelpad=10)
    ax2.tick_params(axis='y', labelcolor='Green')

    # Add title and formatting
    plt.title("Sales vs. Profit by Category", weight='bold', fontsize=20)
    plt.xlabel('Product_Category')

    # Add Legends
    ax1.legend(['Sales'], loc='upper left')
    ax2.legend(['Profit'], loc='upper right')

    plt.tight_layout()
    plt.show()

plot_sales_profit(df)

    


# In[15]:


# Import  libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot Sales by Product_Category
def plot_sales(data):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Barplot for Sales
    sns.barplot(x='Product_Category', y='Sales', data=data, color='blue', ax=ax1)
    ax1.set_ylabel('Sales', color='blue', labelpad=10)
    ax1.tick_params(axis='y', labelcolor='blue')

    # Add title and formatting
    plt.title("Sales by Product Category", weight='bold', fontsize=20)
    plt.xlabel('Product Category')
    plt.tight_layout()
    plt.show()

# Call the function with your DataFrame
plot_sales(df)


# We want to find which country has the highest Sales of Bikes?

# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df['Sales'] = df['Order_Quantity'] * df['Unit_Price']  


bike_sales = df[df['Sub_Category'].str.contains('Bike', case=False)]


country_sales = bike_sales.groupby('Country')['Sales'].sum().reset_index()


country_sales = country_sales.sort_values(by='Sales', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='Country', y='Sales', data=country_sales, palette='viridis')
plt.title('Total Bike Sales by Country', weight='bold', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Secondly we want to find out which Age group buys more bikes:

# In[27]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df['Sales'] = df['Order_Quantity'] * df['Unit_Price']  


bike_sales = df[df['Sub_Category'].str.contains('Bike', case=False)]


age_group_sales = bike_sales.groupby('Age_Group')['Sales'].sum().reset_index()

age_group_sales = age_group_sales.sort_values(by='Sales', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x='Age_Group', y='Sales', data=age_group_sales, palette='coolwarm')
plt.title('Total Bike Sales by Age Group', weight='bold', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # Analyzing Sales Distributions:

# To analyze the distribution of sales, we can use the Seaborn library's histplot function. The histplot() is ideal for visualizing the distribution of numerical data. We will use it to understand how the values in the sales column of our dataset are distributed.

# In[23]:


def plot_sales_distribution(df):
    plt.figure(figsize=(8, 6))
    
    #histogram plot with KDE
    sns.histplot(df['Sales'], kde=True, color='blue', bins=10)
    
    plt.xlabel('Sales', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Distribution Of Sales', fontsize=10)
    
    
plot_sales_distribution(df)    


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Log transform the 'Sales' column
df['Log_Sales'] = np.log1p(df['Sales'])  # Using log1p to handle zero sales values

# Plotting the distribution of log-transformed sales
plt.figure(figsize=(10, 6))
sns.histplot(df['Log_Sales'], bins=50, color='green', kde=True)
plt.title('Log-Transformed Sales Distribution', fontsize=16)
plt.xlabel('Log(Sales)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


# In[25]:


# Identifying potential outliers in the sales data
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out the outliers
outliers = df[(df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)]

# Displaying the outliers
print(outliers)


# In[30]:


# Binning the sales into categories
df['Sales_Bin'] = pd.cut(df['Sales'], bins=[0, 1000, 5000, 10000, 50000, 100000], 
                         labels=['0-1K', '1K-5K', '5K-10K', '10K-50K', '50K-100K'])

# Plotting the bin distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Sales_Bin', data=df, palette='coolwarm')
plt.title('Sales Distribution by Range', fontsize=16)
plt.xlabel('Sales Range', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# # Analyzing the Relationship Between Sales and Profit:
To gain insights into the relationship between sales and profit, we can employ Seaborn's regplot. This visualization not only plots individual data points but also overlays a regression line, providing a clear visual representation of the trend. Additionally, it helps identify potential outliers that might skew the analysis.
# In[35]:


def plot_sales_profit_relationship(df):
    plt.figure(figsize=(10,6))
    
    #Creating the regression plot with sns
    sns.regplot(x='Sales', y='Profit', data=df, scatter=True,
               line_kws={'color': 'red'})
    
    plt.title('Relationship Between Sales And Profit', fontsize=20)
    plt.xlabel('Sales', fontsize=12)
    plt.ylabel('Profit', fontsize=12)
    
    #show the plot
    plt.show()
    
plot_sales_profit_relationship(df) 


# The regplot shows a strong positive correlation between sales and profits. This means that as sales increase, profits also tend to increase. The regression line, which is a red line on the graph, represents this trend. The slope of the line is positive, indicating that the increase in profits is proportional to the increase in sales.
# 
# However, there are also some outliers, which are data points that deviate significantly from the regression line. These outliers might be due to various factors, such as seasonal fluctuations, promotional activities, or unusual events. It's important to investigate these outliers further to understand their underlying causes and determine whether they should be included in the analysis.
# 
# Overall, the regplot suggests a strong relationship between sales and profits, but there are also some factors that might influence this relationship.

# # Distribution Of Profit By Each Category

# In[37]:


sns.boxplot(x='Product_Category', y='Profit', data=df)
plt.title('Profit Distribution by Category')
plt.xlabel('Product_Category')
plt.ylabel('Profit')
plt.show() 


# The boxplot shows the profit distribution for three product categories: Accessories, Clothing, and Bikes. Here's a brief analysis:
# 
# Accessories: The box for Accessories is relatively small, indicating a narrow range of profits. The median profit is low, and there is one outlier.
# Clothing: The clothing category also has a relatively small box, but the median profit is higher than Accessories. There are two outliers.
# Bikes: The box for Bikes is the largest, indicating a wider range of profits. The median profit is the highest among the three categories, and there are several outliers.
# Overall, the boxplot suggests that Bikes have the highest overall profit, but also the greatest variability in profits. Accessories have the lowest profits and the least variability. Clothing is in the middle, with a moderate range of profits.

# In[ ]:




