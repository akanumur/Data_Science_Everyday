## Predict the lowest price
Source: Hackerearth

### Problem Statement

A leading global leader of e-commerce has over 150 million paid subscription users. One of the many perks of the subscription is the privilege of buying products at lower prices. For an upcoming sale, the organization has decided to promote local artisans and their products, to help them through these tough times. However, slashed prices may impact local artists.

To not let discounts affect local artists, the company has decided to determine the lowest price at which a particular good can be sold. Your task is to build a predictive model using Machine Learning that helps them set up a lowest-pricing model for these products.

You have to predict the Low_Cap_Price column.

## Data 
### Data description

Item_Id: Unique item ID

Date: Date

State_of_Country: State no. of the country

Market_Category: Category of the market to which the product belongs to

Product_Category: Category of the product
Grade: Quality of the product
Demand: Demand rate of the product in the market
Low_Cap_Price [Target]: Lowest price that can be offered 
High_Cap_Price: Original maximum price in the current market
Data files

The dataset folder consists of the following files:

__Train.csv__: Contains training data [9798 x 9] that must be used to build the model
__Test.csv__: Contains test data [5763 x 8] to be predicted on
__sample_submission.csv__: Contains sample submission format with dummy values filled for test data

### Evaluation metric
score = max(0,(100-mean_squared_log_error(actual_values,predicted_values)))