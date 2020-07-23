{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "evalue": "Error: Activating Python 3.7.1 64-bit to run Jupyter failed with Error: StdErr from ShellExec, failed to create process.\r\n.",
     "output_type": "error"
    }
   ],
   "source": [
    "#importing the datasets from the libraries\n",
    "import pandas as pd\n",
    "test = pd.read_csv(\"C:/Users/kanum/Desktop/Akshata/DSBA/mini_projects/Loan_Prediction/loan_test_data.csv\")\n",
    "test.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP001002</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001003</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4583</td>\n",
       "      <td>1508.0</td>\n",
       "      <td>128.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>N</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP001005</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP001006</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2583</td>\n",
       "      <td>2358.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP001008</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>6000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>141.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Loan_ID Gender Married Dependents     Education Self_Employed  \\\n",
       "0  LP001002   Male      No          0      Graduate            No   \n",
       "1  LP001003   Male     Yes          1      Graduate            No   \n",
       "2  LP001005   Male     Yes          0      Graduate           Yes   \n",
       "3  LP001006   Male     Yes          0  Not Graduate            No   \n",
       "4  LP001008   Male      No          0      Graduate            No   \n",
       "\n",
       "   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0             5849                0.0         NaN             360.0   \n",
       "1             4583             1508.0       128.0             360.0   \n",
       "2             3000                0.0        66.0             360.0   \n",
       "3             2583             2358.0       120.0             360.0   \n",
       "4             6000                0.0       141.0             360.0   \n",
       "\n",
       "   Credit_History Property_Area Loan_Status  \n",
       "0             1.0         Urban           Y  \n",
       "1             1.0         Rural           N  \n",
       "2             1.0         Urban           Y  \n",
       "3             1.0         Urban           Y  \n",
       "4             1.0         Urban           Y  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train = pd.read_csv(\"C:/Users/kanum/Desktop/Akshata/DSBA/mini_projects/Loan_Prediction/loan_train_data.csv\")\n",
    "train.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing the libraries from pyhton\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(614, 13)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(367, 12)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Loan_ID               0\n",
       "Gender               13\n",
       "Married               3\n",
       "Dependents           15\n",
       "Education             0\n",
       "Self_Employed        32\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount           22\n",
       "Loan_Amount_Term     14\n",
       "Credit_History       50\n",
       "Property_Area         0\n",
       "Loan_Status           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check for null values in train\n",
    "train.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Loan_ID               0\n",
       "Gender               11\n",
       "Married               0\n",
       "Dependents           10\n",
       "Education             0\n",
       "Self_Employed        23\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount            5\n",
       "Loan_Amount_Term      6\n",
       "Credit_History       29\n",
       "Property_Area         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check for null values in test\n",
    "test.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Male      489\n",
       "Female    112\n",
       "Name: Gender, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Genderwise count\n",
    "train['Gender'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Male      286\n",
       "Female     70\n",
       "Name: Gender, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test['Gender'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#combining test and train data\n",
    "#Loan_Status of train is the targets\n",
    "#Appending Train and Test data\n",
    "#Drop the Loan_ID\n",
    "def get_combined_data():\n",
    "    train1 = pd.read_csv(\"C:/Users/kanum/Desktop/Akshata/DSBA/mini_projects/Loan_Prediction/loan_train_data.csv\")\n",
    "    test1 = pd.read_csv(\"C:/Users/kanum/Desktop/Akshata/DSBA/mini_projects/Loan_Prediction/loan_test_data.csv\")\n",
    "    targets = train1.Loan_Status\n",
    "    train1.drop('Loan_Status', 1, inplace=True)\n",
    "    combined = train1.append(test1)\n",
    "    combined.reset_index(inplace=True)\n",
    "    combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)\n",
    "    return combined"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>981.000000</td>\n",
       "      <td>981.000000</td>\n",
       "      <td>954.000000</td>\n",
       "      <td>961.000000</td>\n",
       "      <td>902.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>5179.795107</td>\n",
       "      <td>1601.916330</td>\n",
       "      <td>142.511530</td>\n",
       "      <td>342.201873</td>\n",
       "      <td>0.835920</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>5695.104533</td>\n",
       "      <td>2718.772806</td>\n",
       "      <td>77.421743</td>\n",
       "      <td>65.100602</td>\n",
       "      <td>0.370553</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2875.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>360.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>3800.000000</td>\n",
       "      <td>1110.000000</td>\n",
       "      <td>126.000000</td>\n",
       "      <td>360.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>5516.000000</td>\n",
       "      <td>2365.000000</td>\n",
       "      <td>162.000000</td>\n",
       "      <td>360.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>81000.000000</td>\n",
       "      <td>41667.000000</td>\n",
       "      <td>700.000000</td>\n",
       "      <td>480.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "count       981.000000         981.000000  954.000000        961.000000   \n",
       "mean       5179.795107        1601.916330  142.511530        342.201873   \n",
       "std        5695.104533        2718.772806   77.421743         65.100602   \n",
       "min           0.000000           0.000000    9.000000          6.000000   \n",
       "25%        2875.000000           0.000000  100.000000        360.000000   \n",
       "50%        3800.000000        1110.000000  126.000000        360.000000   \n",
       "75%        5516.000000        2365.000000  162.000000        360.000000   \n",
       "max       81000.000000       41667.000000  700.000000        480.000000   \n",
       "\n",
       "       Credit_History  \n",
       "count      902.000000  \n",
       "mean         0.835920  \n",
       "std          0.370553  \n",
       "min          0.000000  \n",
       "25%          1.000000  \n",
       "50%          1.000000  \n",
       "75%          1.000000  \n",
       "max          1.000000  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Checking the pattern of combined data\n",
    "combined = get_combined_data()\n",
    "combined.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Imputing the Gender with domainant Male\n",
    "def impute_gender():\n",
    "    global combined\n",
    "    combined['Gender'].fillna('Male', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Imputing with dominant Married Status 'Yes'\n",
    "def impute_martial_status():\n",
    "    global combined\n",
    "    combined['Married'].fillna('Yes', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Imputing with dominant Self_Employment 'No'\n",
    "def impute_employment():\n",
    "    global combined\n",
    "    combined['Self_Employed'].fillna('No', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Imputing the LoanAmount with mean \n",
    "def impute_loan_amount():\n",
    "    global combined\n",
    "    combined['LoanAmount'].fillna(combined['LoanAmount'].mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Credit_\n",
    "def impute_credit_history():\n",
    "    global combined\n",
    "    combined['Credit_History'].fillna(2, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0    754\n",
       "0.0    148\n",
       "Name: Credit_History, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined['Credit_History'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "impute_gender()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "impute_martial_status()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "impute_employment()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "impute_loan_amount()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "impute_credit_history()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender                0\n",
       "Married               0\n",
       "Dependents           25\n",
       "Education             0\n",
       "Self_Employed         0\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount            0\n",
       "Loan_Amount_Term     20\n",
       "Credit_History        0\n",
       "Property_Area         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_gender():\n",
    "    global combined\n",
    "    combined['Gender'] = combined['Gender'].map({'Male':1,'Female':0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_martial_status():\n",
    "    global combined\n",
    "    combined['Married'] = combined['Married'].map({'Yes':1,'No':0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_dependents():\n",
    "    global combined\n",
    "    combined['Micro'] = combined['Dependents'].map(lambda d: 1 if d=='1' else 0)\n",
    "    combined['Family'] = combined['Dependents'].map(lambda d: 1 if d=='2' else 0)\n",
    "    combined['Large'] = combined['Dependents'].map(lambda d: 1 if d=='3+' else 0)\n",
    "    combined.drop(['Dependents'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "def process_education():\n",
    "    global combined\n",
    "    combined['Education'] = combined['Education'].map({'Graduate':1,'Not Graduate':0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_employment():\n",
    "    global combined\n",
    "    combined['Self_Employed'] = combined['Self_Employed'].map({'Yes':1,'No':0})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Combined Applicant and Co-Applicant Income\n",
    "def process_income():\n",
    "    global combined\n",
    "    combined['Total_Income'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']\n",
    "    combined.drop(['ApplicantIncome','CoapplicantIncome'], axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "#def process_loan_amount():\n",
    "#    global combined\n",
    "#    combined['Debt_Income_Ratio'] = combined['Total_Income'] / combined['LoanAmount']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "360.0    823\n",
       "180.0     66\n",
       "480.0     23\n",
       "300.0     20\n",
       "240.0      8\n",
       "84.0       7\n",
       "120.0      4\n",
       "36.0       3\n",
       "60.0       3\n",
       "12.0       2\n",
       "350.0      1\n",
       "6.0        1\n",
       "Name: Loan_Amount_Term, dtype: int64"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined['Loan_Amount_Term'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "#approved_term = train[train['Loan_Status']=='Y']['Loan_Amount_Term'].value_counts()\n",
    "#unapproved_term = train[train['Loan_Status']=='N']['Loan_Amount_Term'].value_counts()\n",
    "#df = pd.DataFrame([approved_term,unapproved_term])\n",
    "#df.index = ['Approved','Unapproved']\n",
    "#df.plot(kind='bar', stacked=True, figsize=(15,8))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_loan_term():\n",
    "    global combined\n",
    "    combined['Very_Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t<=60 else 0)\n",
    "    combined['Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>60 and t<180 else 0)\n",
    "    combined['Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>=180 and t<=300  else 0)\n",
    "    combined['Very_Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>300 else 0)\n",
    "    combined.drop('Loan_Amount_Term', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_credit_history():\n",
    "    global combined\n",
    "    combined['Credit_History_Bad'] = combined['Credit_History'].map(lambda c: 1 if c==0 else 0)\n",
    "    combined['Credit_History_Good'] = combined['Credit_History'].map(lambda c: 1 if c==1 else 0)\n",
    "    combined['Credit_History_Unknown'] = combined['Credit_History'].map(lambda c: 1 if c==2 else 0)\n",
    "    combined.drop('Credit_History', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_property():\n",
    "    global combined\n",
    "    property_dummies = pd.get_dummies(combined['Property_Area'], prefix='Property')\n",
    "    combined = pd.concat([combined, property_dummies], axis=1)\n",
    "    combined.drop('Property_Area', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_gender()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_martial_status()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_dependents()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_education()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_employment()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_income()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "#process_loan_amount()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_loan_term()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_credit_history()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "process_property()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Micro</th>\n",
       "      <th>Family</th>\n",
       "      <th>Large</th>\n",
       "      <th>Total_Income</th>\n",
       "      <th>Very_Short_Term</th>\n",
       "      <th>Short_Term</th>\n",
       "      <th>Long_Term</th>\n",
       "      <th>Very_Long_Term</th>\n",
       "      <th>Credit_History_Bad</th>\n",
       "      <th>Credit_History_Good</th>\n",
       "      <th>Credit_History_Unknown</th>\n",
       "      <th>Property_Rural</th>\n",
       "      <th>Property_Semiurban</th>\n",
       "      <th>Property_Urban</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>60</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>120.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6296.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>61</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>99.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3029.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>62</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>165.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6058.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>63</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>142.51153</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4945.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>116.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4166.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>65</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>258.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10321.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>66</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>126.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5454.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>67</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>312.00000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10750.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>68</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>125.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7100.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>69</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>136.00000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4300.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Gender  Married  Education  Self_Employed  LoanAmount  Micro  Family  \\\n",
       "60       1        1          1              0   120.00000      0       0   \n",
       "61       1        1          1              0    99.00000      0       0   \n",
       "62       1        1          0              1   165.00000      0       0   \n",
       "63       1        1          1              0   142.51153      1       0   \n",
       "64       0        0          1              0   116.00000      0       0   \n",
       "65       1        1          1              0   258.00000      0       0   \n",
       "66       1        0          0              0   126.00000      0       0   \n",
       "67       1        1          1              0   312.00000      1       0   \n",
       "68       1        1          0              1   125.00000      0       0   \n",
       "69       0        0          1              0   136.00000      0       0   \n",
       "\n",
       "    Large  Total_Income  Very_Short_Term  Short_Term  Long_Term  \\\n",
       "60      0        6296.0                0           0          0   \n",
       "61      1        3029.0                0           0          0   \n",
       "62      0        6058.0                0           0          1   \n",
       "63      0        4945.0                0           0          0   \n",
       "64      0        4166.0                0           0          0   \n",
       "65      0       10321.0                0           0          0   \n",
       "66      0        5454.0                0           0          1   \n",
       "67      0       10750.0                0           0          0   \n",
       "68      1        7100.0                1           0          0   \n",
       "69      0        4300.0                0           0          0   \n",
       "\n",
       "    Very_Long_Term  Credit_History_Bad  Credit_History_Good  \\\n",
       "60               1                   0                    1   \n",
       "61               1                   0                    1   \n",
       "62               0                   1                    0   \n",
       "63               1                   1                    0   \n",
       "64               1                   1                    0   \n",
       "65               1                   0                    1   \n",
       "66               0                   1                    0   \n",
       "67               1                   0                    1   \n",
       "68               0                   0                    1   \n",
       "69               1                   1                    0   \n",
       "\n",
       "    Credit_History_Unknown  Property_Rural  Property_Semiurban  Property_Urban  \n",
       "60                       0               0                   0               1  \n",
       "61                       0               0                   0               1  \n",
       "62                       0               1                   0               0  \n",
       "63                       0               1                   0               0  \n",
       "64                       0               0                   1               0  \n",
       "65                       0               0                   1               0  \n",
       "66                       0               0                   0               1  \n",
       "67                       0               0                   0               1  \n",
       "68                       0               0                   0               1  \n",
       "69                       0               0                   1               0  "
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined[60:70]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "def feature_scaling(df):\n",
    "    df -= df.min()\n",
    "    df /= df.max()\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Micro</th>\n",
       "      <th>Family</th>\n",
       "      <th>Large</th>\n",
       "      <th>Total_Income</th>\n",
       "      <th>Very_Short_Term</th>\n",
       "      <th>Short_Term</th>\n",
       "      <th>Long_Term</th>\n",
       "      <th>Very_Long_Term</th>\n",
       "      <th>Credit_History_Bad</th>\n",
       "      <th>Credit_History_Good</th>\n",
       "      <th>Credit_History_Unknown</th>\n",
       "      <th>Property_Rural</th>\n",
       "      <th>Property_Semiurban</th>\n",
       "      <th>Property_Urban</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>200</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.117221</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.045979</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>201</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.227207</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.043754</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>202</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.193215</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0.032052</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>203</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.182344</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.039481</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>204</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.166425</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.031109</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>205</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.160637</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.037281</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>206</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.102750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.022650</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>207</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.066570</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.063652</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>208</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.072359</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.013035</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>209</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.170767</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0.024837</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender  Married  Education  Self_Employed  LoanAmount  Micro  Family  \\\n",
       "200       1        1          0              0    0.117221      1       0   \n",
       "201       1        0          1              0    0.227207      0       1   \n",
       "202       1        1          0              0    0.193215      0       0   \n",
       "203       1        1          0              0    0.182344      1       0   \n",
       "204       1        1          0              0    0.166425      0       1   \n",
       "205       0        0          0              0    0.160637      0       0   \n",
       "206       0        0          1              0    0.102750      0       0   \n",
       "207       1        0          0              0    0.066570      0       0   \n",
       "208       1        0          1              0    0.072359      0       0   \n",
       "209       1        0          1              0    0.170767      0       0   \n",
       "\n",
       "     Large  Total_Income  Very_Short_Term  Short_Term  Long_Term  \\\n",
       "200      0      0.045979                0           0          0   \n",
       "201      0      0.043754                0           0          0   \n",
       "202      1      0.032052                0           0          1   \n",
       "203      0      0.039481                0           0          0   \n",
       "204      0      0.031109                0           0          0   \n",
       "205      0      0.037281                0           0          0   \n",
       "206      0      0.022650                0           0          0   \n",
       "207      0      0.063652                0           0          0   \n",
       "208      0      0.013035                0           0          0   \n",
       "209      0      0.024837                0           0          0   \n",
       "\n",
       "     Very_Long_Term  Credit_History_Bad  Credit_History_Good  \\\n",
       "200               1                   0                    1   \n",
       "201               1                   1                    0   \n",
       "202               0                   0                    1   \n",
       "203               1                   0                    1   \n",
       "204               1                   0                    1   \n",
       "205               1                   0                    1   \n",
       "206               1                   0                    1   \n",
       "207               1                   0                    1   \n",
       "208               1                   0                    1   \n",
       "209               1                   0                    1   \n",
       "\n",
       "     Credit_History_Unknown  Property_Rural  Property_Semiurban  \\\n",
       "200                       0               0                   1   \n",
       "201                       0               0                   1   \n",
       "202                       0               0                   0   \n",
       "203                       0               0                   0   \n",
       "204                       0               0                   1   \n",
       "205                       0               0                   1   \n",
       "206                       0               0                   0   \n",
       "207                       0               1                   0   \n",
       "208                       0               0                   0   \n",
       "209                       0               0                   1   \n",
       "\n",
       "     Property_Urban  \n",
       "200               0  \n",
       "201               0  \n",
       "202               1  \n",
       "203               1  \n",
       "204               0  \n",
       "205               0  \n",
       "206               1  \n",
       "207               0  \n",
       "208               1  \n",
       "209               0  "
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "combined[200:210]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.feature_selection import SelectFromModel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_score(clf, X, y, scoring='accuracy'):\n",
    "    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)\n",
    "    return np.mean(xval)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "def recover_train_test_target():\n",
    "    global combined, train\n",
    "    targets = train['Loan_Status'].map({'Y':1,'N':0})\n",
    "    train = combined.head(614)\n",
    "    test = combined.iloc[614:]\n",
    "    return train, test, targets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "train, test, targets = recover_train_test_target()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = RandomForestClassifier(n_estimators=500, max_features='auto')\n",
    "clf = clf.fit(train, targets)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "features = pd.DataFrame()\n",
    "features['Feature'] = train.columns\n",
    "features['Importance'] = clf.feature_importances_\n",
    "features.sort_values(by=['Importance'], ascending=False, inplace=True)\n",
    "features.set_index('Feature', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2b3be491c88>"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABIsAAALACAYAAAATwPL7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzs3Xu85XV93/v3hxlxvHIEpznIiIMGQU5B4AyTHqm3JAKpt/QEFS+p2BBqPZi2OfHR6SN9aA9pT0jiOWop9dJKMEaPBNOmRMlDMYrReJtREYOoXDrqlDQl4AOJiDLwOX+sxZftMDALZu+9ZvZ+Ph+Pecz+/dbvt9b3Owyz136t36W6OwAAAACQJAfMewAAAAAA7DvEIgAAAAAGsQgAAACAQSwCAAAAYBCLAAAAABjEIgAAAAAGsQgAAACAQSwCAAAAYBCLAAAAABjWznsAu3r84x/fGzdunPcwAAAAAFaML37xi3/d3etn2Xafi0UbN27Mtm3b5j0MAAAAgBWjqr4167ZOQwMAAABgEIsAAAAAGMQiAAAAAIZ97ppFAAAAwMpy5513ZseOHbnjjjvmPZQVb926ddmwYUMe9rCHPeTnEIsAAACAJbVjx4485jGPycaNG1NV8x7OitXdufnmm7Njx44cccQRD/l5nIYGAAAALKk77rgjhxxyiFC0xKoqhxxyyF4fwSUWAQAAAEtOKFoei/HnLBYBAAAAMLhmEQAAALCsNm758KI+3/bznr/HbR796Efnb/7mbxb1dR/I9u3b85nPfCaveMUrlu01F4sjiwAAAAAW0c6dO7N9+/a8//3vn/dQHhKxCAAAAFg1rrjiijz72c/OS1/60jz1qU/Nli1b8r73vS+bN2/Osccem+uvvz5JcuaZZ+a1r31tnvnMZ+apT31qPvShDyWZXKz7Na95TY499ticcMIJ+cQnPpEkueiii/KSl7wkL3zhC3PKKadky5Yt+dSnPpXjjz8+b3nLW7J9+/Y885nPzIknnpgTTzwxn/nMZ8Z4nvOc5+T000/P0UcfnVe+8pXp7iTJ1q1b84xnPCNPf/rTs3nz5tx2222566678oY3vCEnnXRSjjvuuLzzne9c9D8jp6EBAAAAq8pXvvKVXHPNNTn44IPz5Cc/OWeddVa+8IUv5G1ve1vOP//8vPWtb00yOZXsk5/8ZK6//vo897nPzXXXXZcLLrggSfLVr341X//613PKKafkm9/8ZpLks5/9bK666qocfPDBueKKK/LmN795RKbbb789l19+edatW5drr702L3/5y7Nt27YkyZe//OVcffXVecITnpCTTz45f/7nf57NmzfnZS97WS6++OKcdNJJ+d73vpdHPOIRefe7352DDjooW7duzQ9/+MOcfPLJOeWUU3LEEUcs2p+PWAQAAACsKieddFIOPfTQJMlTnvKUnHLKKUmSY489dhwplCQvfelLc8ABB+TII4/Mk5/85Hz961/Ppz/96bz+9a9Pkhx99NF50pOeNGLR8573vBx88MG7fc0777wz55xzTq688sqsWbNm7JMkmzdvzoYNG5Ikxx9/fLZv356DDjoohx56aE466aQkyWMf+9gkyUc/+tFcddVV+eAHP5gkufXWW3PttdeKRQAAAAAP1cMf/vDx9QEHHDCWDzjggOzcuXM8tutt6KtqnCK2O4961KPu97G3vOUt+Ymf+Il85Stfyd13351169btdjxr1qzJzp070933ef0k6e6cf/75OfXUUx9ghnvHNYsAAAAAduOSSy7J3Xffneuvvz433HBDjjrqqDzrWc/K+973viTJN7/5zXz729/OUUcddZ99H/OYx+S2224by7feemsOPfTQHHDAAXnve9+bu+666wFf++ijj86NN96YrVu3Jkluu+227Ny5M6eeemre/va358477xxj+P73v79YU07iyCIAAABgmc1yq/t9wVFHHZVnP/vZ+au/+qu84x3vyLp16/K6170ur33ta3Psscdm7dq1ueiii37syKB7HHfccVm7dm2e/vSn58wzz8zrXve6/MIv/EIuueSSPPe5z33Ao5CS5MADD8zFF1+c17/+9fnBD36QRzziEfnYxz6Ws846K9u3b8+JJ56Y7s769evzR3/0R4s673qgw6fGRlWnJXlbkjVJ/mN3n7fL47+a5KwkO5PclOQfdve3po/dleSr002/3d0veqDX2rRpU99zgScAAABg/3fNNdfkaU972ryH8aCceeaZecELXpDTTz993kN50Hb3511VX+zuTbPsv8cji6pqTZILkjwvyY4kW6vq0u7+2oLNvpxkU3ffXlX/OMlvJ3nZ9LEfdPfxswwGAAAAgPma5TS0zUmu6+4bkqSqPpDkxUlGLOruTyzY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bP+mVraIiIiIiCgyWBQREWMnaVdgq27zCtun18wTERERERGLZLAoIiIiIiIiIiIWSs2iiIiIiIiIiIhYKINFERERERERERGxUAaLIiIionck3SXpkqGPTZfjv7GepFfNfrqIiIiIulKzKCIiInpH0p9sr7WC/41NgW/Y3mopXzr1+1ayfdeK/NsRERERo5SZRRERERGUQRxJH5B0oaTLJL2i27+WpNMkXSzpckl/233L+4CHdzOTPiDpqZK+MfTf+5ikl3SPr5H0VknnAPtIerikb0uaL+l7krYc9/FGRERELMl9ageIiIiIqGB1SZd0j6+2vRdwEPBH29tLWhU4V9J3gV8Ae9m+SdL9gfMlnQy8AdjK9jYAkp66lH/zNts7d197GnCw7Z9I2hH4D2C32T7IiIiIiOWRwaKIiIjooz8PBnmGPB14nKTnd9vrAo8A/g94r6SnAHcDGwMbLse/eQKUmUrAk4EvSxo8t+py/PciIiIiRiKDRRERERGFgNfY/s5iO8tSsg2A7WzfIekaYLVpvv9OFl/iP/Vrbuk+zwP+MM1gVURERMSckJpFEREREcV3gFdKWhlA0haS1qTMMLquGyjaFXho9/U3A2sPff+1wKMlrSppXWD36f4R2zcBV0vap/t3JGnr0RxSRERExL2XwaKIiIiI4jPAlcDFkn4IfIoyC/s44AmSLgL2B34EYPt3lLpGP5T0Adu/AL4EXNZ9z4IZ/q39gYMkXQpcAfztDF8bERERMVayXTtDRERERERERETMEZlZFBERERERERERC2WwKCIiIiIiIiIiFspgUURERERERERELJTBooiIiIiIiIiIWCiDRRERERERERERsVAGiyIiIiIiIiIiYqEMFkVERERERERExEIZLIqIiIiIiIiIiIX+P4yudVGj5ogBAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "features.plot(kind='bar', figsize=(20, 10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(614, 4)"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = SelectFromModel(clf, prefit=True)\n",
    "train_reduced = model.transform(train)\n",
    "train_reduced.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(367, 4)"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_reduced = model.transform(test)\n",
    "test_reduced.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',\n",
       "            max_depth=15, max_features='auto', max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=15, min_samples_split=15,\n",
       "            min_weight_fraction_leaf=0.0, n_estimators=15, n_jobs=None,\n",
       "            oob_score=False, random_state=None, verbose=0,\n",
       "            warm_start=False)"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "parameters = {'bootstrap': False,\n",
    "              'min_samples_leaf': 15,\n",
    "              'n_estimators': 15,\n",
    "              'min_samples_split': 15,\n",
    "              'max_features': 'auto',\n",
    "              'max_depth': 15}\n",
    "\n",
    "model = RandomForestClassifier(**parameters)\n",
    "model.fit(train, targets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8095716552088842"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compute_score(model, train, targets, scoring='accuracy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "output = model.predict(test).astype(int)\n",
    "df_output = pd.DataFrame()\n",
    "aux = pd.read_csv(\"C:/Users/kanum/Desktop/Akshata/DSBA/mini_projects/Loan_Prediction/loan_test_data.csv\")\n",
    "df_output['Loan_ID'] = aux['Loan_ID']\n",
    "df_output['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)\n",
    "df_output[['Loan_ID','Loan_Status']].to_csv('output.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}