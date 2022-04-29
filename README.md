# Predicting Life Expectancy (WHO)
Kristofer Rivera -- April 2022
 
===
 
Table of Contents
---
 
* I. [Project Overview](#i-project-overview)<br>
[1. Goals](#1-goal)<br>
[2. Description of Data](#2-description)<br>
[3. Process](#3-description-of-data)<br>
[4. Initial Questions](#4initial-questions)<br>
[5. Deliverables](#5-deliverables)<br>
* II. [Project Data Context](#ii-project-data-context)<br>
[1. Data Dictionary](#1-data-dictionary)<br>
* III. [Project Plan - Data Science Pipeline](#iii-project-plan---using-the-data-science-pipeline)<br>
[1. Project Planning](#1-plan)<br>
[2. Data Acquisition](#2-acquire)<br>
[3. Data Preparation](#3-prepare)<br>
[4. Data Exploration](#4explore)<br>
[5. Modeling & Evaluation](#5-model--evaluate)<br>
[6. Product Delivery](#6-delivery)<br>
* IV. [Project Modules](#iv-project-modules)<br>
* V. [Project Reproduction](#v-project-reproduction)<br>
 
 
 
## I. PROJECT OVERVIEW
 
 
#### 1.  GOAL:
The goal of this project is to use economic and public health data acquired from the World Health Organization (WHO) to create a machine-learning model that can predict a country's life expectancy. My model and insights from data analysis can be used to guide government policy making towards improving the average life expectancy of its population.
 
#### 2. DESCRIPTION OF DATA:
The data set was acquired from Kaggle user KUMARRAJARSHI: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

"Although there have been lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition and mortality rates. It was found that affect of immunization and human development index was not taken into account in the past. Also, some of the past research was done considering multiple linear regression based on data set of one year for all the countries. Hence, this gives motivation to resolve both the factors stated previously by formulating a regression model based on mixed effects model and multiple linear regression while considering data from a period of 2000 to 2015 for all the countries. Important immunization like Hepatitis B, Polio and Diphtheria will also be considered. In a nutshell, this study will focus on immunization factors, mortality factors, economic factors, social factors and other health related factors as well. Since the observations this dataset are based on different countries, it will be easier for a country to determine the predicting factor which is contributing to lower value of life expectancy. This will help in suggesting a country which area should be given importance in order to efficiently improve the life expectancy of its population."

#### 3. PROCESS:

Upon successful acquistion of the *WHO* dataset from Kaggle. It was then transformed into a series of DataFrames which could be used in exploration and model creation. The intial raw dataframe consisted of, 2938 rows and 22 columns. For data cleaning, columns were renames, null values imputed for using the mean and a dummy variable was created for development status. My cleaned data set had the same amount of rows and columns. Through statistical testing and exploratory analysis, I was able to determine top drivers of life expectancy including: 'income_composition_of_resources', 'schooling', 'hiv_aids', 'gdp', 'developing', and 'bmi.' Using these drivers as features, I built several types of regression models to predict life expectancy and evaluated my best model, a polynomial regression model, on out-of-sample data. This model was able to predict life_expectancy with 83% accuracy and a RMSE score on 3.81. 
 

#### 4.INITIAL QUESTIONS:
The focus of the project is identifying the best product category to expand on. Below are some of the initial questions this project looks to answer throughout the Data Science Pipeline.
 
#### Data-Focused Questions:
1. Is there a linear relationship between income composition of resources and life expectancy?

2. Is there a linear relationship between schooling and life expectancy? Is this a real correlation or is it because schooling is correlated with income composition? Which is the more important causal variable for life expectancy?

3. Is there a significant relationship between status as a developing country and infant mortality?

4. Is there a signifiant relationship between gdp and life expectancy?

5. Is there a linear relationship between hiv_aids prevelance and life expectancy?

6. Is there a linear relationship between bmi and life expectancy?

  
#### 5. DELIVERABLES:
- [x] README file - provides an overview of the project and steps for project reproduction
- [x] Draft Jupyter Notebook - provides all steps taken to produce the project
- [x] Final Jupyter Notebook - provides final presentation-ready wrangle, exploration and modeling
 
 
## II. PROJECT DATA CONTEXT
           
#### 1. DATA DICTIONARY:
The final DataFrame used to explore the data for this project contains the following variables (columns).  The variables, along with their data types, are defined below:
 
 
|  Variables                    |    Definition                                                |    DataType        |
| :--------------------:        | :----------------------------------------:                   | :---------------:  |
country                         |  Country of origin                                           |  object            |
year                            |  Year data acquired                                          |  int64             |
life_expectancy                 |  Average life expectancy of population                       |  float64           |
adult_mortality                 |  Rate of adult mortality                                     |  float64           |
infant_deaths                   |  Rate of infant mortality                                    |  int64             |
alcohol                         |  Rate of alcoholism in population                            |  float64           |
percentage_expenditure          |  Rate of ?                                                   |  float64           |
hepatitis_b                     |  Rate of Hepatitis B immunization                            |  float64           |
measles                         |  Rate of Measles immunization                                |  int64             |
bmi                             |  Average Body Mass Index for population                      |  float64           |
under_five_deaths               |  Rate of mortality for children under 5                      |  int64             |
polio                           |  Rate of Polio immunization                                  |  float64           |
total_expenditure               |  Total expenditure? Not positive                             |  float64           |
diphtheria                      |  Name of product                                             |  float64           |
hiv_aids                        |  Prevelance of HIV/AIDS in the population                    |  float64           |
gdp                             |  Gross Domsetic Product of country                           |  float64           |
population                      |  Population of Country                                       |  float64           |
thinness_1to19_years            |  Thinness index from 1 - 19 years old                        |  float64           |
thinness_5to19_years            |  Thinness index from 1 - 19 years old                        |  float64           |
income_composition_of_resources |  Income composition of resources                             |  float64           |
schooling                       |  Average level of schooling                                  |  float64           |
developing                      |  Classified as developing country                            |  uint8             |


 
## III. PROJECT PLAN - USING THE DATA SCIENCE PIPELINE:
The following outlines the process taken through the Data Science Pipeline to complete this project. 
 
Plan➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver Final Notebook
 
#### 1. PLAN
- [x]  Review project expectations
- [x]  Draft project goal to include measures of success
- [x]  Create questions related to the project
- [x]  Create questions related to the data
- [x]  Create a plan for completing the project using the data science pipeline
- [x]  Create a data dictionary to define variables and data context
 
#### 2. ACQUIRE
- [x]  Create .gitignore to store files and ensure the security of sensitive data
- [x]  Create wrangle.py to store functions needed to acquire the Superstore dataset from Codeup database
- [x]  Ensure all imports needed to run the functions are inside the wrangle.py document
- [x]  Using Jupyter Notebook
     - [x]  Run all required imports
     - [x]  Import functions from wrangle.py module
     - [x]  Summarize dataset using methods and document observations
 
#### 3. PREPARE
Using Jupyter Notebook
- [x]  Import functions from wrangle.py module
- [x]  Summarize dataset using methods and document observations
- [x]  Clean data, rename columns, change datatypes
- [x]  Categorical features or discrete features need to be numbers that represent those categories
- [x]  Continuous features may need to be standardized to compare like datatypes
- [x]  Address missing values, data errors, unnecessary data
- [x]  Split data into train, validate, and test samples
Using Python Scripting Program (Jupyter Notebook)
- [x]  Create prepare function within wrangle.py
- [x]  Store functions needed to prepare the Superstore data such as:
   - [x]  Cleaning Function: to clean data for exploration
   - [x]  Encoding Function: to create datetime columns for order date and ship date columns and set order date as an index
   - [x]  Split Function: to split data into train, validate, and test
- [x]  Ensure all imports needed to run the functions are inside the wrangle.py document
 
#### 4.EXPLORE
Using Jupyter Notebook:
- [x]  Answer key questions and find the best category in regards to sales
     - [x]  Document findings
- [x]  Create visualizations with the intent to discover variable relationships
     - [x]  Identify variables related to catgory and sales
     - [x]  Identify any potential data integrity issues
- [x]  Summarize conclusions, provide clear answers, and summarize takeaways
     - [x] Explain plan of action as deduced from work to this point
 
#### 5. MODEL & EVALUATE
Using Jupyter Notebook:
- [x]  Train and fit multiple models with varying algorithms and/or hyperparameters
- [x]  Remove unnecessary features
- [x]  Evaluate best performing models using validate set
- [x]  Choose best performing validation model for use on test set
- [x]  Test final model on out-of-sample testing dataset
- [x]  Summarize performance
- [x]  Interpret and document findings
 
#### 6. DELIVERY
- [x]  Prepare final notebook in Jupyter Notebook
     - [x]  Create clear walk-though of the Data Science Pipeline using headings and dividers
     - [x]  Explicitly define questions asked during the initial analysis
     - [x]  Visualize relationships
     - [x]  Document takeaways
     - [x]  Comment code thoroughly


## IV. PROJECT MODULES:
- [x] wrangle.py - provides reproducible python code to automate acquiring, preparing, and splitting the data
- [x] evaluate.py - provides reproducible python code to automate model evaluation and visualization
- [x] explore.py - provides reproducible python code to automate exploratory data analyis and statistcal testing
 
  
## V. PROJECT REPRODUCTION:
### Steps to Reproduce
- [x] You'll need to download the .csv from Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
- [x] Clone our repo (including the wrangle.py, explore.py, evaluate.py)
- [x] Import python libraries:  pandas, matplotlib, seaborn, numpy, scipy, sklearn
- [x] Follow steps as outlined in the README.md. and draft notebooks
- [x] Run final_notebook.ipynb to view the final product
