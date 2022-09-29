
# Stores Sales Prediction | iNeuron Internship

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Technologies Used](#technologies-used)
  * [Credits](#credits)


## Demo
Link: [https://item-sales-predictions.herokuapp.com/](https://item-sales-predictions.herokuapp.com/)


<img src="images\Project1.png" alt="Project UI/UX" />

## Overview
This is End to End machine learning Regression project which takes information related to product in the store as input and predict the sales of that product.
The [dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) used for model training is from kaggle.Dataset contains 8523 entries of data of various items , 11 features of item and 1 sales as output feature. 


## Motivation
Shoping malls and Big Marts keep trach of individual item sales data in order to forcast future client demand and adjust inventory management.In fastival season it is difficult to manage inventory manually but by using Machine Learning we can give information regarding Item as input to model and it will predict the sales of that item which ultimatly help to inventory management and increase in the sales.

## Technical Aspect
Every End - End Project has life cycle to get desired output. 
1. Data Collection 
    - [Downloaded](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data) data from kaggle website. 
2. Exploratory data analysis
    - Analyzed data using visual & Statistical techniques 
    - Univarient Analysis observations 
        - Items with low fat are bought more 
        <img src="images\FAT_content.png" alt="FAT_CONTENT" />
        - Fruits and vegetables largely sold and also snacks also have good sales 
        <img src="images\fruits.png" alt="fruits" />
        - Medium size stores/malls have more sales.
        <img src="images\outlet_sizw.png" alt="outlet_sizw" />
        - more number of stores/malls located in tier 3 cities 
        <img src="images\tier3.png" alt="tier3" />
        - stores/malls are more of Supermarket type 1
        <img src="images\supermarket.png" alt="supermarket" />
    - Bivarient Analysis obseravations 
        - sales are high for both low and regular fat items
        <img src="images\fat_sales.png" alt="fat_sales" /> 
        - Item Visibility cannot be zero.(This is error because product may rarly purchased)
        <img src="images\visibility.png" alt="visibility" />
        - The sales of seafood and starchy food higher and sales can be improved with having stock of this type of products 
        <img src="images\sea_food.png" alt="sea_food" />
        - Item with MRP 200 -250 dollers having more sale.
        <img src="images\mrp.png" alt="mrp" />
        - stores/malls established 28 years before having good sales margin
        <img src="images\age.png" alt="age" />
    
3. Feature Engineering 
    - Data Cleaning 
        - KNN imputer is used to handle missing values 
        - Lable Encoding is used to convert categorical values into numerical values 
        - Outliers checking done by BoxPlot Method
    - Feature Scaling 
        - Standard Scaling operations are applied to scale the data 
4. Feature Selection 
    - Correlation method is used to check internal correlated features.
    - Used RandomForest Feature Importance to select important features   
5. Model Building
    - Trained data by using Linear Regrssion, Random Forest and XGBoost algorithms.
    - Model Accuracy 
| Left-aligned | Center-aligned | Right-aligned |
| :---         |     :---:      |          ---: |
| git status   | git status     | git status    |
| git diff     | git diff       | git diff      |
    - Random Forest with high accuracy is selcted.
    - Model Hyperparameter is done by using Grid Search Cv 
    - Model Evaluated with R2 Score and RMSE score 
6. Pipeline 
    Sequence of data preprocessing components is called data pipeline. 
    1. Data Ingestion 
        - Download the data from source, extract it, split into train and test dataset and store in the destination
    2. Data Validation 
        - Validate data so noise data will not come in the piepline 
    3. Data Transformation 
        - Apply Feature Engineering , Feature Selction processes on data and store transformed data into destination 
    4. Model Trainer 
        - Training the Random Forest model with tuned parameters 
    5. Model Evaluation 
        - Evaluation done by comparing model's accuracy with base accuracy and recent model. 
    6. Model Pusher 
        - If accuracy of trained model is higher than previous deployed model then push model into working.
7. Frontend 
    - Flask framework used to create API for the system. Frontend is developed in HTML.

8. CI/CD Pipeline and Deployment over Cloud.
    - Heroku platform used to deploy the entire project on cloud with docker and CI/CD is implemented by using github actions.


## Technologies Used
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/800px-Python-logo-notext.svg.png" alt="Python" width="200"/>
<img src="https://static.tildacdn.com/tild3536-6337-4235-a664-373965303839/evidently_ai_logo_fi.png" alt="evidently" width="350"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png" alt="sklearn" width="350"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/2560px-Pandas_logo.svg.png" alt="pandas" width="350"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/2560px-NumPy_logo_2020.svg.png" alt="Numpy" width="350"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Heroku_logo.svg/2560px-Heroku_logo.svg.png" alt="Heroku" width="350"/>
<img src="https://www.docker.com/wp-content/uploads/2022/03/vertical-logo-monochromatic.png" alt="docker" width="350"/>
<img src="https://www.parasoft.com/wp-content/uploads/2021/04/CICD_CICD.png" alt="CICD" width="350"/>





## Credits
- The data set is created by Brij Bhushan Anand.
- iNeuron Team who resolved my queries during the project.
- Krish Naik , Sudhanshu sir and Sunny sir for guiding and mentoring.
=======
# Item_sales_predictions
Full Stack End - End Machine Learning Project with CI/CD Piepline
Working Link : https://item-sales-predictions.herokuapp.com/ 
