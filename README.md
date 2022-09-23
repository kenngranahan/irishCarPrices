# irishCarPrices
Aggregate second hand car prices in the Irish Market

## This project has been designed to do three things:
1. Scrape/clean data from https://www.carsireland.ie/. Data points gathered include the car price, make, model, year, etc.
   This data is then stored on a postgreSQL database installed on the same machine running the scraper.
2. Train and test and ML model to predict a car's price using the data scraped. This repository contains Notebooks detailing the EDA and
   model selection done supporting the final model. The data scraped by step 1 is written to a csv call 
   cleandata.csv, so it is unnessecary carry out step 1 to get the data
3. Wrap the final trained model in an API and GUI. The API can be improted by installing this repository and importing the API from ModelAPI.
   It should be possible to run the GUI also, but user inputs are not checked by the GUI
   

ONLY STEP 2 AND 3 WERE DESGINED WITH PORTABLITY IN MIND. RUNNING THE SCRAPPER FROM THIS REPOSITORY WILL REQUIRE INSTALLING POSTGRESQL ON THE SAME MACHINE

### Step 1: Database and Scraper set up
#### Setting up the Database for the scraper
This requires a postgreSQL database called irishCarPrices with the following tables: 
- carsIrelandScrappedData
- carsIrelandCleanData
- carsIrelandScraperMetaData  
The expected schema for these tables can be found in dataSourcing/configs/databaseConfig.yaml.  
The fomat of the table config in this file is [column_name, column_type] i.e. [year, numeric] means the table should contain a column named 'year' with numeric type.  
The database config can be updated in the dataSourcing/configs/databaseConfig.yaml file. This is where you can update your username, password, host and port.  
The repository also contains config files for the scraper and parser. It's recommended you leave these alone unless you have a granular understanding of the code  

#### How the scrapper works:
The scrapper is designed to be run on a weekly basis. It will check all valid page IDs from the previous week's download. It also estimates how many
new IDs have been added and downloads those too.  
The script parses the HTML source code and writes the data to the database. It cleans the data after all the IDs have been downloaded.  


### Step 2: Model Architecture
#### Overview of the Data
- Based on the EDA shared in this repository, the target variable of the model is Y=log(price).  
- Outliers and missing value are taken care of as detailed in the EDA Notebook. No other changes are made to the data.
- The age and mileage have a linear relationship with Y when sampled for a particular car model e.g. Ford Focus. 
The coeffients, i.e. the slope and intercept are assumed to capture the intial price and depreciation of the car model.
This is important to note as it influences the design decision of the ML model

#### Overview of the ML model
- Encoding of categorical features are done using sklearn's OneHotEncoder
- The age and mileage are normalized and ran through a PCA then fitted to Y using Lasso for each car model. 
The coefficients from this regresion are then used as features for an ensenble regressor.
So for each car model in the dataset (Ford Focus, Nissan Micra, etc.) we have a pair of features (X0, X1) which are engineer by a liner regression.
The features are intened to capture the initial price and depreciation of a car model over time.
- The ensenble regressor used is GradientBoostingRegressor from sklearn. Details of model seletion and fine tuning are found in the Notebook in this
repository


### Step 3: Model API and GUI
#### API
- The API is designed to load a pretrained model stored in this repository
- The API takes a pandas DataFrame as input. The DataFrame is expected to contain columns named after the feature used for training
  and returns predicted car prices as a pandas Series with the same index as the input DataFrme. The list of expected columns can be found using the getExpectedColumns() method
  if the API instance
 - Missing values are allowed for all feature except the make and model of the car, if these features are empty then that sample is ignored and a warning is thrown

#### GUI
- The GUI is an incredibiliy basic interface implemented with Tkinter. User inputs are not checked by the GUI, so bad inputs fed by the user will likely cause issues





