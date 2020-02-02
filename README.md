# Disaster-Response-Piplines

## Prerequisites
- **Python versions 3.0+**
- **nltk 3.4 :** need to download 3 necesasry packages: *punkt* , *wordnet* and *stopwords*
- **pandas 0.23.4**
- **numpy 1.15.4**
- **skcikit-learn 0.20.1**
- **sqlalchemy 1.2.15**
- **pickle 0.7.5**

## Project Motivation:
The goal of this project is to apply my data engineering and software skills by analyzing disaster dataset from [Figure Eight](https://www.figure-eight.com/). It is a web app where an emergency worker can input a new message about the event and get classification results in several categories so that the message can be sent to an ppropriate disaster relief agency.
 ## About the Data:
[Figure Eight](https://www.figure-eight.com/) is used for this project. It contains real messages that were sent during disaster events which will be used to develop a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.
 
 ## Files Descriptions:
 ### -1 **app:** 
 - **templates:** html files for the web pages . 
 - **run.py:** to run the flask app. 
 
 ### -2 **data:**
 - **DisasterResponse.db:** the database the contain data table.
 - **disaster_categories.csv:** csv file contains the catogry names. 
 - **disaster_messages.csv:** csv file contains the messages with extra information. 
 - **process_data.py:** ETL python script to preprocess the data.
 
 ### -3 **models:**
 - **train_classifier.py:** python script to build ml pipline and save the final model. 
  - **classifier.pkl:** the saved model.
 
 ## Instructions:
1- Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2- Run the following command in the app's directory to run your web app.
    `python run.py`

3- Go to http://0.0.0.0:3001/

-----------------------------------------------------------------------------------------------------------------
 *This project is a part of Udacity's Data Science Nanodegree*
 

