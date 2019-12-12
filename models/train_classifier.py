import sys
import pandas as pd 
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import re 
import pickle
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    This function 
    takes:
    database path
    
    returns: 
    X: predictors 
    Y: 36 catogries 
    and list of catogry names 
 
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    df = df.dropna(how='any',axis=0) 
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    
    return X , Y , category_names

def tokenize(text):
    """
    This function 
    takes: text 
    
    returns: cleaned list of words after removing the 
    puncutations and stopwords and lemmetize the text

    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return clean_tokens


def build_model():
    
    """
    This function will create and return a ML pipline  
    
    """
    pipeline = pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(KNeighborsClassifier(n_neighbors = 2)))
    ])
    parameters = {
        'clf__estimator__n_neighbors': [5,10,15]
    }
    cv_kn= GridSearchCV(pipeline, param_grid=parameters)
    
    return cv_kn


def evaluate_model(model, X_test, Y_test, category_names):
    
    """ 
    This function will calculate  precision , recall , accuracy and f1 scores for each category
    
    """
    
    kn_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in kn_pred]),   target_names=category_names))
    print('Accuracy: ')
    print(accuracy_score(Y_test.iloc[:,1:].values, np.array([x[1:] for x in kn_pred])))


def save_model(model, model_filepath):
    """
    This function is used to save the model 
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()