import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function 
    takes:
    message dataset and catogries data set.
     
    returns:
    Merged dataset
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')

    return df 


def clean_data(df):
    
    """
    This function 
    takes:
    
    the merged dataset.
     
    returns:
    cleaned dataset with seperated columns for each cateogry 
    
    """
    #Split the values in the categories column on the ; character so that each value becomes a separate column
    categories = df['categories'].str.split(";" , expand=True)
    #Use the first row of categories dataframe to create column names for the categories data.
    row = categories.iloc[[0]]
    category_colnames = [str(row[cat]).split('\n')[0].split()[1][0:-2] for cat in row]
    #Rename columns of categories with new column names.
    categories.columns = category_colnames
    
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    #For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    for column in categories:
    # set each value to be the last character of the string
        categories[column]= categories[column].astype(str).str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Drop the categories column from the df dataframe since it is no longer needed.
    #Concatenate df and categories data frames.
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    # drop dupicates 
    df.drop_duplicates(keep=False,inplace=True) 
    
    return df




def save_data(df, database_filename):
    
    """
    This function 
    takes:
    dataset and save it with the provided database name 
    
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_messages', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
