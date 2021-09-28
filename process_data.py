import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    loading function for the input csv input files. It also merges the dataframes on their 'id'.
    :param messages_filepath: csv file including the messages
    :param categories_filepath: csv file including the categories for the messages
    :return: df_combined: combined data frame of the messages and categories files
    """
    # import to data frames
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    # merge data frames on 'id'
    df_combined = df_messages.merge(df_categories, on='id')

    return df_combined


def clean_data(df):
    """
    Cleaning the data: expanding the 'categories' to individual columns and removing duplicates.
    :param df: merged data frame with categories in a single column
    :return: df: df with categies expanded over multiple columns
    """
    # take categories and split over multiple columns
    categories_df = df['categories'].str.split(';', expand=True)

    # get first row and extract names of all categories
    row = categories_df.iloc[0]
    category_col_names = row.str.split('-', expand=True)[0]

    # rename the columns with the names of the categories
    categories_df.columns = category_col_names

    # remove the category names from the data frame entries and convert the remaining entries to integer
    for column in categories_df:
        categories_df[column] = categories_df[column].astype(str).str.split('-', expand=True)[1]
        categories_df[column] = categories_df[column].astype(int)

    # replace 'categories' column in df with the new category data frame
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories_df], axis=1)

    # remove duplicates
    if (len(df) - len(df.drop_duplicates())) > 0:
        df = df.drop_duplicates()
    if (len(df) - len(df.drop_duplicates())) == 0:
        return df
    else:
        return "Data frame has remaining duplicates."


def save_data(df, database_filename):
    """
    Saving the merged and cleaned data frame into a sqlite database
    :param df: merged and cleaned data frame
    :param database_filename: name of the sqlite database
    :return: none
    """
    engine = create_engine('sqlite:///database_filename')
    df.to_sql('data_frame', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()