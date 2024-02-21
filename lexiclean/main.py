import re
import string
import logging
from tqdm import tqdm
from IPython.display import clear_output
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)


class LexiClean:
    methods = {
        "clean_html_text": True,
        "remove_urls": True,
        "remove_punctuation": True,
        "remove_foreign_letters": True,
        "remove_numbers": True,
        "lowercasing": True,
        "remove_punctuation": True,
        "remove_white_spaces": True,
        "remove_repeated_characters": True,
        "tokenize": True,
        "remove_stopwords": True,
        "stemming": True,
        "handle_missing_values": True,
    }

    def __init__(self):
        print("Initializing TextCleaner")

    @staticmethod
    def handle_missing_values(df, columns):
        """
           Handle missing values in specified columns of a DataFrame by deleting rows with a number of missing values
           exceeding the given threshold.

           Args:
               df (DataFrame): The DataFrame containing the data.
               columns (list): List of column names in which missing values are to be handled.

           Returns:
               DataFrame: The DataFrame with missing values handled.
        """
        df_copy = df.copy()

        missing_values_count = df_copy[columns].isnull().sum()
        print("Number of missing values in each specified column:")
        print(missing_values_count)

        df_copy.dropna(subset=columns, inplace=True)

        print("Drop operation successful.")

        return df_copy

    @staticmethod
    def clean_html_text(data, columns=None):
        """
        Clean HTML tags and special characters from the text or DataFrame column.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text column to be cleaned.
            columns (list, optional): The name of the text column in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The cleaned text or DataFrame with the cleaned text column.
        """

        if isinstance(data, str):

            clean_text = re.sub('<[^<]+?>', '', data)

            return clean_text
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: re.sub('<[^<]+?>', '', x))

            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_urls(data, columns=None):
        """
            Remove URLs from the text or DataFrame columns.

            Args:
                data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
                columns (list, optional): List of column names in the DataFrame. Defaults to None.

            Returns:
                str or DataFrame: The text with URLs removed or DataFrame with the specified columns cleaned.
            """
        if isinstance(data, str):

            clean_text = re.sub(r'http\S+', '', data)
            return clean_text
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: re.sub(r'http\S+', '', x))
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_punctuation(data, columns=None):
        """
            Remove punctuation from the text or DataFrame columns.

            Args:
                data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
                columns (list, optional): List of column names in the DataFrame. Defaults to None.

            Returns:
                str or DataFrame: The text with punctuation removed or DataFrame with the specified columns cleaned.
        """

        custom_punctuation = "!\"#$%&'()*+,-.:;?@[\\]^_`{|}~"

        if isinstance(data, str):

            cleaned_text = ''.join([char for char in data if char not in custom_punctuation])
            return cleaned_text
        elif isinstance(data, pd.DataFrame) and columns:
            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: ''.join([char for char in x if char not in custom_punctuation]))
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_emojis(data, columns=None):
        """
        Remove emojis from the text or DataFrame columns.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The text with emojis removed or DataFrame with the specified columns cleaned.
        """
        if isinstance(data, str):
            clean_text = ''.join(char for char in data if ord(char) < 128)
            return clean_text
        elif isinstance(data, pd.DataFrame) and columns:
            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: ''.join(char for char in x if ord(char) < 128))
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_foreign_letters(data, columns=None):
        """
        Remove foreign letters from the text or DataFrame columns.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The text with foreign letters removed or DataFrame with the specified columns cleaned.
        """
        if isinstance(data, str):
            clean_text = re.sub(r'[^\x00-\x7F]+', '', data)
            return clean_text
        elif isinstance(data, pd.DataFrame) and columns:
            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_numbers(data, columns=None):
        """
        Remove numbers from the text or DataFrame columns.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The text with numbers removed or DataFrame with the specified columns cleaned.
        """
        if isinstance(data, str):

            clean_text = re.sub(r'\d+', '', data)
            return clean_text
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: re.sub(r'\d+', '', x))
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def lowercasing(data, columns=None):
        """
        Convert text to lowercase in the text or DataFrame columns.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The text converted to lowercase or DataFrame with the specified columns converted.
        """
        if isinstance(data, str):

            return data.lower()
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].str.lower()
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_white_spaces(data, columns=None):
        """
        Remove extra white spaces from the text or DataFrame columns.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The text with extra white spaces removed or DataFrame with the specified columns cleaned.
        """
        if isinstance(data, str):

            clean_text = re.sub('\\s+', ' ', data).strip()
            return clean_text
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: re.sub('\\s+', ' ', x).strip())
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_repeated_characters(data, columns=None):
        """
        Remove repeated characters in words from the text or DataFrame columns.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The text with repeated characters in words removed or DataFrame with the specified columns cleaned.
        """
        if isinstance(data, str):

            clean_text = re.sub(r'(.)\1+', r'\1', data)
            return clean_text
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: re.sub(r'(.)\1+', r'\1', x))
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def tokenize(data, columns=None):
        """
        Tokenize the text or DataFrame columns using NLTK's word_tokenize function.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be tokenized.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            list or DataFrame: The list of tokens or DataFrame with the specified columns tokenized.
        """
        if isinstance(data, str):

            tokens = word_tokenize(data)
            return tokens
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda x: word_tokenize(x))
            return df
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

    @staticmethod
    def remove_stopwords(data, columns=None):
        """
        Remove stopwords from the list of tokens or DataFrame columns.

        Args:
            data (list or DataFrame): The list of tokens or DataFrame containing the token columns to be filtered.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            list or DataFrame: The filtered list of tokens or DataFrame with the specified columns filtered.
        """
        stop_words = set(stopwords.words('english'))

        if isinstance(data, list):

            filtered_tokens = [token for token in data if token.lower() not in stop_words]
            return filtered_tokens
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(
                    lambda tokens: [token for token in tokens if token.lower() not in stop_words])
            return df
        else:
            raise ValueError("Invalid input. Please provide either a list of tokens or a DataFrame with column names.")

    @staticmethod
    def stemming(data, columns=None):
        """
        Perform stemming on the list of tokens or DataFrame columns containing tokens.

        Args:
            data (list or DataFrame): The list of tokens or DataFrame containing the token columns to be stemmed.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            list or DataFrame: The list of stemmed tokens or DataFrame with the specified columns stemmed.
        """
        stemmer = PorterStemmer()

        if isinstance(data, list):

            stemmed_tokens = [stemmer.stem(token) for token in data]
            return stemmed_tokens
        elif isinstance(data, pd.DataFrame) and columns:

            df = data.copy()
            for column in columns:
                df[column] = df[column].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
            return df
        else:
            raise ValueError("Invalid input. Please provide either a list of tokens or a DataFrame with column names.")

    @classmethod
    def preprocess_text(cls, data, columns=None, params=None, join=False):
        """
        Preprocess text data, handling both DataFrame and single text inputs.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be preprocessed.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.
            params (dict, optional): Parameters controlling text preprocessing steps. Defaults to None.
            join (bool): Whether to join tokens into a single string after preprocessing. Defaults to False.

        Returns:
            str or DataFrame: The preprocessed text or DataFrame.
        """

        if params is None:
            params = cls.methods.copy()  # Create a copy of cls.methods
        else:
            params_user = params
            params = cls.methods.copy()  # Create a copy of cls.methods
            for key, value in params_user.items():
                if key in params:
                    params[key] = value

        if isinstance(data, str):
            # If data is a single text
            preprocessed_data = cls._preprocess_single_text(data, params, join)
        elif isinstance(data, pd.DataFrame):
            # If data is a DataFrame
            preprocessed_data = cls._preprocess_dataframe(data, columns, params, join)
        else:
            raise ValueError("Invalid input. Please provide either a single text or a DataFrame.")

        return preprocessed_data

    @classmethod
    def _preprocess_single_text(cls, text, params, join=False):
        """
         Preprocess a single text.

         Args:
            text (str): The text to be preprocessed.
            params (dict): Parameters controlling text preprocessing steps.
            join (bool): Whether to join tokens into a single string after preprocessing. Defaults to False.

         Returns:
            str or list: The preprocessed text.
        """
        cleaned_text = None
        preprocessed_text = text

        if params.get("clean_html_text"):
            preprocessed_text = cls.clean_html_text(preprocessed_text)
        if params.get("remove_urls"):
            preprocessed_text = cls.remove_urls(preprocessed_text)
        if params.get("remove_punctuation"):
            preprocessed_text = cls.remove_punctuation(preprocessed_text)
        if params.get("remove_emojis"):
            preprocessed_text = cls.remove_emojis(preprocessed_text)
        if params.get("remove_foreign_letters"):
            preprocessed_text = cls.remove_foreign_letters(preprocessed_text)
        if params.get("remove_numbers"):
            preprocessed_text = cls.remove_numbers(preprocessed_text)
        if params.get("lowercasing"):
            preprocessed_text = cls.lowercasing(preprocessed_text)
        if params.get("remove_white_spaces"):
            preprocessed_text = cls.remove_white_spaces(preprocessed_text)
        if params.get("remove_repeated_characters"):
            preprocessed_text = cls.remove_repeated_characters(preprocessed_text)
        if params.get("tokenize"):
            tokens = cls.tokenize(preprocessed_text)
            if params.get("remove_stopwords"):
                tokens = cls.remove_stopwords(tokens)
            if params.get("stemming"):
                tokens = cls.stemming(tokens)
            if join:
                cleaned_text = " ".join(tokens)
            else:
                cleaned_text = tokens
        else:
            print("Attention to remove_stopwords and stemming make sure tokenize = true.")
            cleaned_text = preprocessed_text

        print("Operation successful.")
        return cleaned_text

    @classmethod
    def _preprocess_dataframe(cls, df, columns, params, join=False):
        """
        Preprocess DataFrame columns containing text.

        Args:
            df (DataFrame): The DataFrame containing text columns to be preprocessed.
            columns (list): List of column names in the DataFrame.
            params (dict): Parameters controlling text preprocessing steps.
            join (bool): Whether to join tokens into a single string after preprocessing. Defaults to False.

        Returns:
            DataFrame: The DataFrame with preprocessed text columns.
        """
        df_copy = df.copy()
        df_copy = cls.handle_missing_values(df_copy, columns)
        # Calculate total number of rows to preprocess
        total_rows = sum(df[col].count() for col in columns)

        # Initialize tqdm to show progress bar
        pbar = tqdm(total=total_rows)
        for col in columns:
            for index, row in df_copy.iterrows():
                # Apply text preprocessing to each row in the specified column
                df_copy.at[index, col] = cls._preprocess_single_text(row[col], params, join)

                clear_output(wait=True)
                # Update progress bar for each row processed
                pbar.update(1)

        # Close progress bar
        pbar.close()
        print("Operation successful.")
        return df_copy