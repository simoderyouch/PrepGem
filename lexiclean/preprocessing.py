import re
import nltk
import pandas as pd
from IPython.display import clear_output
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from tqdm import tqdm
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)
from nltk.corpus import brown
from .utils import preprocess_columns
english_words = set(w.lower() for w in brown.words())
stop_words = set(stopwords.words('english'))
from nltk.tokenize import TreebankWordTokenizer




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


def remove_nonsense_words(data, columns=None):
    """
       Remove nonsense words from either a single text or DataFrame columns.

       Args:
           data (str or DataFrame): The text or DataFrame containing text columns to be processed.
           columns (list): List of column names in the DataFrame. Required if 'data' is a DataFrame.

       Returns:
           str or DataFrame: The processed text or DataFrame with columns containing removed nonsense words.

       Raises:
           ValueError: If the input is not a single text or a DataFrame with column names provided.
       """

    def preprocessing_function(x):
        return ' '.join(
            [word for word in nltk.wordpunct_tokenize(x.lower()) if word in english_words or not word.isalpha()])
    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_nonsense_words')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


def spell_corrector(data, columns=None):
    """
    Perform spell-checking on either a single text or a DataFrame with text columns.

    Args:
        data (str or DataFrame): The text or DataFrame containing text columns to be spell-checked.
        columns (list): List of column names in which missing values are to be handled.
    Returns:
        str or DataFrame: The text or DataFrame with spell-checked text columns.
    """
    def preprocessing_function(x):
        words_list = nltk.word_tokenize(x.lower())
        corrected_words = [str(TextBlob(word).correct()) for word in words_list]
        return ' '.join(corrected_words)

    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:
        return preprocess_columns(data, columns, preprocessing_function, 'spell_corrector')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


def nosense_words_and_spell_check(data, columns=None):

    """
Perform spell-checking on either a single text or a DataFrame with text columns,
and remove nonsense words.

Args:
    data (str or DataFrame): The text or DataFrame containing text columns to be spell-checked
                             and cleaned from nonsense words.
    columns (list): List of column names in which missing values are to be handled.

Returns:
    str or DataFrame: The text or DataFrame with spell-checked text columns and removed nonsense words.
"""
    def preprocessing_function(x):
        return remove_nonsense_words(spell_corrector(x))
    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'nosense_words_and_spell_check')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


def clean_html_text(data, columns=None):
    """
    Clean HTML tags and special characters from the text or DataFrame column.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text column to be cleaned.
        columns (list, optional): The name of the text column in the DataFrame. Defaults to None.

    Returns:
        str or DataFrame: The cleaned text or DataFrame with the cleaned text column.
    """
    def preprocessing_function(x):
        return re.sub('<[^<]+?>', '', x)
    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'clean_html_text')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")

def remove_urls(data, columns=None):
    """
        Remove URLs from the text or DataFrame columns.

        Args:
            data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
            columns (list, optional): List of column names in the DataFrame. Defaults to None.

        Returns:
            str or DataFrame: The text with URLs removed or DataFrame with the specified columns cleaned.
        """
    def preprocessing_function(x):
        return re.sub(r'\b(?:https?://)?(?:www\.)?(?:\S+\.)+(?:com|ma|org|net|edu|gov|info)\b', '', x)
    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_urls')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


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
    def preprocessing_function(x):
        return ''.join([char for char in x if char not in custom_punctuation])
    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_punctuation')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")



def remove_emojis(data, columns=None):
    """
    Remove emojis from the text or DataFrame columns.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        str or DataFrame: The text with emojis removed or DataFrame with the specified columns cleaned.
    """
    def preprocessing_function(x):
        return ''.join(char for char in x if ord(char) < 128)

    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_emojis')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")



def remove_foreign_letters(data, columns=None):
    """
    Remove foreign letters from the text or DataFrame columns.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        str or DataFrame: The text with foreign letters removed or DataFrame with the specified columns cleaned.
    """
    def preprocessing_function(x):
        return re.sub(r'[^\x00-\x7F]+', '', x)

    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_foreign_letters')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


def remove_numbers(data, columns=None):
    """
    Remove numbers from the text or DataFrame columns.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        str or DataFrame: The text with numbers removed or DataFrame with the specified columns cleaned.
    """
    def preprocessing_function(x):
        return re.sub(r'\d+', '', x)

    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_numbers')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


def lowercasing(data, columns=None):
    """
    Convert text to lowercase in the text or DataFrame columns.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        str or DataFrame: The text converted to lowercase or DataFrame with the specified columns converted.
    """
    def preprocessing_function(x):
        return x.lower()
    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'lowercasing')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


def remove_white_spaces(data, columns=None):
    """
    Remove extra white spaces from the text or DataFrame columns.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        str or DataFrame: The text with extra white spaces removed or DataFrame with the specified columns cleaned.
    """
    def preprocessing_function(x):
        return re.sub('\\s+', ' ', x).strip()
    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_white_spaces')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")




def remove_repeated_characters(data, columns=None):
    """
    Remove repeated characters in words from the text or DataFrame columns.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be cleaned.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        str or DataFrame: The text with repeated characters in words removed or DataFrame with the specified columns cleaned.
    """
    def preprocessing_function(x):
        return re.sub(r'(.)\1+', r'\1', x)

    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'remove_repeated_characters')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")



def tokenize(data, columns=None):
    """
    Tokenize the text or DataFrame columns using NLTK's word_tokenize function.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be tokenized.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        list or DataFrame: The list of tokens or DataFrame with the specified columns tokenized.
    """
    tokenizer = TreebankWordTokenizer()

    def preprocessing_function(text):
        return tokenizer.tokenize(text)

    if isinstance(data, str):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:

        return preprocess_columns(data, columns, preprocessing_function, 'tokenize')
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame with column names.")


def remove_stopwords(data, columns=None):
    """
    Remove stopwords from the list of tokens or DataFrame columns.

    Args:
        data (list or DataFrame): The list of tokens or DataFrame containing the token columns to be filtered.
        columns (list, optional): List of column names in the DataFrame. Defaults to None.

    Returns:
        list or DataFrame: The filtered list of tokens or DataFrame with the specified columns filtered.
    """

    def preprocessing_function(x):
        return [token for token in x if token.lower() not in stop_words]

    if isinstance(data, list):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:
        if data[columns].dtypes == 'object':
            raise  ValueError("Invalid input. Please provide DataFrame with column of  a list of tokens .")
        else:
            return preprocess_columns(data, columns, preprocessing_function, 'remove_stopwords')
    else:
        raise ValueError("Invalid input. Please provide either a list of tokens or a DataFrame with column names.")


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
    def preprocessing_function(x):
        return [stemmer.stem(token) for token in x]

    if isinstance(data, list):
        return preprocessing_function(data)
    elif isinstance(data, pd.DataFrame) and columns:
        if data[columns].dtypes == 'object':
            raise ValueError("Invalid input. Please provide DataFrame with column of  a list of tokens .")
        else:
            return preprocess_columns(data, columns, preprocessing_function, 'stemming')
    else:
        raise ValueError("Invalid input. Please provide either a list of tokens or a DataFrame with column names.")

