import pandas as pd
from IPython.display import clear_output
from tqdm import tqdm
from .utils import preprocess_cell
from .preprocessing import handle_missing_values, clean_html_text, remove_urls, remove_punctuation, remove_emojis, remove_foreign_letters, \
    remove_numbers, lowercasing, remove_white_spaces, remove_repeated_characters, spell_corrector, \
    remove_nonsense_words, nosense_words_and_spell_check, tokenize, remove_stopwords, stemming

preprocessing_pipeline = [
    "clean_html_text",
    "remove_urls",
    "remove_punctuation",
    "remove_emojis",
    "remove_foreign_letters",
    "remove_numbers",
    "lowercasing",
    "remove_punctuation",
    "remove_repeated_characters",
    "nosense_words_and_spell_check",
    "remove_white_spaces",
    "tokenize",
    "remove_stopwords",
    "stemming",
]

preprocessing_functions = {
    "clean_html_text": clean_html_text,
    "remove_urls": remove_urls,
    "remove_punctuation": remove_punctuation,
    "remove_emojis": remove_emojis,
    "remove_foreign_letters": remove_foreign_letters,
    "remove_numbers": remove_numbers,
    "lowercasing": lowercasing,
    "remove_white_spaces": remove_white_spaces,
    "remove_repeated_characters": remove_repeated_characters,
    "spell_corrector": spell_corrector,
    "remove_nonsense_words": remove_nonsense_words,
    "nosense_words_and_spell_check": nosense_words_and_spell_check,
    "tokenize": tokenize,
    "remove_stopwords": remove_stopwords,
    "stemming": stemming
}

def preprocess_single_text(text, pipeline=None,remove=False , join=False):
    """
     Preprocess a single text.
     Args:
        text (str): The text to be preprocessed.
        pipeline (list): Parameters controlling text preprocessing steps.
        remove (bool):  Make it True to define pipeline as Preprocessing steps to remove . Default False.
        join (bool): Whether to join tokens into a single string after preprocessing. Defaults to False.

     Returns:
        str or list: The preprocessed text.
    """
    if not isinstance(text, str):
        print(text)
        raise ValueError("Invalid input. Please provide either a single text or you provide null value")

    pipeline_to_use  = []
    if pipeline is None:
        pipeline_to_use = preprocessing_pipeline
    else:
       if remove:
            pipeline_to_use = preprocessing_pipeline.copy()
            for element_to_remove in pipeline:
                if element_to_remove in pipeline_to_use:
                    pipeline_to_use = [element for element in pipeline_to_use if element_to_remove != element]
       else:
           pipeline_to_use = pipeline



    preprocessed_text = text


    for step in pipeline_to_use:
        if step == "remove_stopwords" and ("tokenize" not in pipeline_to_use):
            continue
        elif step == "stemming" and ("tokenize" not in pipeline_to_use):
            continue
        elif step in preprocessing_functions:
            preprocessed_text = preprocessing_functions[step](preprocessed_text)
        else:
            print(f"Unknown parameter: {step}")



    if join and ("tokenize" in pipeline_to_use):
        cleaned_text = " ".join(preprocessed_text)
    else:
        cleaned_text = preprocessed_text



    return cleaned_text

def preprocess_dataframe(df, columns=None, pipeline=None,remove=False, join=False, missing_values=True):
    """
    Preprocess DataFrame columns containing text.

    Args:
        df (DataFrame): The DataFrame containing text columns to be preprocessed.
        columns (list): List of column names in the DataFrame.
        pipeline (list): Parameters controlling text preprocessing steps.
        remove (bool):  Make it True to define pipeline as Preprocessing steps to remove . Default False.
        join (bool): Whether to join tokens into a single string after preprocessing. Defaults to False.
        missing_values (bool): Whether to handle missing values  before preprocessing. Defaults to False.

    Returns:
        DataFrame: The DataFrame with preprocessed text columns.
    """


    if not isinstance(df, pd.DataFrame):
        raise ValueError("Invalid input. Please provide either  a DataFrame with column names.")
    if columns is None:
        raise ValueError("Invalid columns")
    df_copy = df.copy()
    if missing_values:
        df_copy = handle_missing_values(df_copy, columns)


    for col in columns:
        print("Processing column:", col)
        total_rows = sum(df[col].count() for col in columns)
        pbar = tqdm(total=total_rows, desc=f"Preprocessing '{col}'")
        df_copy[col] = df_copy[col].apply(lambda cell: preprocess_cell(preprocess_single_text(cell, pipeline=pipeline, remove=remove, join=join), pbar))
        pbar.close()


    print("Operation successful.")
    return df_copy

def preprocess_text(data, columns=None, pipeline=None,remove=False, join=False, missing_values=True):
    """
    Preprocess text data, handling both DataFrame and single text inputs.

    Args:
        data (str or DataFrame): The text or DataFrame containing the text columns to be preprocessed.
        columns (list): List of column names in the DataFrame.
        pipeline (list): Parameters controlling text preprocessing steps.
        remove (bool):  Make it True to define pipeline as Preprocessing steps to remove . Default False.
        join (bool): Whether to join tokens into a single string after preprocessing. Defaults to False.
        missing_values (bool): Whether to handle missing values  before preprocessing. Defaults to False.

    Returns:
        str or DataFrame: The preprocessed text or DataFrame.
    """
    if isinstance(data, str):
        # If data is a single text
        preprocessed_data = preprocess_single_text(data, pipeline=pipeline,remove=remove , join=join)
    elif isinstance(data, pd.DataFrame):
        # If data is a DataFrame
        preprocessed_data = preprocess_dataframe(data,columns=columns, pipeline=pipeline, remove=remove, join=join, missing_values=missing_values)
    else:
        raise ValueError("Invalid input. Please provide either a single text or a DataFrame.")

    return preprocessed_data

