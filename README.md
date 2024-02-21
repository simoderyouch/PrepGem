# LexiClean

LexiClean is a Python package for preprocessing text data, designed to simplify the text-cleaning process for natural language processing (NLP) projects.

## Features

- **Handle Missing Values**: Easily handle missing values in specified DataFrame columns.
- **Clean HTML Text**: Remove HTML tags and special characters from text or DataFrame columns.
- **Remove URLs**: Remove URLs from text or DataFrame columns.
- **Remove Punctuation**: Remove punctuation from text or DataFrame columns.
- **Remove Emojis**: Remove emojis from text or DataFrame columns.
- **Remove Foreign Letters**: Remove foreign letters from text or DataFrame columns.
- **Remove Numbers**: Remove numbers from text or DataFrame columns.
- **Lowercasing**: Convert text to lowercase in text or DataFrame columns.
- **Remove White Spaces**: Remove extra white spaces from text or DataFrame columns.
- **Remove Repeated Characters**: Remove repeated characters in words from text or DataFrame columns.
- **Tokenize**: Tokenize text using NLTK's word_tokenize function.
- **Remove Stopwords**: Remove stopwords from text tokens.
- **Stemming**: Perform stemming on text tokens.

## Installation

You can install LexiClean via pip:

```bash
pip install lexiclean 
```


## Usage

#### Importing the module python
```bash
from lexiclean import LexiClean
```
#### Initializing the LexiClean object
```bash
cleaner = LexiClean()
```
#### Preprocessing a single text
```bash
text = "This is an example text for preprocessing."
cleaned_text = cleaner.preprocess_text(text)
print(cleaned_text)
```
```bash
text = "This is an example text for preprocessing."
cleaned_text = cleaner._preprocess_single_text(text)
print(cleaned_text)
```
#### Preprocessing a DataFrame

```bash
import pandas as pd

# Create a sample DataFrame
data = {
    'text_column': ["This is an example text.", "Another example text with numbers: 12345."]
}
df = pd.DataFrame(data)

# Preprocess text column in the DataFrame
cleaned_df = cleaner.preprocess_text(df, columns=['text_column'])
print(cleaned_df)
```
```bash
import pandas as pd

# Create a sample DataFrame
data = {
    'text_column': ["This is an example text.", "Another example text with numbers: 12345."]
}
df = pd.DataFrame(data)

# Preprocess text column in the DataFrame
cleaned_df = cleaner._preprocess_dataframe(df, columns=['text_column'])
print(cleaned_df)
```
### Customizing preprocessing steps
You can customize the preprocessing steps by passing a dictionary of parameters to the preprocess_text method. Available parameters include:

* clean_html_text.
* remove_urls
* remove_punctuation
* remove_emojis
* remove_foreign_letters
* remove_numbers
* lowercasing
* remove_white_spaces
* remove_repeated_characters
* tokenize
* remove_stopwords
* stemming
* handle_missing_values

# Example usage
text = "This is an example text with <html> tags and URLs: https://example.com."
cleaned_text = cleaner.preprocess_text(text, params={"clean_html_text": True, "remove_urls": True, "remove_punctuation": True})
print(cleaned_text)
```
