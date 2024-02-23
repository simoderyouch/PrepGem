# LexiGem

LexiGem is a Python package for preprocessing text data, designed to simplify the text-cleaning process for natural language processing (NLP) projects.

## Features
LexiClean offers the following features:

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
- **Remove Nonsense Words**: Remove nonsense words from text or DataFrame columns.
- **Spell Correction**: Perform spell-checking on text or DataFrame columns.
- **Nonsense Words and Spell Check**: Perform spell-checking and remove nonsense words from text or DataFrame columns.
- **Tokenize**: Tokenize text using NLTK's word_tokenize function.
- **Remove Stopwords**: Remove stopwords from text tokens.
- **Stemming**: Perform stemming on text tokens.

## Installation

You can install LexiGem via pip:

```bash
pip install LexiGem 
```


## Usage

#### Importing the module python
```bash
import lexigem 
```

#### Basic Usage

```bash
text = "This is an example text for preprocessing."
cleaned_text = lexigem.preprocess_text(text)
print(cleaned_text)
```

#### Preprocessing a single text
```bash
text = "This is an example text for preprocessing."
cleaned_text = lexigem.preprocess_single_text(text)
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
cleaned_df = lexigem.preprocess_dataframe(df, columns=['text_column'])
print(cleaned_df)
```
### Default preprocessing pipeline
 Default available preprocessing step is:

* clean_html_text.
* remove_urls
* remove_punctuation
* remove_emojis
* remove_foreign_letters
* remove_numbers
* lowercasing
* remove_white_spaces
* remove_repeated_characters
* nosense_words_and_spell_check
* tokenize
* remove_stopwords
* stemming


```bash
text = "This is an example text with <html> tags and URLs: https://example.com."
cleaned_text = lexigem.preprocess_text(text)
print(cleaned_text) 
```

### Custom preprocessing pipeline
You can customize the preprocessing steps by passing a list of parameters  to the preprocess_text method. Available parameters include:

* clean_html_text.
* remove_urls
* remove_punctuation
* remove_emojis
* remove_foreign_letters
* remove_numbers
* lowercasing
* remove_white_spaces
* remove_repeated_characters
* remove_nonsense_words
* spell_corrector
* nosense_words_and_spell_check
* tokenize
* remove_stopwords
* stemming
* handle_missing_values


##### Example usage
```bash
text = "This is an example text with <html> tags and URLs: https://example.com."
cleaned_text = lexigem.preprocess_text(text, pipeline=["clean_html_text","nosense_words_and_spell_check"])
print(cleaned_text)

```
You can customize the preprocessing steps by passing a  parameter remove with value of True remove=True to the preprocess_text method to remove a step. Available parameters include:

##### Example usage
```bash
text = "This is an example text with <html> tags and URLs: https://example.com."
cleaned_text = lexigem.preprocess_text(text, pipeline=["clean_html_text"], remove=True)
print(cleaned_text)
```

You can use all step as normal function just by passing The text or DataFrame containing the text column to be cleaned
```bash
from LexiGem import remove_urls

# Example text with URLs
text_with_urls = "This is an example text with URLs: https://example.com and http://www.example.org."

# Remove URLs from the text

cleaned_text = remove_urls(text_with_urls)

print("Original text:")
print(text_with_urls)
print("\nText after removing URLs:")
print(cleaned_text)
```
This will output:
```bash
Original text:
This is an example text with URLs: https://example.com and http://www.example.org.

Text after removing URLs:
This is an example text with URLs:  and .
```