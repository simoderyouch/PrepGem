# LexiClean

LexiClean is a Python package for preprocessing text data, designed to simplify and streamline the text cleaning process for natural language processing (NLP) projects.

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
```bash
from lexiclean import LexiClean

# Initialize TextCleaner
cleaner = LexiClean()

# Example usage
text = "This is an example text with <html> tags and URLs: https://example.com."
cleaned_text = cleaner.preprocess_text(text, params={"clean_html_text": True, "remove_urls": True, "remove_punctuation": True})
print(cleaned_text)
```
