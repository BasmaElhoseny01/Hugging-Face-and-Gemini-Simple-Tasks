# Text Processing with BERT, Translation, and Gemini API

![image](https://github.com/user-attachments/assets/46a3a9ea-b484-482c-bfed-b81a48945c32)

This notebook demonstrates three key natural language processing (NLP) tasks: sentiment analysis using BERT, English-to-Arabic translation, and text generation with the Gemini API (`gemini-1.5-flash`).

## Table of Contents
1. [BERT for Sentiment Analysis](#1-bert-for-sentiment-analysis)
2. [English-to-Arabic Translation](#2-english-to-arabic-translation)
3. [Text Generation with Gemini API](#3-text-generation-with-gemini-api)
4. [Requirements](#requirements)
5. [Conclusion](#conclusion)

## 1. BERT for Sentiment Analysis

In this section, we use the pre-trained `DistilBERT` model fine-tuned on the SST-2 dataset to perform sentiment analysis. Sentiment analysis detects whether the sentiment of a text is positive or negative.

### Model:
- **DistilBERT (Fine-tuned SST-2)**: [DistilBERT Base Uncased (SST-2)](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

### Example:

```python
from transformers import pipeline

# Load pre-trained DistilBERT sentiment analysis model
classifier = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Input text for sentiment analysis
text = "I am so excited"

# Perform sentiment analysis
result = classifier(text)
print(result)
```



## 2. English-to-Arabic Translation

In this section, we use the `marefa-mt-en-ar` model to translate English text into Arabic. This model supports machine translation for Arabic language tasks.

### Model:
- **Marefa MT (English to Arabic)**: [Marefa English-Arabic Translation Model](https://huggingface.co/marefa-nlp/marefa-mt-en-ar)

### Example:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load Marefa translation model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("marefa-nlp/marefa-mt-en-ar")
model = AutoModelForSeq2SeqLM.from_pretrained("marefa-nlp/marefa-mt-en-ar")

# Input text for translation
english_text = "Artificial intelligence is transforming industries worldwide."

# Tokenize the text
inputs = tokenizer(english_text, return_tensors="pt", padding=True)

# Generate the translation
outputs = model.generate(inputs["input_ids"], max_length=50)
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the translated Arabic text
print(translated_text)
```


## 3. Text Generation with Gemini API

In this section, we use the `gemini-1.5-flash` model from the `genai` library to generate content based on a text prompt.

### Setup:

Ensure that the `google-generativeai` package is installed and the `api_key` is configured.

```bash
pip install google-generativeai
```


### Example:

```python
import google.generativeai as genai

# Configure your API key for the Gemini model
genai.configure(api_key="your_api_key")

# Define the prompt for text generation
prompt = "The future of AI in healthcare includes"

# Generate text using the Gemini model
response = genai.generate_content(prompt)

# Extract and print the generated text
generated_text = response.result.candidates[0].content.parts[0].text
print(generated_text)

```


## Requirements

- Python 3.7+
- Hugging Face Transformers
- Google Generative AI (`google-generativeai` package)


## Conclusion

This notebook walks through three essential NLP tasks:

- **Sentiment Analysis**: Using BERT for determining sentiment polarity.
- **Translation**: Converting English text to Arabic with Marefa MT.
- **Text Generation**: Generating creative text content using the Gemini model.
