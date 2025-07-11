# Customer Feedback Sentiment Analyzer

This project is a customer sentiment analyzer i.e. it predicts the if the review is postive or negative.

## Features

- **Sentiment Prediction:** Real-time classification of submitted text.
- **Data Preprocessing:** Text cleaning and vectorization.
- **Machine Learning Model:** Utilizes a trained text classification model.

## Setup and Running the Application

- **Install Dependencies:**

  ```bash
  pip install -r requirements.txt
  ```

  (You'll need to create a `requirements.txt` file with `Flask`, `scikit-learn`, `pandas`, `nltk`, `joblib` etc.)

- **Download NLTK Data (if using):**
  Run Python and execute:

  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('punkt') # If using word tokenizers
  nltk.download('wordnet') # If using lemmatization
  ```

-. **Data Preparation & Model Training:**

    - **Download the Dataset:** Obtain the `imdb_reviews.csv` (or your chosen dataset) and place it in the `data/` directory.
    - **Train the Model:** You'll need to run a separate script.
        ```python
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer


        from src.text_processor import TextProcessor
        from src.sentiment_model import SentimentModel # Assuming it handles model saving

        # Load raw data
        df = pd.read_csv('data/imdb_reviews.csv') # Adjust to your actual dataset and path

        # Initialize processor
        processor = TextProcessor()
        df['processed_text'] = df['review'].apply(processor.clean_text) # Assuming 'review' is your text column

        # Prepare data for model
        y = df['sentiment'] # Assuming 'sentiment' is your label column (e.g., 'positive', 'negative')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train sentiment model
        sentiment_model_obj = SentimentModel()
        sentiment_model_obj.train(X_train, y_train)

## Data Source & Citation

The machine learning model for sentiment analysis in this project was initially trained using the following public dataset:

**For IMDb Movie Reviews Dataset (Kaggle):**

- Citation file --> references.bib

## Future Enhancements

- Add more sophisticated NLP techniques (e.g., Word Embeddings, Transformers).
- Implement a database (e.g., SQLite) to persistently store submitted feedback.
- Use the Flask UI/UX for Front-end.
- Add user authentication.
- Allow model retraining through the interface.
