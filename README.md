# 📚 LangChain Book Recommendation System

A sophisticated book recommendation system leveraging LangChain, vector embeddings, and machine learning techniques to provide personalized book recommendations based on description, sentiment, and category analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Machine Learning Components](#machine-learning-components)
- [Web Interface](#web-interface)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [License](#license)

---

## 🔍 Overview

This project creates an intelligent book recommendation system that goes beyond simple category-based recommendations. Using natural language processing and vector search techniques, the system understands book content through descriptions, categorizes books accurately, analyzes sentiment/emotion in book content, and delivers personalized recommendations through an interactive web interface.

---

## ✨ Features

- **Vector-Based Book Search**: Find similar books based on content embedding vectors.
- **Smart Categorization**: Accurate book classification using machine learning.
- **Sentiment Analysis**: Recommendations based on emotional content/tone.
- **Interactive Interface**: User-friendly Gradio web UI for exploring recommendations.
- **Multi-dimensional Analysis**: Combines rating data with content analysis for better recommendations.

---

## 📊 Dataset Information

The project utilizes a comprehensive book dataset with the following attributes:

| Feature             | Description                                 |
|---------------------|---------------------------------------------|
| isbn13/isbn10       | Book identification numbers                 |
| title/subtitle      | Book titles and subtitles                   |
| authors             | Book authors                                |
| categories          | Genre classifications                       |
| thumbnail           | Cover image URLs                            |
| description         | Book descriptions/synopses                  |
| published_year      | Year of publication                         |
| average_rating      | Reader ratings (scale: 1-5)                 |
| num_pages           | Book length                                 |
| ratings_count       | Number of ratings received                  |
| tagged_description  | Enhanced descriptions with metadata         |

Additional derived features:
- `missing_description`: Flag for books lacking descriptions
- `age_of_book`: Years since publication
- `no_words_description`: Word count in descriptions
- `title_subs`: Combined title and subtitle
- `simple_categories`: Normalized genre categories

---

## 📁 Project Structure

```
.
├── 1.data_exploration.ipynb      # Initial data analysis and preparation
├── 2.vector_search.ipynb         # Vector embeddings and similarity search
├── 3.text_classification.ipynb   # Book category classification
├── 4.sentimental_analysis.ipynb  # Emotion analysis of book descriptions
├── 5.gradio.py                   # Interactive web interface
├── books_with_cats.csv           # Dataset with categorized books
├── books_with_emotion.csv        # Dataset with emotion analysis
├── cleaned_books.csv             # Preprocessed dataset
├── cover-not-found.jpg.          # Default image for missing covers
├── tagged_description.txt        # Enhanced book descriptions
├── requirements.txt              # Project dependencies
├── .env                          # Environment variables (API keys, etc.)
└── README.md                     # Project documentation
```

---

## 🛠️ Installation & Setup

> **Python Version Requirements:**  
> - Python 3.12.7  
> - PyTorch 2.7.1 (GPU version)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/LangChain-Book_Recommender.git
   cd LangChain-Book_Recommender
   ```

2. **Set up a Python environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:**  
   > Make sure you have the correct CUDA drivers for PyTorch GPU support.  
   > To install PyTorch 2.7.1 with GPU, you can use:
   > ```bash
   > pip install torch==2.7.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu121
   > ```

4. **Configure environment variables**
   - Create a `.env` file with necessary API keys (e.g., for LangChain, embedding services).

---

## 🚀 Usage

### Notebook Exploration

Explore the project step by step:

1. **Data Exploration**: Run `1.data_exploration.ipynb` to understand the dataset structure and preprocessing steps.
2. **Vector Search**: Explore `2.vector_search.ipynb` to see how book embeddings are created and similarity search works.
3. **Text Classification**: Review `3.text_classification.ipynb` to understand the book categorization process.
4. **Sentiment Analysis**: Examine `4.sentimental_analysis.ipynb` to see how emotional analysis enhances recommendations.

### Web Interface

Launch the Gradio web interface to interact with the recommendation system:

```bash
python 5.gradio.py
```

Then open your browser and navigate to the URL displayed in the terminal (typically http://localhost:7860).

---

## 🧠 Machine Learning Components

### Vector Embeddings
- Books are converted into high-dimensional vectors representing their content.
- Similarity search finds books with similar themes and content.
- Uses dimensionality reduction techniques for visualization.

### Text Classification
- ML models categorize books into proper genres.
- Improves upon original category data by standardizing and enriching classifications.
- Handles multi-category books appropriately.

### Sentiment Analysis
- Analyzes emotional tone in book descriptions.
- Extracts emotional characteristics to match reader preferences.
- Provides another dimension for personalized recommendations.

---

## 💻 Web Interface

The Gradio interface offers several ways to find book recommendations:

- **Title Search**: Find books similar to your favorites.
- **Description-Based**: Enter a description of what you're looking for.
- **Category Filtering**: Browse by genre/category.
- **Mood Selection**: Find books matching specific emotional tones.
- **Advanced Options**: Combine multiple criteria for precise recommendations.

---

## 🔮 Future Improvements

- Incorporate user reading history for personalized recommendations.
- Add collaborative filtering based on user behavior.
- Expand dataset with additional book information.
- Implement more advanced NLP features with larger language models.
- Create mobile-friendly version of the web interface.

---

## 👥 Contributors

- Your Name (@yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Created with LangChain, Jupyter, and Gradio*