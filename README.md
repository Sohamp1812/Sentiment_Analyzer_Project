# Social Media Sentiment Analysis (CNN-LSTM)

## Project Overview
This project develops a deep learning model to classify social media comments into positive or negative sentiments. Initially, a 3-class classification (positive, neutral, negative) was attempted, but to achieve higher accuracy and clearer actionable insights, the problem was refined to a binary classification task.

The model leverages a Convolutional Neural Network (CNN) combined with a Bidirectional Long Short-Term Memory (Bi-LSTM) network, utilizing pre-trained GloVe word embeddings.

## Key Features
-   **Data Preprocessing:** Cleaning and preparing raw social media text data.
-   **Word Embeddings:** Utilization of 100-dimensional GloVe pre-trained word vectors for semantic representation.
-   **CNN-LSTM Architecture:** A hybrid deep learning model combining CNN for local feature extraction and Bi-LSTM for sequential context understanding.
-   **Binary Classification:** Focuses on distinguishing between 'positive' and 'negative' sentiments.
-   **Model Evaluation:** Comprehensive evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
-   **Overfitting Mitigation:** Implementation of Early Stopping and Dropout layers during training.

## Results
The CNN-LSTM binary classification model achieved an **accuracy of 84.32%** on unseen test data, significantly exceeding the target performance of 70-75%.

## Technologies Used
-   Python 3.x
-   TensorFlow / Keras (for Deep Learning)
-   Pandas (for data manipulation)
-   NumPy (for numerical operations)
-   scikit-learn (for data splitting and evaluation metrics)
-   NLTK (for text preprocessing)
-   Seaborn & Matplotlib (for visualizations)
-   Joblib (for saving/loading models and parameters)

## Project Structure
.
#── data/
#   #── cleaned_social_media_comments.csv
#   └── glove/
#       └── glove.6B.100d.txt  # Pre-trained GloVe embeddings (download separately if very large)
#── models/
#   #── best_cnn_lstm_binary_sentiment_model.keras  # Trained binary classification model
#   #── dl_tokenizer_binary.joblib                  # Tokenizer for binary data
#   └── dl_params_binary.joblib                     # Parameters (max_seq_len, vocab_size, label_map)
#── notebooks/
#   └── sentiment_analysis_cnn_lstm.ipynb           # Jupyter Notebook with full project workflow
#── README.md
└── requirements.txt

How to Run the Project

1. Clone the repository:
```bash
git clone [https://github.com/Sohamp1812/Sentiment_Analyzer_Project.git](https://github.com/Sohamp1812/Sentiment_Analyzer_Project.git)
cd Sentiment_Analyzer_Project

2. Set up Python Environment:
It is recommended to use a virtual environment or Anaconda.

# Assuming you have Anaconda installed and can activate base or your environment
conda activate base # Or your specific environment name

# Install dependencies
pip install -r requirements.txt

3. Download GloVe Embeddings:
Since data/glove/glove.6B.100d.txt is NOT included in this repository (due to GitHub's file size limits), you must download it manually:

Go to: https://nlp.stanford.edu/projects/glove/
Download glove.6B.zip (it's a large file).
Unzip the file and place glove.6B.100d.txt specifically into the data/glove/ directory within this project.

4. Run the Jupyter Notebook:

Open notebooks/sentiment_analysis_cnn_lstm.ipynb and run all cells sequentially to reproduce the results.

Author
Soham
