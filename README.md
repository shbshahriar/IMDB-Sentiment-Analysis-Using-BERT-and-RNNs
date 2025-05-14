# IMDB Sentiment Analysis Using BERT and RNNs

This project focuses on sentiment analysis using the IMDB dataset of 50,000 movie reviews. The repository contains implementations of various deep learning models for text classification, including BERT, BiGRU, BiLSTM, and BiRNN.

## Project Structure

The repository is organized as follows:

```
IMDB_Dataset_of_50K_Movie_Reviews/
├── Code/
│   ├── imdb_Bert_tf.ipynb       # Implementation of BERT for sentiment analysis
│   ├── imdb_BiGRU.ipynb         # Implementation of BiGRU for sentiment analysis
│   ├── imdb_BiLSTM.ipynb        # Implementation of BiLSTM for sentiment analysis
│   ├── imdb_BiRNN.ipynb         # Implementation of BiRNN for sentiment analysis
├── Logs/
│   ├── bert_training_log.csv    # Training logs for BERT
│   ├── BiGRU_training_log.csv   # Training logs for BiGRU
│   ├── BiLSTM_training_log.csv  # Training logs for BiLSTM
│   ├── BiRNN_training_log.csv   # Training logs for BiRNN
├── Model/
│   ├── BiGRU.h5                 # Trained BiGRU model
│   ├── BiLSTM.h5                # Trained BiLSTM model
│   ├── BiRNN.h5                 # Trained BiRNN model
│   ├── imdb_best_model.h5       # Best performing model
│   ├── bert_best_model/         # Directory containing the best BERT model
│       ├── config.json
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── training_args.bin
│       ├── vocab.txt
```

## Requirements

To run the notebooks and train the models, you need the following dependencies:

- Python 3.8 or later
- TensorFlow
- PyTorch
- Transformers
- NumPy
- Pandas
- Matplotlib

You can install the required packages using the following command:

```bash
pip install tensorflow torch transformers numpy pandas matplotlib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/shbshahriar/IMDB-Sentiment-Analysis-Using-BERT-and-RNNs.git

   cd IMDB-Sentiment-Analysis-Using-BERT-and-RNNs

   ```

2. Open the desired Jupyter Notebook from the `Code/` directory to train or evaluate a model.

3. Training logs are saved in the `Logs/` directory, and trained models are saved in the `Model/` directory.

## Models

### BERT
- **Notebook**: `imdb_Bert_tf.ipynb`
- **Best Model**: Stored in `Model/bert_best_model/`

### BiGRU
- **Notebook**: `imdb_BiGRU.ipynb`
- **Best Model**: `Model/BiGRU.h5`

### BiLSTM
- **Notebook**: `imdb_BiLSTM.ipynb`
- **Best Model**: `Model/BiLSTM.h5`

### BiRNN
- **Notebook**: `imdb_BiRNN.ipynb`
- **Best Model**: `Model/BiRNN.h5`

## Dataset

The IMDB dataset contains 50,000 movie reviews labeled as positive or negative. It is widely used for sentiment analysis tasks.

## Acknowledgments

- The IMDB dataset is provided by [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/).
- The BERT model is implemented using the [Hugging Face Transformers library](https://huggingface.co/transformers/).


---

Feel free to contribute to this project by submitting issues or pull requests!
