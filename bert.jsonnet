local embedding_dim = 768;
local hidden_dim = 128;



{
  "dataset_reader": {
    "type": "sst_tokens",
    "token_indexers": {
      "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased"
      }
    }
  },
  "train_data_path": "data/stanfordSentimentTreebank/trees/train.txt",
  "validation_data_path": "data/stanfordSentimentTreebank/trees/dev.txt",

  "model": {

    "type": "lstm_classifier",

    "word_embeddings": {
        "allow_unmatched_keys": true,

        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets"]
        },

        "token_embedders": {
            "tokens": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased"
            }
        }
    },

    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim
    }
  },

  "iterator": {
    "type": "bucket",
    "batch_size": 32,
    "sorting_keys": [["tokens", "num_tokens"]]
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 20,
    "patience": 5
  }
}

