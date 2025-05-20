# DL_Assignment_3

# DA6401 Assignment 3

Assignment 3 of the DA6401: Fundamentals of Deep Learning course by Udit Narayan (CS23S038)

### Instructions

1. Add WandB **API-KEY** and **Entity-Name**

   - Insert the WandB API-KEY and entity-name in the `train.py` file within the _main_ function and the _train_rnn_ function.

2. Training Setup

   - To install the required dependencies, run: `pip install -r requirements.txt`

3. Download the dataset using: `python data.py`

   - Start the training process with: `python train.py`

4. WandB Sweep Setup

   - To initiate the WandB sweep, simply call the **main()** function in **train.py**.

---



Example usage: `python train.py --epochs 10 --batchsize 64 --hidden_size 256`

## WandB Sweep Configuration

The WandB sweep utilizes a **Bayesian** optimization method based on the following configuration:

```config = {
    "method": 'bayes',
    "metric": {
        'name': 'val_acc',
        'goal': 'maximize'
    },
    'parameters': {
        "epochs": {"values": [10, 15, 20]},
        "batchsize": {"values": [32, 64, 128]},
        "embedding_size": {"values": [128, 256, 512]},
        "hidden_size": {"values": [256, 512, 1024]},
        "encoder_layers": {"values": [2, 3, 5]},
        "decoder_layers": {"values": [2, 3, 5]},
        "cell_type": {"values": ["GRU", "LSTM", "RNN"]},
        "bi_directional": {"values": ["Yes", "No"]},
        "dropout": {"values": [0.2, 0.3, 0.5]},
        "attention": {"values": ["Yes", "No"]}
    }
}
```

### Training Iteration

The training iteration function is defined as follows:

```
train_iter(
    train_dataloader, val_dataloader, val_y, input_len, target_len,
    epochs, batchsize, embedding_size,
    encoder_layers, decoder_layers, hidden_size,
    cell_type, bi_directional, dropout,
    attention
)
```
