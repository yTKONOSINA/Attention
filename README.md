# BERT (Educational Implementation)

This project implements a BERT-tiny model (Transformer encoder) from scratch using pure Python lists for tensor operations (no PyTorch/NumPy for core math). It loads pre-trained weights from Hugging Face to verify correctness.

## Setup

### 1. Create Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

Install the required packages (mainly for tokenization, obtaining model's weights, and visualization).

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights

Run the helper script to download `bert-tiny` weights from Hugging Face and save them to the `weights/` directory.

```bash
python weights/get_model_weights.py
```

### 4. Run the Model

Run the main script from the root directory to execute the model on example sentences and visualize attention maps.

```bash
python python3 main.py
```

## Project Structure

- `tensor.py`: Custom Tensor class implementing matrix operations using standard lists.
- `Layers/`: Implementation of Transformer components (Linear, LayerNorm, Attention, Predictions).
- `main.py`: Runs the predictions and visualizes attention.
- `attention_visualization.py`: Plotting attention heatmaps.
