# Efficient Attention Mechanisms for Long Text Sequences

## Project Goal
To implement, benchmark, and analyze several novel and existing attention mechanisms for processing long text sequences efficiently. The primary goal is to explore alternatives to the standard quadratic self-attention, focusing on reducing computational complexity and memory footprint while maintaining or improving performance on language modeling tasks.

## Directory Structure
```
/my_attention_research_project/
|-- configs/                   # YAML/JSON configuration files for experiments
|-- data/                      # Scripts for data downloading, preprocessing, DataLoader
|   |-- wikitext103.py
|   `-- tokenizer.py
|-- models/                    # Model implementations
|   |-- attention/             # Core attention mechanism implementations
|   |   |-- vanilla_mha.py
|   |   |-- ahc_attention.py
|   |   |-- pas_attention.py
|   |   `-- moae_attention.py
|   |-- common_layers.py       # Positional Encodings, FFN, LayerNorm etc.
|   `-- transformer.py         # Transformer Encoder/Decoder block definitions
|-- training/                  # Training scripts
|   |-- train_loop.py
|   `-- experiment_runner.py   # Script to launch experiments based on configs
|-- benchmarking/              # Scripts for benchmarking
|   |-- evaluate_perplexity.py
|   `-- measure_speed_memory.py
|-- utils/                     # Utility functions (e.g., logging, config parsing)
|-- notebooks/                 # Jupyter notebooks for analysis and visualization (optional)
`-- README.md                  # Project overview, setup, how to run experiments
```

## Setup Instructions

### Prerequisites
*   Python 3.8+
*   PyTorch (latest stable version recommended)
*   PyYAML (for configuration management)
*   Hugging Face `tokenizers` library (for tokenization)

### Installation (Example)
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd my_attention_research_project
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio
    pip install pyyaml
    pip install tokenizers
    # Add other dependencies as needed
    ```

## How to Run Experiments
(To be detailed later - this section will describe how to use `experiment_runner.py` with configuration files.)

```
