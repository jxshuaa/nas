# Neural Architecture Search (NAS) Project

## Overview
This project implements various Neural Architecture Search (NAS) techniques to automatically discover optimal neural network architectures for specific tasks. The project includes implementations of Random Search, Reinforcement Learning-based NAS, Evolutionary Algorithms-based NAS, and Gradient-based NAS approaches.

## Project Structure
```
neural-architecture-search/
|-- data/                   # Dataset and data loaders
|-- models/                 # Neural network models
|-- scripts/                # Training and evaluation scripts
|-- search/                 # NAS algorithms and search space definitions
|-- notebooks/              # Jupyter notebooks for experimentation
|-- utils/                  # Utility functions
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
|-- main.py                 # Main script to run the NAS
```

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/neural-architecture-search.git
cd neural-architecture-search
```

2. Create a virtual environment (recommended):
```bash
python -m venv nas_env
source nas_env/bin/activate  # On Windows: nas_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running a Basic NAS Experiment
```bash
python main.py --algorithm random --dataset cifar10 --budget 100
```

Available algorithms:
- `random`: Random search (baseline)
- `rl`: Reinforcement Learning-based NAS
- `ea`: Evolutionary Algorithms-based NAS
- `gradient`: Gradient-based NAS

Available datasets:
- `cifar10`: CIFAR-10 dataset
- `mnist`: MNIST dataset

### Using the Notebooks
The `notebooks/` directory contains Jupyter notebooks for experimentation and visualization:
- `01_search_space_exploration.ipynb`: Explore and define the search space
- `02_algorithm_comparison.ipynb`: Compare different NAS algorithms
- `03_performance_visualization.ipynb`: Visualize the results and performance metrics

## Search Space
The current implementation includes the following search space:
- Number of layers: 1-4
- Number of units/filters per layer: 16, 32, 64, 128
- Activation functions: ReLU, Tanh, Sigmoid
- Optimizers: Adam, SGD

## NAS Algorithms
1. **Random Search**: Simple baseline that randomly samples architectures from the search space
2. **Reinforcement Learning-based NAS**: Uses a controller RNN trained with policy gradient to generate architectures
3. **Evolutionary Algorithms-based NAS**: Uses genetic algorithms to evolve architectures over generations
4. **Gradient-based NAS**: Uses gradient descent to optimize the architecture parameters

## Results
Performance comparisons between different NAS algorithms will be visualized in TensorBoard and saved in the `results/` directory.

## License
MIT

## References
- Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. ICLR.
- Pham, H., Guan, M., Zoph, B., Le, Q., & Dean, J. (2018). Efficient Neural Architecture Search via Parameter Sharing. ICML.
- Jin, H., Song, Q., & Hu, X. (2019). Auto-Keras: An Efficient Neural Architecture Search System. KDD. 