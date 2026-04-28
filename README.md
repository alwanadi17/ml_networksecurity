# ML Tabular Framework 🚀

**A High-Performance, Config-Driven Framework for Stateful Tabular Modeling.**

`ml_tabular_framework` is a specialized Python framework designed to streamline the end-to-end tabular data pipeline—from transformation to model training. It prioritizes developer experience, configuration flexibility, and memory efficiency.

## 🌟 The Core Problem Solved

Traditional machine learning pipelines often suffer from high RAM consumption and lack of persistence between stages. If a process crashes or a model needs re-training, the entire preprocessing phase must be repeated. 

**This framework changes that.** By implementing a storage-based persistence layer, it ensures that every transformation is reusable and independent of volatile memory.

## ✨ Key Features

- **RAM Independence:** Unlike traditional pipelines that hold all objects in memory, this framework persists intermediate results to disk. You can process large datasets without Out-of-Memory (OOM) errors.
- **Stateful Reusability:** Intermediate transformations are cached. If you modify your model parameters, you can skip the heavy preprocessing and resume directly from the stored state.
- **Declarative Configuration:** Manage complex data flows via intuitive YAML/JSON configurations. Decouple your transformation logic from your execution code.
- **Modular Data Flow:** Easily edit, swap, or experiment with different data transformations and model architectures without breaking the entire pipeline.

## 🛠 Tech Stack

- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Persistence Layer:** Pickle / Parquet / HDF5 Support

## 🏗 Architecture & Data Flow

The framework operates on a decoupled architecture:
1. **Configuration Engine:** Loads user-defined transformation and training parameters.
2. **Persistence-Aware Transformer:** Executes data cleaning, encoding, and scaling, then automatically serializes the state to storage.
3. **Storage Layer:** Acts as a persistent cache between the preprocessing and training phases.
4. **Execution Trainer:** Retrieves ready-to-use data from storage for high-speed model training and evaluation.

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone [https://github.com/alwanadi17/ml_tabular_framework.git](https://github.com/alwanadi17/ml_tabular_framework.git)

# Navigate to the directory
cd ml_tabular_framework

# Install dependencies
pip install -r requirements-dev.txt
```

## 📈 Roadmap

- [ ] Implement MLFlow and Dagshub for report visualization.

- [ ] Automated Hyperparameter Optimization (HPO) support.

## 🤝 Contributing
Contributions are welcome! If you have ideas for optimization or new features, please feel free to open an issue or submit a pull request.
