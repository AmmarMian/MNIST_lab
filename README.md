# MNIST and Fashion-MNIST Lab Solutions

This repository contains complete solutions for the three-part lab session on numerical optimization for machine learning, focusing on MNIST digit recognition and Fashion-MNIST classification.

## Prerequisites

Before running the solutions, ensure you have the following packages installed in your environment (as described in the lab environment setup):

```bash
# Using uv (recommended)
uv add numpy scipy scikit-learn matplotlib plotly torch torchvision seaborn

# Or using pip
pip install numpy scipy scikit-learn matplotlib plotly torch torchvision seaborn
```

## File Structure

```
.
├── part1_classical_ml_complete.py      # MLP from scratch + SVM
├── part2_cnn_complete.py               # CNNs with PyTorch
├── part3_transfer_complete.py          # Transfer learning on Fashion-MNIST
├── utils.py                            # Common utility functions
└── README.md                           # This file
```

## Part 1: Classical Machine Learning (MNIST)

### Overview
This part implements:
- Multi-Layer Perceptron (MLP) from scratch with numpy
- Backpropagation and stochastic gradient descent
- Support Vector Machines using scikit-learn
- Proper validation methodology with K-fold cross-validation

### Key Concepts Demonstrated
- **Manual gradient computation**: Understanding backpropagation mathematics
- **Optimization algorithms**: SGD with momentum implementation
- **Hyperparameter tuning**: Grid search for SVM parameters
- **Model comparison**: MLP vs SVM performance analysis

### Running Part 1
```bash
python part1_classical_ml_complete.py
```

### Expected Output
- Visualization of MNIST samples
- t-SNE visualization of digit embeddings
- Learning rate experiment results
- Batch size experiment results
- SVM grid search results
- Confusion matrices for both models
- Cross-validation scores

### Key Results to Observe
1. **Learning Rate Impact**: Notice how different learning rates affect convergence
2. **Batch Size Trade-offs**: Smaller batches = noisier gradients but better generalization
3. **Model Comparison**: SVM typically achieves ~98% accuracy, MLP ~97%

## Part 2: Deep Learning with CNNs (MNIST)

### Overview
This part implements:
- Convolutional Neural Networks using PyTorch
- Various optimization algorithms (SGD, Adam, RMSprop)
- Learning rate scheduling strategies
- Advanced CNN architectures

### Key Concepts Demonstrated
- **Automatic differentiation**: PyTorch's autograd system
- **CNN architecture design**: Convolutional layers, pooling, dropout
- **Optimizer comparison**: Understanding different optimization algorithms
- **Learning rate scheduling**: Adaptive learning rates for better convergence

### Running Part 2
```bash
python part2_cnn_complete.py
```

### Expected Output
- Training curves for different experiments
- Filter visualizations from convolutional layers
- Feature map visualizations
- Optimizer comparison plots
- Confusion matrix with >99% accuracy

### Key Results to Observe
1. **CNN vs MLP**: Significant accuracy improvement (>99% vs ~97%)
2. **Optimizer Differences**: Adam typically converges faster than SGD
3. **Learning Rate Scheduling**: Helps escape plateaus and achieve better minima
4. **Feature Learning**: First layer learns edge detectors

## Part 3: Transfer Learning (Fashion-MNIST)

### Overview
This part implements:
- Transfer learning with pre-trained models
- Two-stage training strategy
- Comparison with training from scratch
- Feature space analysis

### Key Concepts Demonstrated
- **Transfer learning benefits**: Faster convergence, better performance
- **Feature extraction vs fine-tuning**: When to use each approach
- **Domain adaptation**: Handling grayscale images with RGB models
- **Feature visualization**: Understanding learned representations

### Running Part 3
```bash
python part3_transfer_complete.py
```

### Expected Output
- Fashion-MNIST sample visualization
- Two-stage training progress
- Comparison plots (transfer vs from-scratch)
- t-SNE visualization of learned features
- Detailed classification reports

### Key Results to Observe
1. **Faster Convergence**: Transfer learning reaches high accuracy in fewer epochs
2. **Better Final Performance**: ~92% vs ~89% accuracy
3. **Feature Quality**: t-SNE shows better class separation
4. **Parameter Efficiency**: Most parameters remain frozen

## Utility Functions

The `utils.py` file provides reusable functions for all parts:

### Visualization
- `plot_samples()`: Display image grids
- `plot_training_curves()`: Training/validation curves
- `plot_confusion_matrix()`: Confusion matrix visualization

### Evaluation
- `evaluate_model()`: Comprehensive model evaluation
- `print_classification_report()`: Detailed per-class metrics
- `analyze_predictions()`: Most/least confident predictions

### PyTorch Helpers
- `set_random_seeds()`: Reproducibility
- `get_device()`: GPU/CPU selection
- `count_parameters()`: Model size analysis

## Experimental Insights

### Part 1 Insights
1. **Batch Size**: Smaller batches (32-128) often generalize better despite noisier gradients
2. **Learning Rate**: 0.01 works well for normalized MNIST data
3. **SVM Kernels**: RBF kernel outperforms linear for MNIST

### Part 2 Insights
1. **Architecture**: Even simple CNNs dramatically outperform MLPs on image data
2. **Batch Normalization**: Speeds up training and improves stability
3. **Dropout**: Essential for preventing overfitting in deeper networks

### Part 3 Insights
1. **When Transfer Learning Helps**: 
   - Limited training data
   - Similar visual features
   - Computational constraints
2. **Fine-tuning Strategy**: Start with frozen backbone, then unfreeze gradually
3. **Learning Rate**: Use lower LR for fine-tuning pre-trained weights

## Common Issues and Solutions

### Memory Issues
If you encounter GPU memory errors:
```python
# Reduce batch size
batch_size = 32  # Instead of 64

# Clear cache periodically
torch.cuda.empty_cache()
```

### Slow Training
For faster experimentation:
```python
# Use subset of data
X_train_subset = X_train[:10000]

# Reduce number of epochs
epochs = 10  # Instead of 20-30
```

### Reproducibility
Always set random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
```

## Extensions and Exercises

### Part 1 Extensions
1. Implement different activation functions (tanh, leaky ReLU)
2. Add L2 regularization to MLP
3. Implement adaptive learning rates (AdaGrad)

### Part 2 Extensions
1. Implement data augmentation
2. Try residual connections
3. Experiment with different CNN architectures

### Part 3 Extensions
1. Try different pre-trained models (EfficientNet, ViT)
2. Implement gradual unfreezing
3. Apply to custom datasets


## Tips for Understanding

1. **Start with Part 1**: Understanding the manual implementation helps appreciate what frameworks do
2. **Visualize Everything**: Use the provided visualization functions liberally
3. **Experiment with Hyperparameters**: Change values and observe effects
4. **Read the Mathematics**: The backpropagation derivations in Part 1 are crucial
5. **Compare Approaches**: Understanding when to use each technique is key

## Performance Benchmarks

Expected accuracies on test sets:
- Part 1 MLP: ~97%
- Part 1 SVM: ~98%
- Part 2 Basic CNN: ~99.0%
- Part 2 Improved CNN: ~99.3%
- Part 3 From Scratch: ~89%
- Part 3 Transfer Learning: ~92%

## Conclusion

These solutions demonstrate the evolution of machine learning techniques from classical methods to modern deep learning. The progression from manual implementation to framework usage to transfer learning mirrors the historical development of the field and provides comprehensive understanding of optimization in machine learning.
