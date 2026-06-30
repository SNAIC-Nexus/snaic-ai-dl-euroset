# README: Deep Learning Project on CNN and Transformer for EuroSAT Classification

This project explores the application of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) for classifying satellite imagery from the EuroSAT dataset. The primary goal is to compare the performance of these two distinct deep learning architectures in the context of geospatial land-use classification.

## Project Overview

The EuroSAT dataset consists of 27,000 labeled satellite images covering 10 land-use classes. This notebook systematically guides through the entire deep learning workflow:

1.  **Environment Setup & Data Loading**: Installs necessary libraries and efficiently loads the EuroSAT RGB dataset, preparing it for high-throughput processing.
2.  **Data Processing & Augmentation**: Implements geometric data augmentations directly on the GPU to maximize hardware utilization and improve model generalization. The raw images are cached in memory to avoid I/O bottlenecks.
3.  **Model Training**: Trains four different configurations:
    *   ResNet50 (CNN) with a frozen backbone and fine-tuned head.
    *   ResNet50 (CNN) with an unfrozen backbone (fine-tuning the entire model).
    *   ViT (Transformer) with a frozen backbone and fine-tuned head.
    *   ViT (Transformer) with an unfrozen backbone (fine-tuning the entire model).
    Each training run utilizes mixed-precision bfloat16 auto-casting and streams metrics to Weights & Biases for real-time tracking.
4.  **Local Evaluation & Performance Dashboard**: Analyzes the training results, presenting a comparison table of final metrics, dual-axes convergence plots for loss and accuracy, and a confusion matrix heatmap for the best-performing model.
5.  **Comparative Inference Interface**: Provides an interactive Gradio web application for side-by-side comparison of CNN and Transformer model predictions on live validation samples.

## Key Technologies Used

*   **PyTorch**: Deep learning framework for model implementation and training.
*   **Hugging Face Transformers**: Leveraged for pre-trained CNN (ResNet50) and Transformer (ViT) models.
*   **`torchvision.transforms.v2`**: For efficient on-GPU data augmentation.
*   **Weights & Biases (WandB)**: For experiment tracking, visualization, and logging of training metrics.
*   **Gradio**: For creating an interactive web-based demo interface.
*   **`datasets` library**: For easy loading and management of the EuroSAT dataset.

## Project Structure

The notebook is organized into distinct cells, each addressing a specific stage of the deep learning pipeline, from data preparation to model evaluation and interactive demonstration. The comments and markdown cells provide detailed explanations for each step and justification for design choices, particularly regarding performance optimization and rubric requirements.
