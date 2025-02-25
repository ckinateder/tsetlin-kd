from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from pickle import dump, load
from time import time
import random
import matplotlib.pyplot as plt
from util import load_pkl, load_or_create
from datasets import MNISTDataset
import os
def visualize_activation_maps(teacher_model, student_model, distilled_model, sample, image_shape, output_filepath, class_idx=None):
    """
    Creates a visualization of activation maps for teacher, student, and distilled models.
    
    Args:
        teacher_model: The teacher TsetlinMachine model
        student_model: The student TsetlinMachine model
        distilled_model: The distilled TsetlinMachine model
        sample: Input sample to visualize
        image_shape: Tuple with image dimensions (height, width)
        output_filepath: Path where to save the output image
        class_idx: Specific class index to visualize. If None, uses the predicted class.
    """
    # Create figure with 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # If class_idx is not provided, use the predicted class from the teacher model
    if class_idx is None:
        # Get prediction from teacher model
        teacher_output = teacher_model.predict(sample.reshape(1, -1))
        class_idx = teacher_output[0]

    class_idx = int(class_idx)

    # Display original image
    axes[0].imshow(sample.reshape(image_shape), cmap='gray')
    axes[0].set_title("Original Sample")
    axes[0].axis('off')
    
    # Generate and display teacher activation map
    teacher_activation = teacher_model.get_activation_map(sample, class_idx=class_idx, image_shape=image_shape)
    axes[1].imshow(teacher_activation)
    axes[1].set_title("Teacher Model Features")
    axes[1].axis('off')
    
    # Generate and display student activation map
    student_activation = student_model.get_activation_map(sample, class_idx=class_idx, image_shape=image_shape)
    print(student_model.number_of_features)
    axes[2].imshow(student_activation)
    axes[2].set_title("Student Model Features")
    axes[2].axis('off')
    
    # Generate and display distilled activation map
    distilled_activation = distilled_model.get_activation_map(sample, class_idx=class_idx, image_shape=image_shape)
    print(distilled_model.number_of_features)
    axes[3].imshow(distilled_activation)
    axes[3].set_title("Distilled Model Features")
    axes[3].axis('off')
    
    # Add overall title
    plt.suptitle(f"Activation Maps Comparison for Class {class_idx}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=150)
    plt.close()
    
    print(f"Activation maps comparison saved to {output_filepath}")

if __name__ == "__main__":
    dataset = load_or_create("data/mnist_dataset.pkl", MNISTDataset)
    X_train, Y_train, X_test, Y_test = dataset.get_data()
    
    experiment_path = os.path.join("results", "downsample", "MNIST-Downsample-Huge", "ds_tnc3500_snc500_T6400_s5.0_te30_se60_downsample0.01")
    teacher_model = load_pkl(os.path.join(experiment_path, "teacher_baseline.pkl"))
    student_model = load_pkl(os.path.join(experiment_path, "student_baseline.pkl"))
    distilled_model = load_pkl(os.path.join(experiment_path, "distilled.pkl"))

    visualize_activation_maps(teacher_model, student_model, distilled_model, X_test[0], (28, 28), os.path.join(experiment_path, "activation_map.png"))
    
    
