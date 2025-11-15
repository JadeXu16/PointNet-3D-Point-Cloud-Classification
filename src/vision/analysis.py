from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision.part2_baseline import Baseline
from vision.part3_pointnet import PointNet
from vision.part5_tnet import PointNetTNet


def get_critical_indices(model: Union[PointNet, PointNetTNet], pts: torch.Tensor) -> np.ndarray:
    '''
    Finds the indices of the critical points in the given point cloud. A
    critical point is a point that contributes to the global feature (i.e
    a point whose calculated feature has a maximal value in at least one 
    of its dimensions)
    
    Hint:
    1) Use the encodings returned by your model
    2) Make sure you aren't double-counting points since points may
       contribute to the global feature in more than one dimension

    Inputs:
        model: The trained model
        pts: (model.pad_size, 3) tensor point cloud representing an object

    Returns:
        crit_indices: (N,) numpy array, where N is the number of critical pts

    '''
    crit_indices = None


    # Set model to evaluation mode
    model.eval()
    
    # Add a batch dimension to the input points (N, 3) -> (1, N, 3) and send to the model's device
    try:
        # Try to get the device from the model's parameters
        device = next(model.parameters()).device
    except StopIteration:
        # If the model has NO parameters (like the test model),
        # default to 'cpu'
        device = 'cpu'
    
    pts_batch = pts.unsqueeze(0).to(device)

    # Get encodings from the model
    with torch.no_grad(): # Disable gradient calculation
        _, encodings = model(pts_batch) 
        
    # encodings shape is (1, N, 1024)
    # Remove the batch dimension
    encodings_squeeze = encodings.squeeze(0)  # shape (N, 1024)

    # Find the *index* of the max value for each of the 1024 dimensions.
    # We find the max along dim=0 (the N points dimension).
    # indices_per_dim will have shape (1024,)
    _, indices_per_dim = torch.max(encodings_squeeze, dim=0)
    
    # Find the *unique* indices as requested
    crit_indices_tensor = torch.unique(indices_per_dim)

    # Convert to numpy array
    crit_indices = crit_indices_tensor.cpu().numpy()

    # Set model back to train mode
    model.train()


    return crit_indices

    
def get_confusion_matrix(
    model: Union[Baseline, PointNet, PointNetTNet], 
    loader: DataLoader, 
    num_classes: int,
    normalize: bool=True, 
    device='cpu'
) -> np.ndarray:
    '''
    Builds a confusion matrix for the given models predictions
    on the given dataset. 
    
    Recall that each ground truth label corresponds to a row in
    the matrix and each predicted value corresponds to a column.

    A confusion matrix can be normalized by dividing entries for
    each ground truch prior by the number of actual isntances the
    ground truth appears in the dataset. (Think about what this means
    in terms of rows and columns in the matrix) 

    Hint:
    1) Generate list of prediction, ground-truth pairs
    2) For each pair, increment the correct cell in the matrix
    3) Keep track of how many instances you see of each ground truth label
       as you go and use this to normalize 

    Args: 
    -   model: The model to use to generate predictions
    -   loader: The dataset to use when generating predictions
    -   num_classes: The number of classes in the dataset
    -   normalize: Whether or not to normalize the matrix
    -   device: If 'cuda' then run on GPU. Run on CPU by default

    Output:
    -   confusion_matrix: a numpy array with shape (num_classes, num_classes)
                          representing the confusion matrix
    '''

    model.eval()
    confusion_matrix = None


    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    # Initialize a counter for ground truth instances (for normalization)
    gt_counts = np.zeros(num_classes, dtype=np.float32)

    # Loop over the data loader
    with torch.no_grad(): # We don't need gradients here
        for data, labels in loader:
            # Move data and labels to the specified device
            data = data.to(device)
            labels = labels.to(device)

            # Get model outputs
            outputs, _ = model(data)

            # Find the predicted class (index with the highest score)
            # preds shape will be (B,)
            _, preds = torch.max(outputs, dim=1)

            # Move labels and predictions to CPU to use with numpy
            labels_cpu = labels.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            # Iterate over the batch and populate the matrix
            for i in range(len(labels_cpu)):
                gt_label = labels_cpu[i]
                pred_label = preds_cpu[i]
                
                # Row = ground truth, Column = prediction
                confusion_matrix[gt_label, pred_label] += 1
                gt_counts[gt_label] += 1

    # Normalize the matrix if requested
    if normalize:
        # We need to divide each row by the total number of GT instances for that row
        
        # Avoid division by zero if a class has 0 instances in the loader, We set the count to 1, so 0 / 1 = 0
        gt_counts[gt_counts == 0] = 1.0 
        
        # Use numpy broadcasting to divide each row
        # .reshape(-1, 1) makes gt_counts a (num_classes, 1) column vector
        confusion_matrix = confusion_matrix / gt_counts.reshape(-1, 1)


    model.train()

    return confusion_matrix