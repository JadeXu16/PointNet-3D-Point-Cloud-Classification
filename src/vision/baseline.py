from typing import Tuple

import torch
from torch import nn


class Baseline(nn.Module):
    '''
    A simple baseline that counts points per voxel in the point cloud
    and then uses a linear classifier to make a prediction
    '''
    def __init__(self,
        classes: int,
        voxel_resolution=4,
        mode="count"
    ) -> None:
        '''
        Constructor for Baseline to define layers.

        Args:
        -   classes: Number of output classes
        -   voxel_resolution: Number of positions per dimension to count
        -   mode: Whether to count the number of points per voxel ("count") or just check binary occupancy ("occupancy")
        '''
        assert mode in ["count", "occupancy"]

        super().__init__()

        self.classifier = None
        self.voxel_resolution = None
        self.mode = None


        self.voxel_resolution = voxel_resolution
        self.mode = mode
        
        # Calculate the total number of voxels
        # This will be the input dimension for our classifier
        in_features = voxel_resolution ** 3
        
        self.classifier = nn.Linear(in_features, classes)



    def count_points(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Create the feature as input to the linear classifier by counting the number of points per voxel.
        This is effectively taking a 3D histogram for every item in a batch.

        Hint: 
        1) torch.histogramdd will be useful here

        Args:
        -   x: tensor of shape (B, N, in_dim)

        Output:
        -   counts: tensor of shape (B, voxel_resolution**3), indicating the percentage of points that landed in each voxel
        '''

        counts = None


        batch_size = x.shape[0]
        batch_counts = []
        
        
        # We must iterate over the batch, as histogramdd is not batched
        for i in range(batch_size):
            # Get a single point cloud, shape (N, 3)
            pc = x[i]

            # Create bin edges as a Tensor
            # (voxel_resolution + 1) edges define voxel_resolution bins
            # We use .device and .dtype to ensure the tensor is on the correct device (CPU/GPU)
            edges = torch.linspace(-2.0, 2.0, steps=self.voxel_resolution + 1, device=pc.device, dtype=pc.dtype)
            
            # bins parameter is now a list of Tensors
            bins = [edges, edges, edges]
            
            # Compute the 3D histogram
            # Note: we no longer need to pass the range_tuple parameter
            hist = torch.histogramdd(pc, bins=bins).hist
            
            # Flatten the (res, res, res) tensor to (res**3)
            hist_flat = hist.view(-1)
            
            # Normalize the feature vector to get "percentage"
            hist_sum = hist_flat.sum()
            hist_normalized = hist_flat / hist_sum
            
            batch_counts.append(hist_normalized)
            
        # Stack all feature vectors into a single batch tensor
        # Shape: (B, voxel_resolution**3)
        counts = torch.stack(batch_counts, dim=0)


        return counts


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the Baseline model. Make sure you handle the case where the mode 
        is set to "occupancy" by thresholding the result of count_points on zero.

        Args:
            x: tensor of shape (B, N, 3), where B is the batch size and N is the number of points per
               point cloud
        Output:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   None, just used to allow reuse of training infrastructure
        '''

        class_outputs = None


        # Get the voxel features from the input point clouds
        # shape: (B, voxel_resolution**3)
        features = self.count_points(x)
        
        # Check if we are in "occupancy" mode
        if self.mode == "occupancy":
            # If so, convert counts to binary
            features = (features > 0).float()
            
        # Pass the features (either counts or occupancy) through the linear classifier
        # output shape: (B, classes)
        class_outputs = self.classifier(features)


        return class_outputs, None