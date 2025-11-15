from typing import Tuple

import torch
from vision.part3_pointnet import PointNet
from torch import nn


class TNet(nn.Module):

    def __init__(self,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        regression_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for TNet to define layers.

        Hint: The architecture is almost the same as your PointNet, just with a different
              output dimension

        Just like with PointNet, you will need to repeat the first hidden dim.
        See mlp(64, 64) in the diagram. Furthermore, you will want to include
        a BatchNorm1d after each layer in the encoder except for the final layer
        for easier training.


        Args:
        -   classes: Number of output classes
        -   in_dim: Input dimensionality for points.
        -   hidden_dims: The dimensions of the encoding MLPs. This is similar to
                         that of PointNet
        -   regression_dims: The dimensions of regression MLPs. This is similar
                         to the classifier dims in PointNet
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.encoder_head = None
        self.regression_head = None
        self.in_dim = None

        
        # Store in_dim for the forward pass
        self.in_dim = in_dim

        # === Define the Encoder Head ===
        # This is identical to the PointNet encoder
        
        encoder_layers = []
        
        # Layer 1: in_dim -> 64 (e.g., 3 -> 64)
        encoder_layers.extend([
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]), # BN 1
            nn.ReLU()
        ])
        
        # Layer 2: 64 -> 64
        encoder_layers.extend([
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]), # BN 2
            nn.ReLU()
        ])
        
        # Layer 3: 64 -> 128
        encoder_layers.extend([
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]), # BN 3
            nn.ReLU()
        ])
        
        # Layer 4: 128 -> 1024 (No BN or ReLU)
        encoder_layers.append(
            nn.Linear(hidden_dims[1], hidden_dims[2])
        )

        self.encoder_head = nn.Sequential(*encoder_layers)
        
        # === Define the Regression Head ===
        # This is analogous to the PointNet classifier, but outputs in_dim * in_dim (9) features.
        
        regression_layers = []
        
        # Layer 1: 1024 -> 512
        regression_layers.extend([
            nn.Linear(hidden_dims[2], regression_dims[0]),
            nn.ReLU()
        ])
        
        # Layer 2: 512 -> 256
        regression_layers.extend([
            nn.Linear(regression_dims[0], regression_dims[1]),
            nn.ReLU()
        ])
        
        # Layer 3: 256 -> in_dim * in_dim (9)
        regression_layers.append(
            nn.Linear(regression_dims[1], in_dim * in_dim)
        )
        
        self.regression_head = nn.Sequential(*regression_layers)



    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the T-Net. Compute the transformation matrices, but do not apply them yet.
        The forward pass is the same as that of your original PointNet, except for:
        1) Adding an identity matrix (be sure to set the device to x.device)
        2) Reshaping the output

        Args:
            x: tensor of shape (B, N, in_dim), where B is the batch size, N is the number of points per
               point cloud, and in_dim is the input point dimension

        Output:
        -   transform_matrices: tensor of shape (B, in_dim, in_dim) containing transformation matrices
                       These will be used to transform the point cloud.
        '''

        transform_matrices = None


        # === Encoder ===
        
        # Manually apply layers from self.encoder_head, permuting for BatchNorm
        
        # --- Layer 1 (Linear + BN + ReLU) ---
        pt = self.encoder_head[0](x)   # (B, N, 3) -> (B, N, 64)
        pt = pt.permute(0, 2, 1)       # (B, N, 64) -> (B, 64, N)
        pt = self.encoder_head[1](pt)  # BatchNorm1d(64)
        pt = self.encoder_head[2](pt)  # ReLU
        pt = pt.permute(0, 2, 1)       # (B, 64, N) -> (B, N, 64)

        # --- Layer 2 (Linear + BN + ReLU) ---
        pt = self.encoder_head[3](pt)  # (B, N, 64) -> (B, N, 64)
        pt = pt.permute(0, 2, 1)       # (B, N, 64) -> (B, 64, N)
        pt = self.encoder_head[4](pt)  # BatchNorm1d(64)
        pt = self.encoder_head[5](pt)  # ReLU
        pt = pt.permute(0, 2, 1)       # (B, 64, N) -> (B, N, 64)
        
        # --- Layer 3 (Linear + BN + ReLU) ---
        pt = self.encoder_head[6](pt)  # (B, N, 64) -> (B, N, 128)
        pt = pt.permute(0, 2, 1)       # (B, N, 128) -> (B, 128, N)
        pt = self.encoder_head[7](pt)  # BatchNorm1d(128)
        pt = self.encoder_head[8](pt)  # ReLU
        pt = pt.permute(0, 2, 1)       # (B, 128, N) -> (B, N, 128)

        # --- Layer 4 (Linear only) ---
        encodings = self.encoder_head[9](pt) # (B, N, 128) -> (B, N, 1024)
        
        # === Global Max Pooling ===
        global_feature, _ = torch.max(encodings, dim=1) # (B, 1024)
        
        # === Regression Head ===
        
        # Pass global feature through the regression head
        # output shape: (B, in_dim * in_dim), (B, 9)
        output = self.regression_head(global_feature)
        
        # Create an identity matrix on the same device as x
        # shape: (in_dim, in_dim), (3, 3)
        identity = torch.eye(self.in_dim, device=x.device)
        
        # Reshape output to (B, 3, 3) and add the identity matrix
        transform_matrices = output.view(-1, self.in_dim, self.in_dim) + identity


        return transform_matrices


class PointNetTNet(nn.Module):

    def __init__(
        self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        tnet_hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        tnet_regression_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet with T-Net. The main difference between our
        original PointNet model and this one is the addition of a T-Net to predict
        a transform to apply to the input point cloud.

        Hint:
        1) Think about how to drectly reuse your PointNet implementation from earlier

        Args:
        -   classes: Number of output classes
        -   hidden_dims: The dimensions of the encoding MLPs.
        -   classifier_dims: The dimensions of classifier MLPs.
        -   tnet_hidden_dims: The dimensions of the encoding MLPs for T-Net
        -   tnet_regression_dims: The dimensions of the regression MLPs for T-Net
        -   pts_per_obj: The number of points that each point cloud is padded to
        '''
        super().__init__()

        self.tnet = None
        self.point_net = None


        # Initialize the T-Net
        self.tnet = TNet(
            in_dim=in_dim,
            hidden_dims=tnet_hidden_dims,
            regression_dims=tnet_regression_dims,
            pts_per_obj=pts_per_obj
        )
        
        # Initialize the original PointNet in part 3
        self.point_net = PointNet(
            classes=classes,
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            classifier_dims=classifier_dims,
            pts_per_obj=pts_per_obj
        )



    def apply_tnet(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the transformation matrices by passing x into T-Net, and
        compute the transformed points by batch matrix multiplying x by the
        transformation matrices.

        Hint: Use torch.bmm for batched matrix multiplication. Multiply x by
        the transformation matrix rather than the other way around.

        Args:
        -   x: tensor of shape (B, pts_per_obj, 3), where B is the batch size and
               pts_per_obj is the number of points per point cloud

        Outputs:
        -   x_transformed: tensor of shape (B, pts_per_obj, 3) containing the
                           transformed point clouds per object.
        '''
        x_transformed = None


        # Get the transformation matrices from the T-Net
        # x shape: (B, N, 3); transform_matrices shape: (B, 3, 3)
        transform_matrices = self.tnet(x)
        
        # Apply the transformation using batched matrix multiplication
        # (B, N, 3) bmm (B, 3, 3) -> (B, N, 3)
        x_transformed = torch.bmm(x, transform_matrices)


        return x_transformed


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.

        Hint:
        1) Apply the T-Net transforms via apply_tnet
        2) Use your original PointNet architecture on the transformed pointcloud

        Args:
        -   x: tensor of shape (B, pts_per_obj, 3), where B is the batch size and
               pts_per_obj is the number of points per point cloud

        Outputs:
        -   class_outputs: tensor of shape (B, classes) containing raw scores for each class
        -   encodings: tensor of shape (B, N, hidden_dims[-1]), the final vector for each input point
                       before global maximization. This will be used later for analysis.
        '''
        class_outputs = None
        encodings = None


        # Apply the T-Net transformation to the input points
        x_transformed = self.apply_tnet(x)
        
        # Pass the transformed points through the original PointNet
        class_outputs, encodings = self.point_net(x_transformed)


        return class_outputs, encodings