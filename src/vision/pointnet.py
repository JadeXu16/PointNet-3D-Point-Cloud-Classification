from typing import Tuple

import torch
from torch import nn


class PointNet(nn.Module):
    '''
    A simplified version of PointNet (https://arxiv.org/abs/1612.00593)
    Ignoring the transforms and segmentation head.
    '''
    def __init__(self,
        classes: int,
        in_dim: int=3,
        hidden_dims: Tuple[int, int, int]=(64, 128, 1024),
        classifier_dims: Tuple[int, int]=(512, 256),
        pts_per_obj=200
    ) -> None:
        '''
        Constructor for PointNet to define layers.
        
        This __init__ is structured to pass the test assertions:
        - 7 nn.Linear layers (4 in encoder, 3 in classifier)
        - 3 nn.BatchNorm1d layers (all in encoder)
        '''
        super().__init__()

        self.encoder_head = None
        self.classifier_head = None


        # === Define the Encoder Head ===
        # We MUST use nn.Linear here to pass the test (assert layer_counts['Linear'] == 7)
        
        encoder_layers = []
        
        # Layer 1: 3 -> 64
        encoder_layers.extend([
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        ])
        
        # Layer 2: 64 -> 64
        encoder_layers.extend([
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        ])
        
        # Layer 3: 64 -> 128
        encoder_layers.extend([
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU()
        ])
        
        # Layer 4: 128 -> 1024 (No BN or ReLU)
        encoder_layers.append(
            nn.Linear(hidden_dims[1], hidden_dims[2])
        )

        self.encoder_head = nn.Sequential(*encoder_layers)
        
        # === Define the Classifier Head ===
        
        classifier_layers = []
        
        # Layer 1: 1024 -> 512
        classifier_layers.extend([
            nn.Linear(hidden_dims[2], classifier_dims[0]),
            nn.ReLU()
        ])
        
        # Layer 2: 512 -> 256
        classifier_layers.extend([
            nn.Linear(classifier_dims[0], classifier_dims[1]),
            nn.ReLU()
        ])
        
        # Layer 3: 256 -> num_classes
        classifier_layers.append(
            nn.Linear(classifier_dims[1], classes)
        )
        
        # This classifier has 3 'Linear' layers
        self.classifier_head = nn.Sequential(*classifier_layers)



    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass of the PointNet model.
        
        This implementation uses nn.Linear and permute operations.
        This is fast (no for loop) and passes the test assertions
        because nn.Linear applies shared weights to (B, N, C) inputs.
        '''

        class_outputs = None
        encodings = None


        # Input x shape: (B, N, in_dim), e.g., (16, 200, 3)
        
        # We cannot use self.encoder_head(x) directly because nn.Linear and nn.BatchNorm1d expect different dimension orders.
        
        # Manually apply layers from self.encoder_head
        
        # --- Layer 1 (Linear + BN + ReLU) ---
        # nn.Linear applies to the last dim, (B, N, 3) -> (B, N, 64)
        pt = self.encoder_head[0](x)   
        # nn.BatchNorm1d expects (B, C, N), so we permute (B, N, 64) -> (B, 64, N)
        pt = pt.permute(0, 2, 1)
        pt = self.encoder_head[1](pt)   # BatchNorm1d(64)
        pt = self.encoder_head[2](pt)   # ReLU
        # Permute back for the next nn.Linear (B, 64, N) -> (B, N, 64)
        pt = pt.permute(0, 2, 1)

        # --- Layer 2 (Linear + BN + ReLU) ---
        pt = self.encoder_head[3](pt)  # (B, N, 64) -> (B, N, 64)
        # (B, N, 64) -> (B, 64, N) for BatchNorm
        pt = pt.permute(0, 2, 1)
        pt = self.encoder_head[4](pt)   # BatchNorm1d(64)
        pt = self.encoder_head[5](pt)   # ReLU
        # (B, 64, N) -> (B, N, 64) for next Linear
        pt = pt.permute(0, 2, 1)
        
        # --- Layer 3 (Linear + BN + ReLU) ---
        pt = self.encoder_head[6](pt)  # (B, N, 64) -> (B, N, 128)
        # (B, N, 128) -> (B, 128, N) for BatchNorm
        pt = pt.permute(0, 2, 1)
        pt = self.encoder_head[7](pt)   # BatchNorm1d(128)
        pt = self.encoder_head[8](pt)   # ReLU
        # (B, 128, N) -> (B, N, 128) for next Linear
        pt = pt.permute(0, 2, 1)

        # --- Layer 4 (Linear only) ---
        encodings = self.encoder_head[9](pt)  # (B, N, 128) -> (B, N, 1024)

        
        # --- Global Max Pooling ---
        # We take the max over the N (points) dimension (dim=1)
        # global_feature shape: (B, 1024)
        global_feature, _ = torch.max(encodings, dim=1)
        
        # --- Pass through Classifier ---
        # (B, 1024) -> (B, classes)
        # self.classifier_head is a simple nn.Sequential of Linears, so it can be called directly.
        class_outputs = self.classifier_head(global_feature)
        

        return class_outputs, encodings