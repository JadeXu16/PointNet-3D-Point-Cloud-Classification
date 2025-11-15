import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np # Import numpy for data loading

class Argoverse(Dataset):

    def load_path_with_classes(self, split: str, data_root: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Builds (path, class) pairs by pulling all txt files found under
        the data_root directory under the given split. Also builds a list
        of all labels we have provided data for.

        Each of the classes have a total of 200 point clouds numbered from 0 to 199.
        We will be using point clouds 0-169 for the train split and point clouds 
        170-199 for the test split. This gives us a 85/15 train/test split.

        Args:
        -   split: Either train or test. Collects (path, label) pairs for the specified split
        -   data_root: Root directory for training and testing data
        
        Output:
        -   pairs: List of all (path, class) pairs found under data_root for the given split 
        -   class_list: List of all classes present in the dataset *sorted in alphabetical order*
        """

        pairs = []
        class_list = []


        # Define the range of file indices for train/test splits
        if split == 'train':
            file_range = range(0, 170) # 0-169
        else:  # split == 'test':
            file_range = range(170, 200) # 170-199

        # Iterate through each item in the data_root directory
        for class_name in os.listdir(data_root):
            class_dir = os.path.join(data_root, class_name)
            
            # Check if it's a directory
            if os.path.isdir(class_dir):
                class_list.append(class_name)
                
                # Iterate through the file indices for the specified split
                for i in file_range:
                    file_name = f"{i}.txt"
                    file_path = os.path.join(class_dir, file_name)
                    
                    # Add the (path, class_name) tuple to our list
                    if os.path.exists(file_path):
                        pairs.append((file_path, class_name))

        # Sort the class list alphabetically as required
        class_list.sort()


        return pairs, class_list


    def get_class_dict(self, class_list: List[str]) -> Dict[str, int]:
        """
        Creates a mapping from classes to labels. For example, [Animal, Car, Bus],
        would map to {Animal:0, Bus:1, Car:2}. *Note: for consistency, we sort the
        input classes in alphabetical order before creating the mapping (gradescope)
        tests will probably fail if you forget to do so*

        Args:
        -   class_list: List of classes to create mapping from

        Output: 
        -   classes: dictionary containing the class to label mapping
        """

        classes = dict()


        # Use enumerate to get (index, class_name) pairs
        # Create a dictionary mapping class_name -> index
        classes = {class_name: i for i, class_name in enumerate(class_list)}


        return classes
    

    def __init__(self, split: str, data_root: str, pad_size: int) -> None:
        """
        Initializes the dataset. *Hint: Use the functions above*

        Args:
        -   split: Which split to pull data for. Either train or test
        -   data_root: The root of the directory containing all the data
        -   pad_size: The number of points each point cloud should contain when
                      when we access them. This is used in the pad_points function.

        Variables:
        -   self.instances: List of (path, class) pairs
        -   class_dict: Mapping from classes to labels
        -   pad_size: Number of points to pad each point cloud to
        """
        super().__init__()
        
        file_label_pairs, classes = self.load_path_with_classes(split, data_root)
        self.instances = file_label_pairs
        self.class_dict = self.get_class_dict(classes)
        self.pad_size = pad_size


    def get_points_from_file(self, path: str) -> torch.Tensor:
        """
        Returns a tensor containing all of the points in the given file

        Args:
        -   path: Path to the file that we should extract points from

        Output:
        -   pts: A tensor of shape (N, 3) where N is the number of points in the file
        """

        pts = None


        # Load the text file as a numpy array
        # Each line is a point (x, y, z)
        # specify float32 for compatibility with PyTorch
        point_data = np.loadtxt(path, dtype=np.float32, skiprows=1)  # Note: use skiprows=1 to skip the first line (the class name)!
        
        # Convert the numpy array to a PyTorch tensor
        pts = torch.from_numpy(point_data)


        return pts


    def pad_points(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Pads pts to have pad_size points in it. Let p1 be the first point in 
        the tensor. We want to pad pts by adding p1 to the end of pts until 
        it has size (pad_size, 3). 

        Args:
        -   pts: A tensor of shape (N, 3) where N is the number of points in the tensor

        Output: 
        -   pts_full: A tensor of shape (pad_size, 3)
        """

        pts_full = None

        
        num_points = pts.shape[0]
        
        # If the point cloud is already the target size, just return it
        if num_points == self.pad_size:
            return pts
            
        # Calculate how many points we need to add
        num_to_pad = self.pad_size - num_points
        
        # Get the first point (p1)
        # Use pts[0:1, :] to keep its dimension as (1, 3) rather than (3,)
        p1 = pts[0:1, :]
        
        # Create a padding tensor by repeating p1
        padding = p1.repeat(num_to_pad, 1) # Shape will be (num_to_pad, 3)
        
        # Concatenate the original points and the padding
        pts_full = torch.cat((pts, padding), dim=0)

        
        return pts_full


    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the (points, label) pair at the given index.

        Hint: 
        1) get info from self.instances
        2) use get_points_from_file and pad_points

        Args:
        -   i: Index to retrieve

        Output:
        -   pts: Points contained in the file at the given index
        -   label: Tensor containing the label of the point cloud at the given index
        """

        pts = None
        label = None


        # Get the (path, class_name) tuple for the index i
        path, class_name = self.instances[i]
        
        # Get the integer label from the class name
        label_int = self.class_dict[class_name]
        # Convert the integer to a tensor (long is standard for classification)
        label = torch.tensor(label_int, dtype=torch.long)
        
        # 3. Load the points from the file
        pts_raw = self.get_points_from_file(path)
        
        # 4. Pad the points to the required size
        pts = self.pad_points(pts_raw)
        

        return pts, label


    def __len__(self) -> int:
        """
        Returns number of examples in the dataset

        Output: 
        -    l: Length of the dataset
        """
        
        l = None


        # The length of the dataset is the number of (path, class) pairs
        l = len(self.instances)


        return l