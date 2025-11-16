# PointNet: 3D Point Cloud Classification

This project is a [PyTorch](https://pytorch.org/) implementation of the [PointNet](https://arxiv.org/abs/1612.00593) architecture, designed for 3D object classification tasks.

The project follows a staged progression, building from a simple baseline to a full PointNet implementation with spatial transformers, and includes utilities for model analysis and visualization.

---

## The Journey in Four Parts

The project follows a staged progression, just like the course brief, with each part feeding into the next.

### Part I – Establishing the Baseline (Data + Voxel Model)
- Implemented the `Argoverse` dataset (`vision.dataloader.Argoverse`) that ingests the TXT point clouds,
  pads them to a fixed length, and exposes consistent train/test splits (0‑169 vs 170‑199).
- Built a voxel-counting baseline (`vision.baseline.Baseline`) that uses `torch.histogramdd` to convert each
  cloud into a 3D occupancy histogram before a single linear classifier.
- **Takeaway:** the baseline provides a fast sanity check (~35–40% validation accuracy) but clearly underfits
  and ignores the fine structure of the point sets.

### Part II – PointNet from First Principles
- Recreated the simplified PointNet encoder/classifier (`vision.pointnet.PointNet`) using only linear layers
  and shared-weight operations; no 1x1 convolutions are hidden from view.
- Implemented the full forward pass manually to reason about dimension ordering, batching, and global max pooling.
- **Takeaway:** this network drastically improves generalization (≈78% validation accuracy) while remaining compact
  enough for classroom training schedules.

### Part III – Going Further with a T-Net
- Added the optional PointNet input transform (`vision.tnet.TNet` + `PointNetTNet`) to learn per-object alignment
  matrices and feed the transformed clouds through the original classifier.
- **Takeaway:** the learned transforms stabilize training on challenging classes, nudging performance past the vanilla
  PointNet while keeping the code modular.

### Part IV – Analysis, Diagnostics, and Visualization
- Authored utilities in `vision.analysis` and `vision.utils` to compute confusion matrices, highlight critical
  points, and export Plotly visualizations (`point_cloud.html`, `point_critical_points.html`).
- These helpers made it easy to reason about misclassifications and to demonstrate what the network “looks at.”

---

## Repository Snapshot

- `src/vision/*.py` – implementations for each part (data loader, baseline, PointNet, T-Net, analysis, training loop)
- `tests/` – pytest suites that mirror the original rubric and guard against regressions
- `output/` – auto-created directory that stores the best checkpoint (`PointNet.pt`, `PointNetTNet.pt`, …)
- `main.ipynb` – experimentation notebook (qualitative analysis, plots, sanity checks)

---

## Performance Summary

| Stage | What Was Implemented | Main Benefit | Typical Outcome |
| :--- | :--- | :--- | :--- |
| I. Data Loader + Baseline | Train/test splits + voxel histogram classifier | Quick end‑to‑end sanity check | ~50% test acc |
| II. PointNet | Shared MLP encoder + global max pooling | Learns permutation‑invariant features | >65% test acc |
| III. PointNet + T-Net | Input alignment module feeding the classifier | Better robustness across classes | >65% test acc |
| IV. Analysis Utilities | Confusion matrices, critical-point plots | Interpretability + debugging | Qualitative insights |

Numbers come from the experiments recorded in `main.ipynb`. They depend on hardware, optimizer settings, and
training time but reflect the relative gains observed throughout the project.

---

## How to Run

1.  **Clone & Download Data**
    ```bash
    git clone [https://github.com/your-username/pointnet-argoverse.git](https://github.com/your-username/pointnet-argoverse.git)
    cd pointnet-argoverse
    
    # Download the Argoverse 1.1 Motion Forecasting Dataset
    # (Provide the specific link or download page here)
    # e.g., Visit: [https://www.argoverse.org/av1.html#download-link](https://www.argoverse.org/av1.html#download-link)
    
    # Place the unzipped point cloud sweeps in the following structure:
    # data/sweeps/<CLASS>/<0-199>.txt
    ```

2.  **Create the Environment**
    ```bash
    cd conda
    
    mamba env create -f environment.yml   # or: conda env create ...
    
    conda activate pointnet_env
    cd ..
    
    # Install the 'vision' package in editable mode
    pip install -e .
    ```

3.  **Run Tests**
    ```bash
    pytest
    ```

4.  **Train a Model**
    
    * **Option 1: (Recommended) Run training script**
        ```bash
        # (This assumes you created a train.py script)
        python train.py --model PointNet --data_root data/sweeps --pad_size 200 --epochs 50
        
        # (If your module is runnable, use -m)
        python -m vision.training.train --model PointNet --data_root data/sweeps --pad_size 200
        ```
    
    * **Option 2: Use in your own script/notebook**
        You can also import and use the models and training loop directly.
        ```python
        from vision.dataloader import Argoverse
        from vision.pointnet import PointNet
        from vision.training import train
        
        train_set = Argoverse(split="train", data_root="data/sweeps", pad_size=200)
        val_set = Argoverse(split="test", data_root="data/sweeps", pad_size=200)
        
        model = PointNet(num_classes=10, pad_size=200)
        train(model, train_set, val_set, epochs=50)
        ```
    * Best checkpoints will be saved to `output/PointNet*.pt`.

5.  **Inspect Results**
    * Generate confusion matrices via `vision.utils.generate_and_plot_confusion_matrix`.
    * Visualize critical points with `vision.utils.plot_crit_points_from_file`.

---

## Acknowledgements & Contributions

This project was originally based on a course assignment. The overall structure, testing rubric,
and data format were provided by the course staff.

My contributions here include:
- Rewriting the README, packaging, and documentation so the assignment can stand alone on GitHub.
- Implementing the dataset loader, baseline, PointNet, PointNet+TNet, and analysis utilities that meet the rubric.
- Adding training/evaluation snippets, visualization helpers, and notebook analyses.
