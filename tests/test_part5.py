from vision.part5_tnet import PointNetTNet, TNet
from tests.model_test_utils import count_params, get_layers, get_model_layer_counts
import torch

def test_tnet_shapes():
    model = TNet()
    x = torch.rand((2, 200, 3))
    matrices = model(x)
    assert matrices.shape == (2, 3, 3)


def test_pointnet_tnet():
    model = PointNetTNet(20)
    params = count_params(model)
    layers = get_layers(model)
    layer_counts = get_model_layer_counts(layers)

    assert params > 1.61e6 and params < 1.62e6
    assert layer_counts['Linear'] == 14
    assert layer_counts['BatchNorm1d'] == 6
    assert layers[-1].out_features == 20
