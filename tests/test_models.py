import torch

from model import SwinJSCC, SwinJSCC_CNN, SwinJSCC_split, SwinJSCC_split_CNN


def test_swin_forward_shape():
    network = SwinJSCC(
        SNRdB=10,
        spp=0.25,
        img_shape=(32, 32),
        dims=(8, 12, 16, 20),
        patch_shape=(2, 2),
        window_size=2,
        head_dim=4,
    )
    output = network(torch.rand(1, 3, 32, 32))
    assert output["x_hat"].shape == (1, 3, 32, 32)
    assert output["y"].is_complex()


def test_cnn_forward_shape():
    network = SwinJSCC_CNN(SNRdB=10, spp=0.25, img_shape=(32, 32), dim=8)
    output = network(torch.rand(1, 3, 32, 32))
    assert output["x_hat"].shape == (1, 3, 32, 32)
    assert output["y"].is_complex()


def test_split_models_have_thirteen_blocks():
    swin = SwinJSCC_split(
        img_shape=(32, 32),
        dims=(8, 12, 16, 20),
        patch_shape=(2, 2),
        window_size=2,
        head_dim=4,
    )
    cnn = SwinJSCC_split_CNN(img_shape=(32, 32), dim=8)
    assert len(swin.enc_blocks) == len(swin.dec_blocks) == 13
    assert len(cnn.enc_blocks) == len(cnn.dec_blocks) == 13
