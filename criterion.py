import torch
from torch import nn


PROBE_KEYS = ("l_1", "l_2", "l_3_0", "l_3_1", "l_3_2", "l_4", "l_y")


class SemComLoss(nn.Module):
    def __init__(self, loss_type="MSE"):
        super().__init__()
        self.loss_type = loss_type
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        image, label = target if isinstance(target, (tuple, list)) else (target, None)
        result = {}
        if "x_hat" in output:
            result["MSE"] = self.mse(output["x_hat"], image)
            result["PSNR"] = 10 * torch.log10(result["MSE"].reciprocal())

        if self.loss_type == "layer_CE":
            if label is None:
                raise ValueError("layer_CE requires labeled ImageFolder batches")
            for key in PROBE_KEYS:
                result[f"CE_{key[2:]}"] = self.cross_entropy(output[key], label)
            result["loss"] = sum(value for key, value in result.items() if key.startswith("CE_"))
        elif self.loss_type == "MSE":
            result["loss"] = result["MSE"]
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return result
