import torch
from torch import nn


class AWGNChannel(nn.Module):
    '''
    Untrainable SISO Channel for torch
    '''
    def __init__(self, SNRdB):
        super(AWGNChannel, self).__init__()
        self.snr = 10 ** (SNRdB / 10) # dB to linear scale

    def forward(self, x):
        assert x.dtype == torch.complex64, f"input dtype should be complex64. Now is {x.dtype}"
    
        # Noise Generation
        sig_power = torch.mean(torch.abs(x)**2)
        no = torch.sqrt(sig_power/self.snr)
        n = no * torch.randn(x.shape, dtype=torch.complex64, device=x.device)
        
        # Add noise
        y = x + n
        return y


class RicianChannel(nn.Module):
    '''
    Coherent Rician fading channel with zero-forcing equalization.

    Per-symbol fading h ~ CN(nu, 2*sigma^2), normalized so that
    E[|h|^2] = nu^2 + 2*sigma^2 = 1, with K = nu^2 / (2*sigma^2).
    The receiver applies y_hat = (h*x + n) / h, assuming perfect CSI.
    '''
    def __init__(self, K, SNRdB):
        super(RicianChannel, self).__init__()
        self.snr = 10 ** (SNRdB / 10)
        self.K = K
        self.nu = (K / (K + 1)) ** 0.5
        self.sigma = (1 / (2 * (K + 1))) ** 0.5

    def forward(self, x):
        assert x.dtype == torch.complex64, f"input dtype should be complex64. Now is {x.dtype}"

        # Sample Rician fading h (per-symbol), E[|h|^2] = 1
        h_real = self.nu + self.sigma * torch.randn(x.shape, device=x.device)
        h_imag = self.sigma * torch.randn(x.shape, device=x.device)
        h = torch.complex(h_real, h_imag)

        # Faded signal
        hx = h * x

        # AWGN sized from the faded signal power
        sig_power = torch.mean(torch.abs(hx) ** 2)
        no = torch.sqrt(sig_power / self.snr)
        n = no * torch.randn(x.shape, dtype=torch.complex64, device=x.device)

        # Receive + zero-forcing equalization
        y_hat = (hx + n) / h
        return y_hat


class PseudoRicianChannel(nn.Module):
    '''
    Rician-like additive channel
    Based on Gaussian Scale Mixture
    '''
    def __init__(self, K, SNRdB):
        super(PseudoRicianChannel, self).__init__()  # was: super(AWGNChannel, ...) -- bug fix
        self.snr = 10 ** (SNRdB / 10)  # Effective SNR: signal_power / E[|n_eff|^2]
        self.nu = K ** 0.5 

    def forward(self, x):
        assert x.dtype == torch.complex64, f"input dtype should be complex64. Now is {x.dtype}"
        n_shape = (2,) + x.shape
        
        # Sample scale for GSM (channel)
        h_real = self.nu + torch.randn(n_shape, device=x.device)
        h_imag = torch.randn(n_shape, device=x.device)
        no_scale = 1 / torch.sqrt(h_real ** 2 + h_imag ** 2) # refers to no. But, is a tensor (twice of #symbols)
        
        # Sample noise from scale (GSM)
        n_scalar = no_scale * torch.randn(n_shape, device=x.device)
        n_raw = torch.complex(n_scalar[0], n_scalar[1])
        
        # normalize noise to ESNR
        sig_amp = torch.mean(torch.abs(x) ** 2)**0.5
        noise_amp = torch.mean(torch.abs(n_raw) ** 2)**0.5
        n = n_raw * sig_amp / (noise_amp * (self.snr**0.5))   # <-- swapped sig_amp / noise_amp

        # Add noise
        y = x + n
        return y


class SimulationChannel(nn.Module):
    '''
    Channel selector that dispatches to a concrete simulated channel based on
    `channel_type`, so a single model class can serve multiple channel kinds
    instead of one subclass per kind.

    Args:
        channel_type: "AWGN" or "Rician"
        SNRdB: target SNR in dB
        k_factor: Rician K-factor (ignored when channel_type == "AWGN")
    '''
    def __init__(self, channel_type, SNRdB, k_factor=1.0):
        super(SimulationChannel, self).__init__()
        self.channel_type = channel_type
        if channel_type is None or channel_type == "AWGN":
            self.channel = AWGNChannel(SNRdB=SNRdB)
        elif channel_type == "Rician":
            self.channel = RicianChannel(K=k_factor, SNRdB=SNRdB)
        else:
            raise ValueError(
                f"Unknown channel_type: {channel_type!r}. "
                f"Expected one of: 'AWGN', 'Rician'."
            )

    def forward(self, x):
        return self.channel(x)