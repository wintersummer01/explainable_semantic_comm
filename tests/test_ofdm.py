import numpy as np

from prototype.ofdm import ResourceGrid, OFDMDemodulatorThread, OFDMModulatorThread


def test_noiseless_ofdm_roundtrip():
    rng = np.random.default_rng(7)
    grid = ResourceGrid(1.4, N_ports=1, power=1 / 8)
    symbols = (
        rng.standard_normal(grid.num_data_symbols)
        + 1j * rng.standard_normal(grid.num_data_symbols)
    ).astype(np.complex64)
    frame = next(OFDMModulatorThread(grid).process(symbols))
    recovered = OFDMDemodulatorThread(grid, symbols_per_gop=symbols.size).process(frame)
    np.testing.assert_allclose(recovered, symbols, rtol=2e-5, atol=2e-5)
