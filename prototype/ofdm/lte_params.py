

# USRP setup =====================================
carrier_frequency = 2.2e9
Tx_gain, Rx_gain = 10, 20

# LTE bandwidth setup ============================
bandwidth_options = {
#   BW,     #SC,    FFT_SIZE 
    1.4:    (72,    128),
    3:      (180,   256),
    5:      (300,   512),
    10:     (600,   1024),
    15:     (900,   1536),
    20:     (1200,  2048)
}

# Baseband parameters  ================================
modulation_order = 16
POWER = 1/8 # Digital domain power control to obey signal range limit [-1, 1]

# Number of ports
N_PORTS = 2
assert 12 % N_PORTS == 0, "Currently N_PORTS should meet 12 % N_PORTS == 0"

#%%
# LTE parameters =====================================
delta_f = 15e3  # Subcarrier spacing in Hz

normal_CP_time = 4.7e-6 # Normal CP: 4.7  us
first_CP_time = 5.2e-6 # First CP: 5.2 us

N_PSS = 62 # Number of subcarrier for PSS
num_symbols_per_slot = 7  # 7 OFDM per single slot
num_slot_per_subframe = 2 # 2 slot per subframe
num_subframe_per_frame = 10   # 10 subframe per frame
num_symbols_frame = num_symbols_per_slot * \
                    num_slot_per_subframe * \
                    num_subframe_per_frame

SSS_I_FRAME = 13
SSS_P_FRAME = 19
SSS_N_FRAME = 37

# Some computed values
T = 1 / delta_f # OFDM symbol time without CP in seconds


PDSCH_PLACEHOLDER = 999 # temporary number to indicate PDSCH symbols
ZERO_FILLED = 0