import copy
import time
import math
import numpy as np
import scipy
from einops import rearrange

from prototype.queue_block import CommThread
from prototype.ofdm.lte_params import (
    bandwidth_options, delta_f, normal_CP_time, first_CP_time,
    num_symbols_per_slot, num_slot_per_subframe, num_subframe_per_frame,
    N_PSS, PDSCH_PLACEHOLDER, ZERO_FILLED, 
    SSS_I_FRAME, SSS_P_FRAME, SSS_N_FRAME
)
from prototype.ofdm.lte_utils import get_zc_sequence, generate_sss

class ResourceGrid:
    def __init__(self, bandwidth, N_ports, power = 1):
        # Parameter Settings ======================================
        self.SC, self.FFT_SIZE = bandwidth_options[bandwidth]
        self.N_ports = N_ports
        self.power = power
        self.sampling_rate = delta_f * self.FFT_SIZE
        self.normal_CP_length = round(normal_CP_time * self.sampling_rate)
        self.first_CP_length = round(first_CP_time * self.sampling_rate)
        self.slot_length = (
            self.FFT_SIZE * num_symbols_per_slot + \
            self.normal_CP_length * (num_symbols_per_slot - 1) + \
            self.first_CP_length
        )
        self.frame_length = self.slot_length * num_slot_per_subframe * num_subframe_per_frame
        self._sss_idx = 0
        
        # Resource Grid Pre-generation =============================
        self.maps = np.ones(
            (num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, self.SC),
            dtype=np.complex64
        ) * PDSCH_PLACEHOLDER 
        self.maps = np.repeat(np.expand_dims(self.maps, axis=0), self.N_ports, axis=0)
        self._set_pss()
        self._set_sss()
        self._set_dmrs()
            
        self.pdsch_idx = np.where(np.transpose(self.maps, (0, 4, 1, 2, 3)) == PDSCH_PLACEHOLDER)
        self.pdsch_idx = (self.pdsch_idx[0], self.pdsch_idx[2], self.pdsch_idx[3], self.pdsch_idx[4], self.pdsch_idx[1]) # For sequential mapping
        self.num_data_symbols = len(self.pdsch_idx[0])
        
        # empty_waveform (For streaming) ================================
        self.empty_waveform = self._proc_frame(ZERO_FILLED)
        

    def _set_pss(self):
        zc_seq = get_zc_sequence(N_PSS, 25)
        self.pss_start_idx = self.SC - (N_PSS+1) - ((self.SC - (N_PSS+1))//2)
        self.pss = np.zeros(self.SC, np.complex64)
        self.pss[self.pss_start_idx:self.pss_start_idx+N_PSS] = zc_seq
        
        self.td_pss_with_cp = self._proc_td(self.pss)
        
        # for sfn in [0, 5]: # PSS is mapped at slot 0, symbol 6 of subframe 0, 5
        for sfn in [0]: # PSS is mapped at slot 0, symbol 6 of subframe 0, 5
            for antenna in range(self.N_ports):
                self.maps[antenna, sfn, 0, 6, :] = self.pss
        
    def _set_sss(self):
        self.sss = {}
        for SSS_IDX in [SSS_I_FRAME, SSS_P_FRAME, SSS_N_FRAME]:
            sss = {}
            sss['first_iq'], sss['second_iq'] = generate_sss(SSS_IDX, SSS_IDX + 4)
            sss['first'] = np.zeros(self.SC, np.complex64)
            sss['first'][self.pss_start_idx:self.pss_start_idx+N_PSS] = sss['first_iq']
            sss['second'] = np.zeros(self.SC, np.complex64)
            sss['second'][self.pss_start_idx:self.pss_start_idx+N_PSS] = sss['second_iq']
            self.sss[SSS_IDX] = sss
        self.frame_type = SSS_N_FRAME

    def _set_dmrs(self):        
        self.dmrs = get_zc_sequence(self.SC//self.N_ports, 1)
        dmrs_stacked = np.tile(
            np.reshape(self.dmrs, (1, 1, self.SC//self.N_ports)),
            [num_subframe_per_frame, num_slot_per_subframe, 1]
        )
        sn = 0 # DMRS is mapped every symbol 0 in a slot
        for antenna in range(self.N_ports):
            self.maps[antenna, ..., sn, :] = ZERO_FILLED
            self.maps[antenna, ..., sn, antenna::self.N_ports] = dmrs_stacked
        
    def _proc_frame(self, symbols):
        assert type(symbols) in [int, float] or symbols.size == self.num_data_symbols, "Symbol shape should be pdsch size!"
        
        # Frame Generation ==================================
        data_map = copy.deepcopy(self.maps)
        data_map[self.pdsch_idx] = symbols
        
    
        # OFDM signal processing 
        fft_symbols = np.zeros(
            (self.N_ports, num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, self.FFT_SIZE),
            np.complex64
        )
        # insert zero in the middle (do not use DC area)
        fft_symbols[..., 1:self.SC//2+1] = data_map[..., self.SC//2:] # right half is put in the positive area
        fft_symbols[..., -(self.SC//2):] = data_map[..., :self.SC//2] # left half is put in the negative area

        # ifft and CP insertion
        td_symbols = np.fft.ifft(fft_symbols, axis=-1, norm='ortho').astype(np.complex64)
        td_symbols_with_cp_0th = np.concatenate(
            [
                td_symbols[..., 0:1, -self.first_CP_length:],
                td_symbols[..., 0:1, :],
            ],
            axis=-1
        )
        td_symbols_with_cp_otherwise = np.concatenate(
            [
                td_symbols[..., 1:, -self.normal_CP_length:],
                td_symbols[..., 1:, :],
            ],
            axis=-1
        )
        td_symbols_with_cp = np.concatenate(
            [
                np.reshape(td_symbols_with_cp_0th, (self.N_ports, num_subframe_per_frame, num_slot_per_subframe, -1)),
                np.reshape(td_symbols_with_cp_otherwise, (self.N_ports, num_subframe_per_frame, num_slot_per_subframe, -1)),
            ],
            axis=-1
        )

        # Digital power normalization
        td_symbols_with_cp *= np.sqrt(self.power)
        return np.reshape(td_symbols_with_cp, (self.N_ports, -1))

    def _proc_td(self, symbols):
        fd_sym = np.zeros(self.FFT_SIZE, np.complex64)
        fd_sym[1:self.SC//2+1] = symbols[self.SC//2:] # right half is put in the positive area
        fd_sym[-(self.SC//2):] = symbols[:self.SC//2] # left half is put in the negative area
        td_sym = np.fft.ifft(fd_sym, axis=-1, norm='ortho').astype(np.complex64)
        td_sym = np.concatenate(
            [
                td_sym[-self.normal_CP_length:],
                td_sym,
            ],
            axis=-1
        )
        return td_sym
        

    @property
    def frame_type(self):
        return self._sss_idx

    @frame_type.setter
    def frame_type(self, idx):
        if self._sss_idx != idx:
            assert idx in [SSS_I_FRAME, SSS_P_FRAME, SSS_N_FRAME], f"Not a normal SSS idx! got: {idx}"
            # SSS
            for antenna in range(self.N_ports):
                # self.maps[antenna, 0, 0, 5, :] = self.sss[idx]['first']
                # self.maps[antenna, 5, 0, 5, :] = self.sss[idx]['second']
                self.maps[antenna, 0, 0, 5, :] = ZERO_FILLED
                self.maps[antenna, 5, 0, 5, :] = ZERO_FILLED
                
            # PSS
            if idx == SSS_I_FRAME:
                for sfn in [0]: 
                    for antenna in range(self.N_ports):
                        self.maps[antenna, sfn, 0, 6, :] = self.pss
            else:
                for sfn in [0]: 
                    for antenna in range(self.N_ports):
                        self.maps[antenna, sfn, 0, 6, :] = ZERO_FILLED
                    
            self._sss_idx = idx
    
            
    

class OFDMModulatorThread(CommThread):
    def __init__(self, rg: ResourceGrid, **kwargs):
        super().__init__(**kwargs)
        self.rg = rg
        self.data_per_frame = self.rg.num_data_symbols
    
    def run(self):
        while True:
            item = self.in_queue.get()
            waveforms = np.zeros((self.rg.N_ports, 0), dtype=np.complex64)
            for output in self.process(item):
                waveforms = np.concatenate((waveforms, output), axis=-1)
            self.out_queue.put(waveforms)
        
    def process(self, symbols):
        num_frames = math.ceil(symbols.size / self.data_per_frame)
        end_padding = np.zeros(num_frames*self.data_per_frame - symbols.size)
        symbol_list = np.array_split(np.concatenate((symbols, end_padding)), num_frames)
        SSS_TYPEs = [SSS_P_FRAME] * num_frames
        SSS_TYPEs[0] = SSS_I_FRAME
        
        for symbol, SSS_TYPE in zip(symbol_list, SSS_TYPEs):
            self.rg.frame_type = SSS_TYPE
            waveform = self.rg._proc_frame(symbol)
            yield waveform


class OFDMSynchronizerThread(CommThread):
    def __init__(self, rg: ResourceGrid, symbols_per_gop=16384, **kwargs):
        super().__init__(**kwargs)
        self.rg = rg
        self.N_Rx = rg.N_ports
        self.FFT_SIZE = rg.FFT_SIZE
        self.first_CP_length = rg.first_CP_length
        self.normal_CP_length = rg.normal_CP_length
        self.frame_length = rg.frame_length
        self.HALF = self.frame_length // 2
        self.pss_ref = rg.td_pss_with_cp
        self.sampling_rate = rg.sampling_rate
        self.slot_length = rg.slot_length
        self.recv_buffer = np.empty((self.N_Rx, 0))
        self.num_frames = math.ceil(symbols_per_gop / rg.num_data_symbols)
        self.stack_size = 0
        self.detect_thresh = 1000
        
    def run(self):
        # Discard first 10 frames
        self._gather_recv(self.frame_length*10)

        # Detect noise level (Assume no Tx)
        noise_frame = self._gather_recv(self.frame_length*10)
        self.noise_level = np.mean(np.abs(noise_frame)**2)
        self.noise_level = 1.9746484554225267e-08
        print(f"Set noise level to :{self.noise_level}")

        while True:
            rcv_frame = self._gather_recv(self.frame_length)
            detected = self._detect_power(rcv_frame)
            if not detected:
                continue

            second_frame = self._gather_recv(self.frame_length)
            double_frame = np.concatenate((rcv_frame, second_frame), axis=-1)
            sync_idx = self._pss_sync(double_frame.copy())

            if sync_idx > self.frame_length:
                more_frame = self._gather_recv(sync_idx)
                double_frame = np.concatenate((double_frame[..., sync_idx:], more_frame), axis=-1)
                sync_idx = 0
            
            frame1 = double_frame[..., sync_idx:sync_idx+self.frame_length].copy()
            frames = [self.process(frame1)]

            if self.num_frames > 1:
                more_frame = self._gather_recv(sync_idx)
                frame2 = np.concatenate((double_frame[..., sync_idx+self.frame_length:], more_frame), axis=-1)
                frames.append(self.process(frame2))

            if self.num_frames > 2:
                for _ in range(self.num_frames - 2):
                    frame = self._gather_recv(self.frame_length)
                    frames.append(self.process(frame))

            self.out_queue.put(frames)


    def _detect_power(self, rcv_frame):
        rcv_power = np.abs(rcv_frame - np.mean(rcv_frame, axis=-1))**2
        if np.max(rcv_power) > self.noise_level * self.detect_thresh:
            print(np.max(rcv_power) / self.noise_level)
            return True
        else:
            return False

    def _gather_recv(self, num):
        while self.recv_buffer.shape[1] < num:
            item = self.in_queue.get()
            self.recv_buffer = np.concatenate((self.recv_buffer, item), axis=-1)
        
        output = self.recv_buffer[..., :num]
        self.recv_buffer = self.recv_buffer[..., num:]

        output -= np.mean(output, axis=-1)
        return output
    
    def _pss_sync(self, frame_rcv):
        # Timing synchronization ================================
        frame_rcv_ifosync_mean = frame_rcv[0]
        # Average every antenna reception to compute correlation
        corr = np.abs(scipy.signal.correlate(frame_rcv_ifosync_mean, self.pss_ref, 'valid', 'fft'))
        # Scipy corr w/ FFT is much much faster!

        ss_start_idx = self.FFT_SIZE * 6 + self.first_CP_length + self.normal_CP_length * 5
        sync_idx = np.argmax(corr) - ss_start_idx

        return sync_idx
    
    def _sss_sync(self, frame_rcv):
        td_sss_idx = self.FFT_SIZE * 5 + self.first_CP_length + self.normal_CP_length * 5
        rcv_sss_first = frame_rcv[0, td_sss_idx:td_sss_idx+self.FFT_SIZE]
        rcv_sss_first = np.fft.fft(rcv_sss_first, norm='ortho')
        rcv_sss_f_iq = np.zeros(N_PSS, dtype=np.complex64)
        rcv_sss_f_iq[N_PSS//2:] = rcv_sss_first[1:N_PSS//2+1]
        rcv_sss_f_iq[:N_PSS//2] = rcv_sss_first[-(N_PSS//2):]

        best_corr = 0
        for SSS_IDX in [SSS_I_FRAME, SSS_P_FRAME, SSS_N_FRAME]:
            for flip, key in zip((False, True), ('first_iq', 'second_iq')):
                ref_sss = self.rg.sss[SSS_IDX][key]
                corr = np.abs(np.mean(rcv_sss_f_iq * ref_sss))
                if best_corr < corr:
                    best_corr = corr
                    best_sss = SSS_IDX
                    best_flp = flip
        
        return best_sss, best_flp
        
        
    def process(self, frame_rcv_timesync):
        # Fractional frequency offset synchronization ==============
        # Done by CP correlation
        # Note: as the offset between two identical sequences become longer, the range of detectable CFO decreases (due to phase ambiguity)
        frame_rcv_timesync /= np.sqrt(self.rg.power)
        td_symbols_rcv = np.reshape(frame_rcv_timesync, (self.N_Rx, num_subframe_per_frame * num_slot_per_subframe, -1))

        first_CPs = td_symbols_rcv[..., :self.first_CP_length]
        first_signal_for_CPs = td_symbols_rcv[..., self.FFT_SIZE:self.FFT_SIZE+self.first_CP_length]

        td_symbols_rcv_wo_first = td_symbols_rcv[:, :, self.FFT_SIZE+self.first_CP_length:]
        td_symbols_rcv_wo_first = np.reshape(td_symbols_rcv_wo_first, (self.N_Rx, num_subframe_per_frame * num_slot_per_subframe * (num_symbols_per_slot - 1), -1))
        other_CPs = td_symbols_rcv_wo_first[..., :self.normal_CP_length]
        other_signal_for_CPs = td_symbols_rcv_wo_first[..., -self.normal_CP_length:]

        phase_diff = np.angle(
            np.sum(np.conj(other_CPs) * other_signal_for_CPs) + \
            np.sum(np.conj(first_CPs) * first_signal_for_CPs)
        )

        sample_idx_diff = self.FFT_SIZE
        ffo = phase_diff * self.sampling_rate / (2 * np.pi * sample_idx_diff)

        n = np.arange(frame_rcv_timesync.shape[-1])
        compensation_ifo = np.exp(-1j * ffo * n / self.sampling_rate * 2 * np.pi).astype(np.complex64)
        frame_rcv_ffosync = frame_rcv_timesync * np.reshape(compensation_ifo, (1, -1))

        # Residual frequency offset estimation  =====================
        # with DMRS
        dmrs_rcv = np.reshape(frame_rcv_ffosync, (self.N_Rx, num_subframe_per_frame * num_slot_per_subframe, -1))
        dmrs_rcv = dmrs_rcv[..., self.first_CP_length:self.first_CP_length+self.FFT_SIZE] # DMRS in the first symbol

        phase_diff = np.angle(
            np.sum(np.sum(np.conj(dmrs_rcv[:, :-1, :]) * dmrs_rcv[:, 1:, :], axis=-1), axis=0)
        )
        sample_idx_diff = self.slot_length
        rfo = phase_diff * self.sampling_rate / (2 * np.pi * sample_idx_diff)
        rfo = np.concatenate([rfo, rfo[-1:]], axis=0)
        rfo = np.reshape(rfo, (1, -1, 1))

        n = np.reshape(np.arange(self.slot_length), (1, 1, -1))
        compensation_rfo = np.exp(-1j * rfo * n / self.sampling_rate * 2 * np.pi).astype(np.complex64)
        frame_rcv_rfosync = frame_rcv_ffosync * np.reshape(compensation_rfo, (1, -1))

        return frame_rcv_rfosync



class OFDMSynchronizerShotThread(OFDMSynchronizerThread):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def run(self):
        while True:
            rcv_frame = self.in_queue.get()
            sync_idx = self._pss_sync(rcv_frame.copy())
            if sync_idx + self.frame_length*self.num_frames > rcv_frame.shape[1]:
                print("Sync Fail!")
                continue

            frames = []
            for i in range(self.num_frames):
                frame = rcv_frame[..., sync_idx+i*self.frame_length:sync_idx+(i+1)*self.frame_length].copy()
                frames.append(frame)
                
            self.out_queue.put(frames)


class OFDMDemodulatorThread(CommThread):
    def __init__(self, rg: ResourceGrid, symbols_per_gop=16384, **kwargs):
        super().__init__(**kwargs)
        self.rg = rg
        self.N_Rx = rg.N_ports
        self.first_CP_length = rg.first_CP_length
        self.normal_CP_length = rg.normal_CP_length
        self.FFT_SIZE = rg.FFT_SIZE
        self.pdsch_idx = rg.pdsch_idx
        self.symbols_per_gop = symbols_per_gop
        
    def run(self):
        while True:
            item = self.in_queue.get()
            gop_symbols = []
            for rcv_frame in item:
                iq_rcv = self.process(rcv_frame)
                gop_symbols.append(iq_rcv)
            gop_symbols = np.concatenate(gop_symbols)
            gop_symbols = gop_symbols[:self.symbols_per_gop]
            self.out_queue.put(gop_symbols)
        
    def process(self, frame_rcv_rfosync):
        # Frequency domain demodulation =============================
        # CP removal and ifft
        fft_symbols_rcv = np.reshape(frame_rcv_rfosync, (self.N_Rx, num_subframe_per_frame * num_slot_per_subframe, -1))
        fft_symbols_rcv = fft_symbols_rcv[..., self.first_CP_length-self.normal_CP_length:]
        fft_symbols_rcv = np.reshape(fft_symbols_rcv, (self.N_Rx, num_subframe_per_frame * num_slot_per_subframe * num_symbols_per_slot, -1))
        fft_symbols_rcv = fft_symbols_rcv[..., self.normal_CP_length:]

        fft_symbols_rcv = np.fft.fft(fft_symbols_rcv, axis=-1, norm='ortho')
        fft_symbols_rcv = np.reshape(
            fft_symbols_rcv,
            (self.N_Rx, num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, self.FFT_SIZE)
        )

        resource_maps_rcv = np.zeros(
            (self.N_Rx, num_subframe_per_frame, num_slot_per_subframe, num_symbols_per_slot, self.rg.SC),
            dtype=np.complex64
        )

        resource_maps_rcv[..., self.rg.SC//2:] = fft_symbols_rcv[..., 1:self.rg.SC//2+1]
        resource_maps_rcv[..., :self.rg.SC//2] = fft_symbols_rcv[..., -(self.rg.SC//2):]

        # PDSCH decode ==============================================
        # Channel estimation
        dmrs_rcv = resource_maps_rcv[..., 0, :]

        h = np.zeros((num_subframe_per_frame, num_slot_per_subframe, self.rg.SC//self.N_Rx, self.N_Rx, self.N_Rx), dtype=np.complex64)
        for rx in range(self.N_Rx):
            for tx in range(self.N_Rx):
                h[..., rx, tx] = dmrs_rcv[rx, :, :, tx::self.N_Rx] / np.reshape(self.rg.dmrs, (1, 1, -1))

        # Nearest neighbor channel interpolation (in subcarrier domain)
        h = np.repeat(h, self.N_Rx, axis=2)
        # shape: (num_subframe_per_frame, num_slot_per_subframe, num_subcarriers, N_RX, N_TX)

        # ZF combining
        h_inv = np.linalg.inv(h)

        # Matrix multiplication
        resource_maps_rcv_zf = np.einsum('abcij,jabdc->iabdc', h_inv, resource_maps_rcv)

        # PDSCH extraction
        iq_rcv = resource_maps_rcv_zf[self.pdsch_idx]
        
        return iq_rcv
            
            
