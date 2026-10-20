import copy
import numpy as np
import threading
try:
    import uhd
except ImportError:  # UHD is only required for the hardware path.
    uhd = None

from queue import Queue
from prototype.queue_block import CommThread


def _require_uhd():
    if uhd is None:
        raise RuntimeError(
            "UHD Python bindings are required for USRP execution. "
            "Install UHD for your platform or use prototype/run_virtual.py."
        )

class ErrChecker(threading.Thread):
    def __init__(self, tx_stream):
        super().__init__()
        _require_uhd()
        self.tx_stream = tx_stream
        self.async_tx = uhd.types.TXAsyncMetadata()
        self.ERR = False
        
    def run(self):
        while True:
            self.tx_stream.recv_async_msg(self.async_tx)
            if self.async_tx.event_code in [2, 4]:
                self.ERR = True  

class USRPTxThread(threading.Thread):
    def __init__(self, thread_type="USRPTxThread_single", **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.tx_thread = globals()[thread_type]
        
    def run(self):
        while True:
            tx_thread = self.tx_thread(**self.kwargs)
            tx_thread.start()
            tx_thread.join()
            del tx_thread
    
    
class USRPTxThread_single(CommThread):
    def __init__(self, usrp_addr, N_Tx, Fs, Fc, Tx_gain, delay_samples, use_lo_offset=True, **kwargs):
        super().__init__(**kwargs)
        _require_uhd()
        # USRP preparation =========================================================
        self.usrp = uhd.usrp.MultiUSRP(f'addr0={usrp_addr}')
        # Note: X310 does not support set_master_clock_rate and only support change of master clock rate at initialization.

        self.usrp.set_time_now(uhd.types.TimeSpec(0.0))

        if N_Tx == 1:
            self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 0)  # Transmit on daughterboard A of first device
            # self.usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec(""), 0)  # Transmit on daughterboard A of first device
            self.usrp.set_tx_antenna("TX/RX", 0)
            
            print('Tx:')
            print(self.usrp.get_tx_subdev_spec(0))
        else:
            raise AssertionError("MIMO configuration not developed yet")
        
        assert N_Tx == self.usrp.get_tx_num_channels(), "N_PORTS should be equal to USRP TX_CHANNELS"
        
        self.use_lo_offset = use_lo_offset

        lo_offset = Fs / 2 + 1e6 if use_lo_offset else 0
        req = uhd.libpyuhd.types.tune_request(Fc, lo_offset)
        
        self.channels = list(range(N_Tx))
        for c in self.channels:
            self.usrp.set_tx_rate(Fs, c)
            self.usrp.set_tx_freq(req, c)
            self.usrp.set_tx_gain(Tx_gain, c)
        
        self.delay_zeros = np.zeros((len(self.channels), delay_samples), dtype=np.complex64)
    
    
    def _send_usrp(self, waveform, streamer):
        metadata = uhd.types.TXMetadata()
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        metadata.has_time_spec = True
        metadata.time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + 0.05)
        # This time_spec must be filled for multi channel to work well!
        
        i = 0
        while i < waveform.shape[1]:
            i += streamer.send(waveform[:, i:], metadata)
            metadata.start_of_burst = False
            metadata.has_time_spec = False
        # End signal
        metadata.end_of_burst = True
        streamer.send(np.zeros((len(self.channels), 0), dtype=np.complex64), metadata)
    
    
    def run(self):
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = self.channels
        tx_streamer = self.usrp.get_tx_stream(st_args)

        err_checker = ErrChecker(tx_streamer)     
        err_checker.start()  

        self._send_usrp(self.delay_zeros, tx_streamer)
        while True:
            waveform = self.in_queue.get()
            self._send_usrp(waveform, tx_streamer)
                
            if err_checker.ERR:
                print("\nSEQ_ERROR, RESTART!\n")
                # del tx_streamer
                break
            

class USRPTxStreamThread_single(USRPTxThread_single):
    def __init__(self, empty_waveform, **kwargs):
        super().__init__(**kwargs)
        self.empty_waveform = empty_waveform
    
    def run(self):
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = self.channels
        tx_streamer = self.usrp.get_tx_stream(st_args)

        metadata = uhd.types.TXMetadata()
        metadata.has_time_spec = True
        metadata.start_of_burst = True
        metadata.end_of_burst = False
        metadata.time_spec = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + 0.05)
        # This time_spec must be filled for multi channel to work well!

        err_checker = ErrChecker(tx_streamer)     
        err_checker.start()  

        j = 0
        while True:
            if j > 10:
                waveform = self.empty_waveform
                j += 1
            elif self.in_queue.qsize() > 0:
                waveform = self.in_queue.get()
                # print("GET")
            else:
                waveform = self.empty_waveform
                
            i = tx_streamer.send(waveform, metadata)
            metadata.start_of_burst = False
            metadata.has_time_spec = False
            
            while i < waveform.shape[1]:
                i += tx_streamer.send(waveform[:, i:], metadata)
                
            if err_checker.ERR:
                print("\nSEQ_ERROR, RESTART!\n")
                # del tx_streamer
                break



class USRPRxThread(CommThread):
    def __init__(self, usrp_addr, N_Rx, Fs, Fc, Rx_gain, use_lo_offset=True, **kwargs):
        super().__init__(**kwargs)
        _require_uhd()
        # USRP preparation =========================================================
        self.usrp = uhd.usrp.MultiUSRP(f'addr0={usrp_addr}')
        # Note: X310 does not support set_master_clock_rate and only support change of master clock rate at initialization.

        # .usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
        print("USRP loaded. Session Ready.")

        if N_Rx == 1:
            # self.usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec(""), 0)  # Transmit on daughterboard A of first device
            self.usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0"), 0)  # Receive on daughterboard A of second device
            self.usrp.set_rx_antenna("RX2", 0)
            
            print('Rx:')
            print(self.usrp.get_rx_subdev_spec(0))
        else:
            raise AssertionError("MIMO configuration not developed yet")

        assert N_Rx == self.usrp.get_rx_num_channels(), "N_PORTS should be equal to USRP TX_CHANNELS"
        
        lo_offset = Fs / 2 + 1e6 if use_lo_offset else 0
        req = uhd.libpyuhd.types.tune_request(Fc, lo_offset)
        self.channels = list(range(N_Rx))
        for c in self.channels:
            self.usrp.set_rx_rate(Fs, c)
            self.usrp.set_rx_freq(req, c)
            self.usrp.set_rx_gain(Rx_gain, c)
            # self.usrp.set_rx_dc_offset(False, c) # X410 does not support this!
            
    
    def run(self):
        st_args = uhd.usrp.StreamArgs("fc32", "sc16")
        st_args.channels = self.channels
        rx_streamer = self.usrp.get_rx_stream(st_args)
        
        # Get timing sync
        INIT_DELAY = 0.1
        transmission_time = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + INIT_DELAY)

        # Start Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = False
        stream_cmd.time_spec = transmission_time

        metadata = uhd.types.RXMetadata()
        num_channels = len(self.channels)
        num_samples_per_rcv = rx_streamer.get_max_num_samps()
        recv_buffer = np.zeros((num_channels, num_samples_per_rcv), dtype=np.complex64)
        
        rx_streamer.issue_stream_cmd(stream_cmd)
        while True:
            rx_streamer.recv(recv_buffer, metadata)
            recv = copy.deepcopy(recv_buffer)
            self.out_queue.put(recv)
            
            

class USRPRxShotThread(USRPRxThread):
    def __init__(self, rcv_samples, Q_run: Queue, **kwargs):
        super().__init__(**kwargs)
        self.rcv_samples = rcv_samples
        self.initial_clip = 1000
        self.Q_run = Q_run
            

    def run(self):
        while True:
            if self.Q_run.qsize() >= 1:
                st_args = uhd.usrp.StreamArgs("fc32", "sc16")
                st_args.channels = self.channels
                rx_streamer = self.usrp.get_rx_stream(st_args)
                
                # Get timing sync
                INIT_DELAY = 0.1
                transmission_time = uhd.types.TimeSpec(self.usrp.get_time_now().get_real_secs() + INIT_DELAY)

                # Start Stream
                stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
                stream_cmd.stream_now = False
                stream_cmd.time_spec = transmission_time
                rx_streamer.issue_stream_cmd(stream_cmd)

                metadata = uhd.types.RXMetadata()

                num_channels = len(self.channels)
                num_samples_per_rcv = rx_streamer.get_max_num_samps()
                recv_buffer = np.zeros((num_channels, num_samples_per_rcv), dtype=np.complex64)
                    
                    
                rx_streamer.issue_stream_cmd(stream_cmd)
                total_samples = self.initial_clip+self.rcv_samples*2
                samples = np.zeros((num_channels, total_samples), dtype=np.complex64)

                head, tail = 0, 0
                try:
                    while tail < total_samples:
                        tail += rx_streamer.recv(recv_buffer, metadata)
                        tail = min(tail, total_samples)
                        samples[:, head:tail] = recv_buffer[:, :tail-head]
                        head = tail
                        # print(f'\r{head}/{num_samples}', end='')

                        if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                            # print(metadata.error_code)
                            if metadata.error_code != uhd.types.RXMetadataErrorCode.timeout:
                                break
                except RuntimeError as ex:
                    print(ex)

                samples_clip = samples[:, self.initial_clip:]
                self.out_queue.put(samples_clip)
                self.Q_run.get()
