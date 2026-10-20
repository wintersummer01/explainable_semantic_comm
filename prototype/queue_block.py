from queue import Queue
import threading
import time

class CommThread(threading.Thread):
    def __init__(self, in_queue:Queue=None, out_queue:Queue=None):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        while True:
            item = self.in_queue.get()
            output = self.process(item)
            self.out_queue.put(output)

    def process(self):
        raise NotImplementedError("Please implement function 'process'!")
    