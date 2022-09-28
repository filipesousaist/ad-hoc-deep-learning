from threading import Thread
import sys
import fcntl
import os
import select
import time


class FCNTLInputThread(Thread):
    # Adapted from https://blog.tomecek.net/post/non-blocking-stdin-in-python/

    def __init__(self):
        super().__init__()
        self._running = True

    def run(self) -> None:
        fd = sys.stdin.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        epoll = select.epoll()
        epoll.register(fd, select.EPOLLIN)

        try:
            while self._running:
                events = epoll.poll(1)
                for _, _ in events:
                    string = ""
                    while True:
                        char = sys.stdin.read(64)
                        if not char:
                            break
                        string += char
                    if string.lower().startswith("q"):
                        self._running = False
        finally:
            epoll.unregister(fd)
            epoll.close()

        print("[INFO] Thread exiting...")

    def stop(self):
        self._running = False


class SelectInputThread(Thread):
    # Adapted from https://repolinux.wordpress.com/2012/10/09/non-blocking-read-from-stdin-in-python/

    def __init__(self):
        super().__init__()
        self._running = True

        # files monitored for input
        self.read_list = [sys.stdin]
        # select() should wait for this many seconds for input.
        # A smaller number means more cpu usage, but a greater one
        # means a more noticeable delay between input becoming
        # available and the program starting to work on it.
        self.timeout = 0.1  # seconds
        self.last_work_time = time.time()

    def run(self) -> None:
        while self.read_list and self._running:
            ready = select.select(self.read_list, [], [], self.timeout)[0]
            if ready:
                for file in ready:
                    line = file.readline()
                    if not line:  # EOF, remove file from input list
                        self.read_list.remove(file)
                    elif line.rstrip():  # optional: skipping empty lines
                        if line.lower().startswith("q"):
                            self._running = False

        print("[INFO] Thread exiting...")

    def stop(self):
        self._running = False


i_thread = SelectInputThread()
i_thread.start()
i = 2000
while i_thread.is_alive() and i > 0:
    time.sleep(0.01)
    print(i)
    i -= 1

i_thread.stop()
while i_thread.is_alive():
    pass

print("[INFO] Quitting...")
