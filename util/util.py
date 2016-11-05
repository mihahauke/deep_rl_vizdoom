from __future__ import print_function
from threading import Lock
import threading


def sec_to_str(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    res = ""
    if h > 0:
        res += str(h) + "h "
    if h > 0 or m > 0:
        res += str(m) + "m "
    res += str(s) + "s"
    return res


_print_lock = Lock()


def threadsafe_print(*args):
    with _print_lock:
        print(*args)


class ThreadsafeCounter(object):
    def __init__(self, initval=0):
        self._val = 0
        self._lock = threading.Lock()

    def get(self):
        return self._val

    def inc(self, val=1):
        with self._lock:
            self._val += val
        return self._val


class Record(object):
    pass
