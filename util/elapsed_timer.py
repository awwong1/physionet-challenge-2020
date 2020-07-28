from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def ElapsedTimer():
    start_time = default_timer()

    class _Timer():
        start = start_time
        end = default_timer()
        duration = end - start

    yield _Timer

    end_time = default_timer()
    _Timer.end = end_time
    _Timer.duration = end_time - start_time
