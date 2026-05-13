# Minimal tqdm stub for tests when tqdm is not installed
class tqdm:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable or [])

def trange(n, **kwargs):
    return range(n)

__all__ = ['tqdm','trange']
