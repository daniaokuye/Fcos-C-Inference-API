import time


class Profiler(object):
    def __init__(self, names=['main']):
        self.names = names
        self.lasts = {k: 0 for k in names}
        self.totals = self.lasts.copy()
        self.counts = self.lasts.copy()
        self.means = self.lasts.copy()
        self.reset()

    def _insert(self, name):
        self.lasts[name] = time.time()
        self.totals[name] = 0
        self.counts[name] = 0
        self.means[name] = 0
        self.names.append(name)

    def reset(self):
        last = time.time()
        for name in self.names:
            self.lasts[name] = last
            self.totals[name] = 0
            self.counts[name] = 0
            self.means[name] = 0

    def start(self, name='main'):
        if name not in self.names:
            self._insert(name)
        self.lasts[name] = time.time()

    def stop(self, name='main'):
        if name not in self.names:
            self._insert(name)
        self.totals[name] += time.time() - self.lasts[name]
        self.counts[name] += 1
        self.means[name] = self.totals[name] / self.counts[name]

    def bump(self, name='main'):
        self.stop(name)
        self.start(name)
