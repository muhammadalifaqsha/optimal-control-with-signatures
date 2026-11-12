"""Result cache for the two-stage workflow: compute on one machine, print and plot on another."""
import os
import pickle


class ResultCache:
    """mode='compute': `save` stores the values of one notebook section and rewrites the cache file at once,
    so an interrupted run keeps what it has finished (sections cached by earlier runs are kept until recomputed).
    mode='load': the file is read once and `load` returns the stored values of a section."""

    def __init__(self, mode, path):
        assert mode in ("compute", "load"), "mode must be 'compute' or 'load'"
        self.mode, self.path, self.data = mode, path, {}
        if mode == "load":
            with open(path, "rb") as f:
                self.data = pickle.load(f)
            print(f"load mode: {len(self.data)} cached sections read from {path}: {list(self.data)}")
        elif os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)

    @property
    def computing(self):
        return self.mode == "compute"

    def save(self, key, **values):
        """compute mode: store the arrays/dicts of one section under `key` and rewrite the cache file."""
        if self.mode != "compute":
            return
        self.data[key] = values
        folder = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(folder, exist_ok=True)
        with open(self.path + ".tmp", "wb") as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(self.path + ".tmp", self.path)

    def load(self, key):
        """load mode: the stored values of one section (a dict)."""
        if key not in self.data:
            raise KeyError(f"section '{key}' is not in {self.path}: run the notebook with MODE = 'compute' first")
        return self.data[key]
