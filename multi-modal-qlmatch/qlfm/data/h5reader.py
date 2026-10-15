"""Reader for the MicroBooNE open-data HDF5 files (hep_hpc ntuple + ph5concat layout).

Each table is a group with one (N, k) dataset per column plus `event_id` (N, 3) and
`event_id.seq_cnt` (n_nonempty, 2) = [event sequence number, row count]. Rows of a
table are contiguous and ordered by event sequence number.
"""
import h5py
import numpy as np


class OpenSampleFile:
    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, "r")
        self.n_events = self.f["event_table/event_id"].shape[0]
        self._offsets = {}

    def tables(self):
        return list(self.f.keys())

    def columns(self, table):
        return [k for k in self.f[table].keys() if not k.startswith("event_id.")]

    def offsets(self, table):
        """(start, count) arrays of length n_events for `table`."""
        if table not in self._offsets:
            sc = self.f[f"{table}/event_id.seq_cnt"][:]
            cnt = np.zeros(self.n_events, np.int64)
            cnt[sc[:, 0]] = sc[:, 1]
            start = np.zeros(self.n_events, np.int64)
            start[1:] = np.cumsum(cnt)[:-1]
            self._offsets[table] = (start, cnt)
        return self._offsets[table]

    def read_range(self, table, ev_lo, ev_hi, cols=None):
        """Read all rows of `table` for events [ev_lo, ev_hi).

        Returns (dict col -> array squeezed to 1-D when k == 1, evt_index per row).
        """
        start, cnt = self.offsets(table)
        r0 = start[ev_lo]
        r1 = start[ev_hi - 1] + cnt[ev_hi - 1]
        g = self.f[table]
        cols = cols or self.columns(table)
        out = {}
        for c in cols:
            a = g[c][r0:r1]
            out[c] = a[:, 0] if a.ndim == 2 and a.shape[1] == 1 else a
        evt = np.repeat(np.arange(ev_lo, ev_hi), cnt[ev_lo:ev_hi])
        return out, evt

    def close(self):
        self.f.close()
