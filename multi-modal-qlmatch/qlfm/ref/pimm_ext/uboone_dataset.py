"""pimm reader for MicroBooNE open-data wire hits (our column-wise cache): one sample = one plane of one event.

Points: (wire * 0.3 cm, drift x cm, plane * 10 cm); energy: hit integral (ADC*tick). The plane offset along z
encodes plane identity (U/V/Y) without mixing planes; per-plane clouds are 2D, so only in-plane augmentations
are used in the configs. Splits/subsets come from our splits file (same as all other methods).
"""
import numpy as np
from torch.utils.data import Dataset

from pimm.datasets.builder import DATASETS
from pimm.datasets.transform import Compose

from qlfm import geometry as G
from qlfm.data.dataset import Corpus

PLANE_DZ_CM = 10.0


def plane_cloud(plane, wire, time, integral, p):
    m = plane == p
    coord = np.stack([wire[m] * G.WIRE_PITCH_CM, (time[m] - G.TRIG_TICK) * G.TICK_TO_CM,
                      np.full(m.sum(), p * PLANE_DZ_CM)], 1).astype(np.float32)
    energy = np.clip(integral[m], 1.0, None).astype(np.float32)[:, None]
    return coord, energy, np.nonzero(m)[0]


@DATASETS.register_module()
class MicroBooNEPlaneDataset(Dataset):
    def __init__(self, corpus, splits, subset="pre_f0.90", transform=None, max_len=None, min_points=64,
                 seed=0, test_mode=False, loop=1):
        self.c = Corpus(corpus)
        idx = np.load(splits)[subset]
        rng = np.random.default_rng(seed)
        samples = np.stack([np.repeat(idx, 3), np.tile(np.arange(3), len(idx))], 1)
        samples = samples[rng.permutation(len(samples))]
        if max_len is not None:
            samples = samples[:max_len]
        self.samples = samples
        self.transform = Compose(transform)
        self.min_points = min_points
        self.loop = loop

    def get_data(self, i):
        g, p = self.samples[i % len(self.samples)]
        s, j = self.c.locate(g)
        sh = self.c.shards[s]
        h0, hn = int(sh.ev("hit_off")[j]), int(sh.ev("hit_cnt")[j])
        plane = np.asarray(sh.hit("plane")[h0:h0 + hn])
        coord, energy, _ = plane_cloud(plane, np.asarray(sh.hit("wire")[h0:h0 + hn]).astype(np.float32),
                                       np.asarray(sh.hit("time")[h0:h0 + hn]).astype(np.float32) / 8.0,
                                       np.asarray(sh.hit("integral")[h0:h0 + hn]).astype(np.float32), p)
        if len(coord) < self.min_points:                      # rare near-empty planes: take the next sample
            return self.get_data((i + 1) % len(self.samples))
        return dict(coord=coord, energy=energy, name=f"{g}:{p}")

    def __getitem__(self, i):
        return self.transform(self.get_data(i))

    def __len__(self):
        return len(self.samples) * self.loop
