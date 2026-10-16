"""MicroBooNE detector constants used by the tokenizers and label definitions.

Sources: uboone/OpenSamples microboone_utils.py, arXiv:2011.01375, PMT map from
NuTufts/ublightmodelnet & LArbys/lardly (keyed by OpChannel; x ~ -11 cm behind anode).
"""
import numpy as np

N_PLANES = 3
N_WIRES = (2400, 2400, 3456)
N_TICKS = 6400                # stored TPC window, 0.5 us / tick
TRIG_TICK = 800               # trigger (anode, t0=0) tick in the stored window
TICK_TO_CM = 0.0548965        # drift cm per tick (v_d = 0.1098 cm/us)
CATHODE_TICK = TRIG_TICK + 256.35 / TICK_TO_CM   # ~5470
WIRE_PITCH_CM = 0.3

# active volume (approx., from microboone_utils) and 3 cm fiducial margin
AV = dict(x=(0.0, 256.35), y=(-116.5, 116.5), z=(0.0, 1036.8))
FV_MARGIN_CM = 3.0

N_PMT = 32
PMT_WINDOW_US = 23.4          # beam-gate PMT readout (1500 samples @ 64 MHz)

# OpChannel -> (y, z) cm
PMT_YZ = np.array([
    (55.313, 951.861), (55.822, 911.066), (27.607, 989.712), (-0.722, 865.598),
    (-28.625, 990.356), (-56.514, 951.865), (-56.309, 911.939), (55.625, 751.884),
    (55.800, 711.073), (-0.502, 796.208), (-0.051, 664.203), (-56.284, 751.905),
    (-56.408, 711.274), (55.822, 540.929), (55.771, 500.134), (-0.549, 585.284),
    (-0.875, 453.096), (-56.205, 540.616), (-56.323, 500.221), (54.693, 328.212),
    (54.646, 287.976), (-0.706, 373.839), (-0.829, 242.014), (-57.022, 328.341),
    (-56.261, 287.639), (55.249, 128.355), (55.249, 87.7605), (27.431, 51.1015),
    (-0.303, 173.743), (-28.576, 50.4745), (-56.203, 128.179), (-56.615, 87.8695),
], dtype=np.float32)


def tick_to_x(t):
    """Apparent drift coordinate (cm) assuming t0 = trigger."""
    return (np.asarray(t) - TRIG_TICK) * TICK_TO_CM


def in_volume(xyz, margin=0.0):
    """Boolean mask for points (N,3) inside the active volume shrunk by `margin` cm."""
    xyz = np.asarray(xyz)
    m = np.ones(len(xyz), bool)
    for i, k in enumerate("xyz"):
        lo, hi = AV[k]
        m &= (xyz[:, i] > lo + margin) & (xyz[:, i] < hi - margin)
    return m
