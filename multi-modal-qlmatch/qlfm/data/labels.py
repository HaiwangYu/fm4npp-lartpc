"""Downstream task definitions (single source of truth). Thresholds fixed after the data audit.

T1 hit     : per-hit neutrino vs cosmic/noise, on events with a visible in-AV neutrino.
T2 evt_nu  : event has a visible in-AV neutrino interaction (1) vs essentially no neutrino charge (0).
T3 flavor  : FV + visible: 0 = numu CC, 1 = nue CC, 2 = NC.
T4 energy  : FV + visible + CC: regress log(E_nu / GeV).
"""
import numpy as np

from qlfm.geometry import FV_MARGIN_CM, in_volume

HIT_NU_EDEP_MEV = 0.05    # hit is "neutrino" if its MC deposit exceeds this
VIS_POS_Y_HITS = 30       # T2 positive: >= this many collection-plane neutrino hits
VIS_NEG_HITS = 5          # T2 negative: <= this many neutrino hits (all planes)

TASKS = ("hit", "evt_nu", "flavor", "energy")
N_OUT = {"hit": 1, "evt_nu": 1, "flavor": 3, "energy": 1}


def event_labels(ev):
    """ev: dict of per-event arrays (from Corpus.ev_array). Returns eligibility masks + targets."""
    vtx = ev["nu_vtx"]
    in_av = in_volume(vtx)
    in_fv = in_volume(vtx, FV_MARGIN_CM)
    ny = ev["nu_nhits"][:, 2]
    ntot = ev["nu_nhits"].sum(1)
    visible = in_av & (ny >= VIS_POS_Y_HITS)
    nonu = ntot <= VIS_NEG_HITS
    apdg = np.abs(ev["nu_pdg"])
    cc = ev["is_cc"] == 1
    flavor = np.where(~cc, 2, np.where(apdg == 12, 1, 0))
    return {
        "hit": (visible, None),
        "evt_nu": (visible | nonu, visible.astype(np.float32)),
        "flavor": (in_fv & visible, flavor.astype(np.int64)),
        "energy": (in_fv & visible & cc, np.log(np.clip(ev["nu_energy"], 1e-3, None)).astype(np.float32)),
    }


EV_KEYS = ("nu_vtx", "nu_nhits", "nu_pdg", "is_cc", "nu_energy")
