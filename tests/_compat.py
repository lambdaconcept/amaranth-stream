"""Compatibility helpers for running the tests on released Amaranth versions.

``amaranth.sim.Period`` only exists in Amaranth 0.6+. On Amaranth 0.5.x the
simulator takes plain floating point seconds, so emulate ``Period`` by
returning seconds.
"""

try:
    from amaranth.sim import Period
except ImportError:
    def Period(*, s=0, ms=0, us=0, ns=0, ps=0, fs=0, Hz=None, kHz=None, MHz=None, GHz=None):
        if Hz is not None or kHz is not None or MHz is not None or GHz is not None:
            freq = (Hz or 0) + (kHz or 0) * 1e3 + (MHz or 0) * 1e6 + (GHz or 0) * 1e9
            return 1.0 / freq
        return s + ms * 1e-3 + us * 1e-6 + ns * 1e-9 + ps * 1e-12 + fs * 1e-15
