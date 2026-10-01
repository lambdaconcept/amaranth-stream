"""Tests for amaranth_stream.axi_stream (AXIStreamSignature, AXIStreamToStream, StreamToAXIStream)."""

import pytest

from amaranth import *
from amaranth.hdl import unsigned, Signal, ClockDomain
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out, connect
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.axi_stream import (
    AXIStreamSignature,
    AXIStreamToStream,
    StreamToAXIStream,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_axi.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


# ---------------------------------------------------------------------------
# AXIStreamSignature tests
# ---------------------------------------------------------------------------

class TestAXIStreamSignature:
    """Test AXIStreamSignature creation."""

    def test_axi_signature_basic(self):
        """Create basic AXI-Stream signature."""
        sig = AXIStreamSignature(32)
        assert sig.data_width == 32
        assert sig.id_width == 0
        assert sig.dest_width == 0
        assert sig.user_width == 0
        assert "tdata" in sig.members
        assert "tvalid" in sig.members
        assert "tready" in sig.members
        assert "tkeep" in sig.members
        assert "tstrb" in sig.members
        assert "tlast" in sig.members
        assert "tid" not in sig.members
        assert "tdest" not in sig.members
        assert "tuser" not in sig.members

    def test_axi_signature_with_id(self):
        """Create AXI-Stream signature with TID."""
        sig = AXIStreamSignature(16, id_width=4, dest_width=8, user_width=2)
        assert sig.data_width == 16
        assert sig.id_width == 4
        assert sig.dest_width == 8
        assert sig.user_width == 2
        assert "tid" in sig.members
        assert "tdest" in sig.members
        assert "tuser" in sig.members

    def test_axi_signature_rejects_non_multiple_of_8(self):
        """data_width must be a multiple of 8."""
        with pytest.raises(ValueError, match="multiple of 8"):
            AXIStreamSignature(10)


# ---------------------------------------------------------------------------
# AXIStreamToStream tests
# ---------------------------------------------------------------------------

class _AXIToStreamHarness(wiring.Component):
    """Test harness for AXIStreamToStream."""

    def __init__(self, data_width):
        self._data_width = data_width
        axi_sig = AXIStreamSignature(data_width)
        stream_sig = Signature(unsigned(data_width), has_first_last=True, has_keep=True)
        self._bridge = AXIStreamToStream(axi_sig, stream_sig)
        super().__init__({
            "axi": In(axi_sig),
            "o_stream": Out(stream_sig),
        })

    def elaborate(self, platform):
        m = Module()
        m.submodules.bridge = bridge = self._bridge

        # Connect AXI input
        m.d.comb += [
            bridge.axi.tdata.eq(self.axi.tdata),
            bridge.axi.tvalid.eq(self.axi.tvalid),
            self.axi.tready.eq(bridge.axi.tready),
            bridge.axi.tkeep.eq(self.axi.tkeep),
            bridge.axi.tstrb.eq(self.axi.tstrb),
            bridge.axi.tlast.eq(self.axi.tlast),
        ]

        # Connect stream output
        m.d.comb += [
            self.o_stream.payload.eq(bridge.o_stream.payload),
            self.o_stream.valid.eq(bridge.o_stream.valid),
            bridge.o_stream.ready.eq(self.o_stream.ready),
            self.o_stream.first.eq(bridge.o_stream.first),
            self.o_stream.last.eq(bridge.o_stream.last),
            self.o_stream.keep.eq(bridge.o_stream.keep),
        ]

        return m


class TestAXIStreamToStream:
    """Test AXIStreamToStream bridge."""

    def test_axi_to_stream_basic(self):
        """Bridge AXI-Stream to amaranth_stream."""
        dut = _AXIToStreamHarness(16)
        results = []

        async def tb(ctx):
            # Send a 3-beat packet via AXI
            for i, (data, last) in enumerate([(0x1234, 0), (0x5678, 0), (0x9ABC, 1)]):
                ctx.set(dut.axi.tdata, data)
                ctx.set(dut.axi.tvalid, 1)
                ctx.set(dut.axi.tlast, last)
                ctx.set(dut.axi.tkeep, 0x3)
                ctx.set(dut.o_stream.ready, 1)
                _, _, payload, valid, first, last_val, keep = await ctx.tick().sample(
                    dut.o_stream.payload, dut.o_stream.valid,
                    dut.o_stream.first, dut.o_stream.last,
                    dut.o_stream.keep)
                if valid:
                    results.append({
                        "payload": payload,
                        "first": first,
                        "last": last_val,
                        "keep": keep,
                    })

            # One more tick to capture the last beat
            ctx.set(dut.axi.tvalid, 0)
            _, _, payload, valid, first, last_val, keep = await ctx.tick().sample(
                dut.o_stream.payload, dut.o_stream.valid,
                dut.o_stream.first, dut.o_stream.last,
                dut.o_stream.keep)
            if valid:
                results.append({
                    "payload": payload,
                    "first": first,
                    "last": last_val,
                    "keep": keep,
                })

        _run_sim(dut, tb, vcd_name="test_axi_to_stream.vcd")
        assert len(results) >= 2  # At least some beats transferred


# ---------------------------------------------------------------------------
# StreamToAXIStream tests
# ---------------------------------------------------------------------------

class _StreamToAXIHarness(wiring.Component):
    """Test harness for StreamToAXIStream."""

    def __init__(self, data_width):
        self._data_width = data_width
        stream_sig = Signature(unsigned(data_width), has_first_last=True, has_keep=True)
        axi_sig = AXIStreamSignature(data_width)
        self._bridge = StreamToAXIStream(stream_sig, axi_sig)
        super().__init__({
            "i_stream": In(stream_sig),
            "axi": Out(axi_sig),
        })

    def elaborate(self, platform):
        m = Module()
        m.submodules.bridge = bridge = self._bridge

        # Connect stream input
        m.d.comb += [
            bridge.i_stream.payload.eq(self.i_stream.payload),
            bridge.i_stream.valid.eq(self.i_stream.valid),
            self.i_stream.ready.eq(bridge.i_stream.ready),
            bridge.i_stream.first.eq(self.i_stream.first),
            bridge.i_stream.last.eq(self.i_stream.last),
            bridge.i_stream.keep.eq(self.i_stream.keep),
        ]

        # Connect AXI output
        m.d.comb += [
            self.axi.tdata.eq(bridge.axi.tdata),
            self.axi.tvalid.eq(bridge.axi.tvalid),
            bridge.axi.tready.eq(self.axi.tready),
            self.axi.tkeep.eq(bridge.axi.tkeep),
            self.axi.tstrb.eq(bridge.axi.tstrb),
            self.axi.tlast.eq(bridge.axi.tlast),
        ]

        return m


class TestStreamToAXIStream:
    """Test StreamToAXIStream bridge."""

    def test_stream_to_axi_basic(self):
        """Bridge amaranth_stream to AXI-Stream."""
        dut = _StreamToAXIHarness(16)
        results = []

        async def tb(ctx):
            # Send beats via stream interface
            for data, last, keep in [(0x1234, 0, 0x3), (0x5678, 1, 0x3)]:
                ctx.set(dut.i_stream.payload, data)
                ctx.set(dut.i_stream.valid, 1)
                ctx.set(dut.i_stream.last, last)
                ctx.set(dut.i_stream.keep, keep)
                ctx.set(dut.axi.tready, 1)
                _, _, tdata, tvalid, tlast, tkeep, tstrb = await ctx.tick().sample(
                    dut.axi.tdata, dut.axi.tvalid,
                    dut.axi.tlast, dut.axi.tkeep, dut.axi.tstrb)
                if tvalid:
                    results.append({
                        "tdata": tdata,
                        "tlast": tlast,
                        "tkeep": tkeep,
                        "tstrb": tstrb,
                    })

            # One more tick
            ctx.set(dut.i_stream.valid, 0)
            _, _, tdata, tvalid, tlast, tkeep, tstrb = await ctx.tick().sample(
                dut.axi.tdata, dut.axi.tvalid,
                dut.axi.tlast, dut.axi.tkeep, dut.axi.tstrb)
            if tvalid:
                results.append({
                    "tdata": tdata,
                    "tlast": tlast,
                    "tkeep": tkeep,
                    "tstrb": tstrb,
                })

        _run_sim(dut, tb, vcd_name="test_stream_to_axi.vcd")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Roundtrip test
# ---------------------------------------------------------------------------

class _RoundtripHarness(wiring.Component):
    """AXI → Stream → AXI roundtrip harness."""

    def __init__(self, data_width):
        self._data_width = data_width
        axi_sig = AXIStreamSignature(data_width)
        stream_sig = Signature(unsigned(data_width), has_first_last=True, has_keep=True)
        self._a2s = AXIStreamToStream(axi_sig, stream_sig)
        self._s2a = StreamToAXIStream(stream_sig, axi_sig)
        super().__init__({
            "axi_in": In(axi_sig),
            "axi_out": Out(axi_sig),
        })

    def elaborate(self, platform):
        m = Module()
        m.submodules.a2s = a2s = self._a2s
        m.submodules.s2a = s2a = self._s2a

        # Connect AXI input to a2s
        m.d.comb += [
            a2s.axi.tdata.eq(self.axi_in.tdata),
            a2s.axi.tvalid.eq(self.axi_in.tvalid),
            self.axi_in.tready.eq(a2s.axi.tready),
            a2s.axi.tkeep.eq(self.axi_in.tkeep),
            a2s.axi.tstrb.eq(self.axi_in.tstrb),
            a2s.axi.tlast.eq(self.axi_in.tlast),
        ]

        # Connect a2s output to s2a input
        m.d.comb += [
            s2a.i_stream.payload.eq(a2s.o_stream.payload),
            s2a.i_stream.valid.eq(a2s.o_stream.valid),
            a2s.o_stream.ready.eq(s2a.i_stream.ready),
            s2a.i_stream.first.eq(a2s.o_stream.first),
            s2a.i_stream.last.eq(a2s.o_stream.last),
            s2a.i_stream.keep.eq(a2s.o_stream.keep),
        ]

        # Connect s2a output to AXI output
        m.d.comb += [
            self.axi_out.tdata.eq(s2a.axi.tdata),
            self.axi_out.tvalid.eq(s2a.axi.tvalid),
            s2a.axi.tready.eq(self.axi_out.tready),
            self.axi_out.tkeep.eq(s2a.axi.tkeep),
            self.axi_out.tstrb.eq(s2a.axi.tstrb),
            self.axi_out.tlast.eq(s2a.axi.tlast),
        ]

        return m


class TestAXIRoundtrip:
    """Test AXI → Stream → AXI roundtrip."""

    def test_axi_roundtrip(self):
        """Data survives AXI → stream → AXI roundtrip."""
        dut = _RoundtripHarness(16)
        results = []

        async def tb(ctx):
            test_data = [(0x1234, 0, 0x3), (0x5678, 0, 0x3), (0x9ABC, 1, 0x1)]

            for tdata, tlast, tkeep in test_data:
                ctx.set(dut.axi_in.tdata, tdata)
                ctx.set(dut.axi_in.tvalid, 1)
                ctx.set(dut.axi_in.tlast, tlast)
                ctx.set(dut.axi_in.tkeep, tkeep)
                ctx.set(dut.axi_out.tready, 1)
                _, _, out_tdata, out_tvalid, out_tlast, out_tkeep = await ctx.tick().sample(
                    dut.axi_out.tdata, dut.axi_out.tvalid,
                    dut.axi_out.tlast, dut.axi_out.tkeep)
                if out_tvalid:
                    results.append({
                        "tdata": out_tdata,
                        "tlast": out_tlast,
                        "tkeep": out_tkeep,
                    })

            # Flush remaining
            ctx.set(dut.axi_in.tvalid, 0)
            for _ in range(3):
                _, _, out_tdata, out_tvalid, out_tlast, out_tkeep = await ctx.tick().sample(
                    dut.axi_out.tdata, dut.axi_out.tvalid,
                    dut.axi_out.tlast, dut.axi_out.tkeep)
                if out_tvalid:
                    results.append({
                        "tdata": out_tdata,
                        "tlast": out_tlast,
                        "tkeep": out_tkeep,
                    })

        _run_sim(dut, tb, vcd_name="test_axi_roundtrip.vcd")
        # Verify data integrity
        assert len(results) >= 2
        # The data values should match what was sent
        data_sent = [0x1234, 0x5678, 0x9ABC]
        data_received = [r["tdata"] for r in results]
        # At least the first values should match (combinational path)
        for sent, received in zip(data_sent, data_received):
            assert sent == received


# ---------------------------------------------------------------------------
# tfirst support tests
# ---------------------------------------------------------------------------

class TestAXIStreamSignatureTfirst:
    """Test AXIStreamSignature with has_tfirst parameter."""

    def test_axi_stream_signature_tfirst(self):
        """Signature with has_tfirst=True includes tfirst member."""
        sig = AXIStreamSignature(32, has_tfirst=True)
        assert sig.has_tfirst is True
        assert "tfirst" in sig.members
        assert "tlast" in sig.members
        assert "tdata" in sig.members

    def test_axi_stream_tfirst_backward_compat(self):
        """Default has_tfirst=False works exactly as before — no tfirst member."""
        sig = AXIStreamSignature(32)
        assert sig.has_tfirst is False
        assert "tfirst" not in sig.members
        # All standard members still present
        assert "tdata" in sig.members
        assert "tvalid" in sig.members
        assert "tready" in sig.members
        assert "tkeep" in sig.members
        assert "tstrb" in sig.members
        assert "tlast" in sig.members

    def test_axi_stream_signature_tfirst_equality(self):
        """Signatures with different has_tfirst are not equal."""
        sig_with = AXIStreamSignature(32, has_tfirst=True)
        sig_without = AXIStreamSignature(32, has_tfirst=False)
        assert sig_with != sig_without
        assert sig_with == AXIStreamSignature(32, has_tfirst=True)

    def test_axi_stream_signature_tfirst_repr(self):
        """repr includes has_tfirst when True."""
        sig = AXIStreamSignature(32, has_tfirst=True)
        assert "has_tfirst=True" in repr(sig)
        sig2 = AXIStreamSignature(32)
        assert "has_tfirst" not in repr(sig2)


class _AXIToStreamTfirstHarness(wiring.Component):
    """Test harness for AXIStreamToStream with tfirst."""

    def __init__(self, data_width):
        self._data_width = data_width
        axi_sig = AXIStreamSignature(data_width, has_tfirst=True)
        stream_sig = Signature(unsigned(data_width), has_first_last=True, has_keep=True)
        self._bridge = AXIStreamToStream(axi_sig, stream_sig)
        super().__init__({
            "axi": In(axi_sig),
            "o_stream": Out(stream_sig),
        })

    def elaborate(self, platform):
        m = Module()
        m.domains += ClockDomain("sync")
        m.submodules.bridge = bridge = self._bridge

        # Connect AXI input (including tfirst)
        m.d.comb += [
            bridge.axi.tdata.eq(self.axi.tdata),
            bridge.axi.tvalid.eq(self.axi.tvalid),
            self.axi.tready.eq(bridge.axi.tready),
            bridge.axi.tkeep.eq(self.axi.tkeep),
            bridge.axi.tstrb.eq(self.axi.tstrb),
            bridge.axi.tlast.eq(self.axi.tlast),
            bridge.axi.tfirst.eq(self.axi.tfirst),
        ]

        # Connect stream output
        m.d.comb += [
            self.o_stream.payload.eq(bridge.o_stream.payload),
            self.o_stream.valid.eq(bridge.o_stream.valid),
            bridge.o_stream.ready.eq(self.o_stream.ready),
            self.o_stream.first.eq(bridge.o_stream.first),
            self.o_stream.last.eq(bridge.o_stream.last),
            self.o_stream.keep.eq(bridge.o_stream.keep),
        ]

        return m


class TestAXIStreamToStreamWithTfirst:
    """Test AXIStreamToStream bridge when AXI-Stream has tfirst."""

    def test_axi_stream_to_stream_with_tfirst(self):
        """Bridge AXI-Stream (with tfirst) to stream — first comes directly from tfirst."""
        dut = _AXIToStreamTfirstHarness(16)
        results = []

        async def tb(ctx):
            # Send a 3-beat packet: tfirst=1 on first beat, 0 on others
            beats = [
                (0x1234, 1, 0, 0x3),  # (data, tfirst, tlast, tkeep)
                (0x5678, 0, 0, 0x3),
                (0x9ABC, 0, 1, 0x3),
            ]
            for data, tfirst, tlast, tkeep in beats:
                ctx.set(dut.axi.tdata, data)
                ctx.set(dut.axi.tvalid, 1)
                ctx.set(dut.axi.tfirst, tfirst)
                ctx.set(dut.axi.tlast, tlast)
                ctx.set(dut.axi.tkeep, tkeep)
                ctx.set(dut.o_stream.ready, 1)
                _, _, payload, valid, first, last_val, keep = await ctx.tick().sample(
                    dut.o_stream.payload, dut.o_stream.valid,
                    dut.o_stream.first, dut.o_stream.last,
                    dut.o_stream.keep)
                if valid:
                    results.append({
                        "payload": payload,
                        "first": first,
                        "last": last_val,
                        "keep": keep,
                    })

            # One more tick to capture the last beat
            ctx.set(dut.axi.tvalid, 0)
            _, _, payload, valid, first, last_val, keep = await ctx.tick().sample(
                dut.o_stream.payload, dut.o_stream.valid,
                dut.o_stream.first, dut.o_stream.last,
                dut.o_stream.keep)
            if valid:
                results.append({
                    "payload": payload,
                    "first": first,
                    "last": last_val,
                    "keep": keep,
                })

        _run_sim(dut, tb, vcd_name="test_axi_to_stream_tfirst.vcd")

        # Verify: first should directly reflect tfirst (combinational),
        # so first beat has first=1, subsequent beats have first=0
        assert len(results) >= 2
        assert results[0]["first"] == 1, f"First beat should have first=1, got {results[0]}"
        for r in results[1:]:
            assert r["first"] == 0, f"Non-first beat should have first=0, got {r}"


class _StreamToAXITfirstHarness(wiring.Component):
    """Test harness for StreamToAXIStream with tfirst."""

    def __init__(self, data_width):
        self._data_width = data_width
        stream_sig = Signature(unsigned(data_width), has_first_last=True, has_keep=True)
        axi_sig = AXIStreamSignature(data_width, has_tfirst=True)
        self._bridge = StreamToAXIStream(stream_sig, axi_sig)
        super().__init__({
            "i_stream": In(stream_sig),
            "axi": Out(axi_sig),
        })

    def elaborate(self, platform):
        m = Module()
        m.submodules.bridge = bridge = self._bridge

        # Connect stream input
        m.d.comb += [
            bridge.i_stream.payload.eq(self.i_stream.payload),
            bridge.i_stream.valid.eq(self.i_stream.valid),
            self.i_stream.ready.eq(bridge.i_stream.ready),
            bridge.i_stream.first.eq(self.i_stream.first),
            bridge.i_stream.last.eq(self.i_stream.last),
            bridge.i_stream.keep.eq(self.i_stream.keep),
        ]

        # Connect AXI output (including tfirst)
        m.d.comb += [
            self.axi.tdata.eq(bridge.axi.tdata),
            self.axi.tvalid.eq(bridge.axi.tvalid),
            bridge.axi.tready.eq(self.axi.tready),
            self.axi.tkeep.eq(bridge.axi.tkeep),
            self.axi.tstrb.eq(bridge.axi.tstrb),
            self.axi.tlast.eq(bridge.axi.tlast),
            self.axi.tfirst.eq(bridge.axi.tfirst),
        ]

        return m


class TestStreamToAXIStreamWithTfirst:
    """Test StreamToAXIStream bridge when AXI-Stream has tfirst."""

    def test_stream_to_axi_stream_with_tfirst(self):
        """Bridge stream to AXI-Stream (with tfirst) — tfirst is driven from first."""
        dut = _StreamToAXITfirstHarness(16)
        results = []

        async def tb(ctx):
            # Send beats via stream interface with first signal
            beats = [
                (0x1234, 1, 0, 0x3),  # (data, first, last, keep)
                (0x5678, 0, 0, 0x3),
                (0x9ABC, 0, 1, 0x3),
            ]
            for data, first, last, keep in beats:
                ctx.set(dut.i_stream.payload, data)
                ctx.set(dut.i_stream.valid, 1)
                ctx.set(dut.i_stream.first, first)
                ctx.set(dut.i_stream.last, last)
                ctx.set(dut.i_stream.keep, keep)
                ctx.set(dut.axi.tready, 1)
                _, _, tdata, tvalid, tfirst, tlast, tkeep, tstrb = await ctx.tick().sample(
                    dut.axi.tdata, dut.axi.tvalid,
                    dut.axi.tfirst, dut.axi.tlast,
                    dut.axi.tkeep, dut.axi.tstrb)
                if tvalid:
                    results.append({
                        "tdata": tdata,
                        "tfirst": tfirst,
                        "tlast": tlast,
                        "tkeep": tkeep,
                        "tstrb": tstrb,
                    })

            # One more tick
            ctx.set(dut.i_stream.valid, 0)
            _, _, tdata, tvalid, tfirst, tlast, tkeep, tstrb = await ctx.tick().sample(
                dut.axi.tdata, dut.axi.tvalid,
                dut.axi.tfirst, dut.axi.tlast,
                dut.axi.tkeep, dut.axi.tstrb)
            if tvalid:
                results.append({
                    "tdata": tdata,
                    "tfirst": tfirst,
                    "tlast": tlast,
                    "tkeep": tkeep,
                    "tstrb": tstrb,
                })

        _run_sim(dut, tb, vcd_name="test_stream_to_axi_tfirst.vcd")

        # Verify: tfirst should directly reflect the stream's first signal
        assert len(results) >= 2
        assert results[0]["tfirst"] == 1, f"First beat should have tfirst=1, got {results[0]}"
        for r in results[1:]:
            assert r["tfirst"] == 0, f"Non-first beat should have tfirst=0, got {r}"
