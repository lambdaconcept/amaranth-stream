"""Tests for amaranth_stream.pipeline (Pipeline, BufferizeEndpoints)."""

import random
import pytest

from amaranth import *
from amaranth.hdl import unsigned, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.buffer import Buffer, PipeValid, PipeReady
from amaranth_stream.pipeline import Pipeline, BufferizeEndpoints
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=100_000, vcd_name=None):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    if vcd_name:
        with sim.write_vcd(vcd_name):
            sim.run_until(Period(ns=deadline_ns))
    else:
        sim.run_until(Period(ns=deadline_ns))


class PassThrough(wiring.Component):
    """Simple pass-through component for testing pipelines."""

    def __init__(self, signature):
        self._stream_sig = signature
        super().__init__({
            "i_stream": In(signature),
            "o_stream": Out(signature),
        })

    def elaborate(self, platform):
        m = Module()
        sig = self._stream_sig
        # Wire-through
        fwd_names = ["payload"]
        if sig.has_first_last:
            fwd_names.extend(["first", "last"])
        if sig.param_shape is not None:
            fwd_names.append("param")
        if sig.has_keep:
            fwd_names.append("keep")
        for name in fwd_names:
            m.d.comb += getattr(self.o_stream, name).eq(
                getattr(self.i_stream, name))
        m.d.comb += [
            self.o_stream.valid.eq(self.i_stream.valid),
            self.i_stream.ready.eq(self.o_stream.ready),
        ]
        return m


# ===========================================================================
# Pipeline tests
# ===========================================================================

class TestPipeline:
    """Test Pipeline (declarative pipeline builder)."""

    def test_pipeline_single_stage(self):
        """Single buffer stage pipeline."""
        sig = Signature(unsigned(8))
        buf = Buffer(sig)
        dut = Pipeline(buf)
        results = []
        expected = [0xAA, 0xBB, 0xCC, 0xDD]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pipeline_single.vcd")
        assert results == expected

    def test_pipeline_multi_stage(self):
        """Chain of 3 buffer stages."""
        sig = Signature(unsigned(8))
        dut = Pipeline(
            Buffer(sig),
            Buffer(sig),
            Buffer(sig),
        )
        results = []
        expected = [0x10, 0x20, 0x30, 0x40, 0x50]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=300_000, vcd_name="test_pipeline_multi.vcd")
        assert results == expected

    def test_pipeline_mixed_stages(self):
        """Mix of PipeValid, PipeReady, and PassThrough stages."""
        sig = Signature(unsigned(8))
        dut = Pipeline(
            PipeValid(sig),
            PassThrough(sig),
            PipeReady(sig),
        )
        results = []
        expected = [0xAA, 0xBB, 0xCC]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pipeline_mixed.vcd")
        assert results == expected

    def test_pipeline_with_first_last(self):
        """Pipeline preserves first/last framing."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = Pipeline(
            Buffer(sig),
            Buffer(sig),
        )
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0xA1, 0xA2, 0xA3])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            pkt = await receiver.recv_packet(ctx)
            results.extend(pkt)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pipeline_fl.vcd")

        assert len(results) == 3
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0
        assert results[2]["first"] == 0
        assert results[2]["last"] == 1
        assert [r["payload"] for r in results] == [0xA1, 0xA2, 0xA3]

    def test_pipeline_stress(self):
        """Random valid/ready stress test."""
        sig = Signature(unsigned(8))
        dut = Pipeline(
            Buffer(sig),
            Buffer(sig),
        )
        results = []
        expected = list(range(20))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_pipeline_stress.vcd")
        assert results == expected

    def test_pipeline_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        with pytest.raises(ValueError, match="at least one stage"):
            Pipeline()


# ===========================================================================
# BufferizeEndpoints tests
# ===========================================================================

class TestBufferizeEndpoints:
    """Test BufferizeEndpoints (wrap component with buffers)."""

    def test_bufferize_passthrough(self):
        """Wrap a simple pass-through component."""
        sig = Signature(unsigned(8))
        inner = PassThrough(sig)
        dut = BufferizeEndpoints(inner)
        results = []
        expected = [0xAA, 0xBB, 0xCC, 0xDD]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=300_000, vcd_name="test_bufferize.vcd")
        assert results == expected

    def test_bufferize_with_first_last(self):
        """BufferizeEndpoints preserves first/last framing."""
        sig = Signature(unsigned(8), has_first_last=True)
        inner = PassThrough(sig)
        dut = BufferizeEndpoints(inner)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0xA1, 0xA2, 0xA3])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            pkt = await receiver.recv_packet(ctx)
            results.extend(pkt)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=300_000, vcd_name="test_bufferize_fl.vcd")

        assert len(results) == 3
        assert results[0]["first"] == 1
        assert results[2]["last"] == 1
        assert [r["payload"] for r in results] == [0xA1, 0xA2, 0xA3]

    def test_bufferize_stress(self):
        """Random valid/ready stress test with bufferized endpoints."""
        sig = Signature(unsigned(8))
        inner = PassThrough(sig)
        dut = BufferizeEndpoints(inner)
        results = []
        expected = list(range(15))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_bufferize_stress.vcd")
        assert results == expected

    def test_bufferize_pipe_valid_only(self):
        """BufferizeEndpoints with pipe_valid only."""
        sig = Signature(unsigned(8))
        inner = PassThrough(sig)
        dut = BufferizeEndpoints(inner, pipe_valid=True, pipe_ready=False)
        results = []
        expected = [0x11, 0x22, 0x33]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_bufferize_pv.vcd")
        assert results == expected
