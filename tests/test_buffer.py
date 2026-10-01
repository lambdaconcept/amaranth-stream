"""Tests for amaranth_stream.buffer (Buffer, PipeValid, PipeReady, Delay)."""

import random
import pytest

from amaranth.hdl import unsigned
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.buffer import Buffer, PipeValid, PipeReady, Delay
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_buffer.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


# ---------------------------------------------------------------------------
# PipeValid tests
# ---------------------------------------------------------------------------

class TestPipeValid:
    """Test PipeValid (forward-registered pipeline stage)."""

    def test_pipe_valid_passthrough(self):
        """Data passes through PipeValid correctly with 1-cycle latency."""
        sig = Signature(unsigned(8))
        dut = PipeValid(sig)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_pv_passthrough.vcd")
        assert results == expected

    def test_pipe_valid_backpressure(self):
        """PipeValid handles backpressure (ready deasserted)."""
        sig = Signature(unsigned(8))
        dut = PipeValid(sig)
        results = []
        expected = [0xAA, 0xBB, 0xCC]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=42)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_pv_backpressure.vcd")
        assert results == expected

    def test_pipe_valid_throughput(self):
        """Sustained 1 transfer/cycle when no backpressure."""
        sig = Signature(unsigned(8))
        dut = PipeValid(sig)
        results = []
        count = 20
        expected = list(range(count))

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
                 deadline_ns=100_000, vcd_name="test_pv_throughput.vcd")
        assert results == expected

    def test_pipe_valid_with_first_last(self):
        """first/last signals are buffered correctly through PipeValid."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PipeValid(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_pv_first_last.vcd")

        assert len(results) == 3
        assert results[0]["payload"] == 0x10
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["payload"] == 0x20
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0
        assert results[2]["payload"] == 0x30
        assert results[2]["first"] == 0
        assert results[2]["last"] == 1


# ---------------------------------------------------------------------------
# PipeReady tests
# ---------------------------------------------------------------------------

class TestPipeReady:
    """Test PipeReady (backward-registered pipeline stage)."""

    def test_pipe_ready_passthrough(self):
        """Data passes through PipeReady correctly."""
        sig = Signature(unsigned(8))
        dut = PipeReady(sig)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_pr_passthrough.vcd")
        assert results == expected

    def test_pipe_ready_backpressure(self):
        """PipeReady handles backpressure correctly."""
        sig = Signature(unsigned(8))
        dut = PipeReady(sig)
        results = []
        expected = [0xAA, 0xBB, 0xCC, 0xDD]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_pr_backpressure.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# Buffer full mode tests
# ---------------------------------------------------------------------------

class TestBufferFullMode:
    """Test Buffer with both pipe_valid=True and pipe_ready=True."""

    def test_buffer_full_mode(self):
        """Both pipe_valid and pipe_ready: data passes through correctly."""
        sig = Signature(unsigned(8))
        dut = Buffer(sig, pipe_valid=True, pipe_ready=True)
        results = []
        expected = [0x01, 0x02, 0x03, 0x04, 0x05]

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
                 deadline_ns=100_000, vcd_name="test_buf_full.vcd")
        assert results == expected

    def test_buffer_full_mode_backpressure(self):
        """Full mode with random backpressure."""
        sig = Signature(unsigned(8))
        dut = Buffer(sig, pipe_valid=True, pipe_ready=True)
        results = []
        expected = list(range(10))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=11)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=22)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_buf_full_bp.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# Buffer passthrough mode tests
# ---------------------------------------------------------------------------

class TestBufferPassthrough:
    """Test Buffer with pipe_valid=False and pipe_ready=False (wire-through)."""

    def test_buffer_passthrough_mode(self):
        """Neither pipe_valid nor pipe_ready: wire-through."""
        sig = Signature(unsigned(8))
        dut = Buffer(sig, pipe_valid=False, pipe_ready=False)
        results = []
        expected = [0xDE, 0xAD, 0xBE, 0xEF]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_buf_passthrough.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# Delay tests
# ---------------------------------------------------------------------------

class TestDelay:
    """Test Delay (N-stage pipeline delay)."""

    def test_delay_zero_stages(self):
        """0 stages = wire-through."""
        sig = Signature(unsigned(8))
        dut = Delay(sig, stages=0)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_delay_0.vcd")
        assert results == expected

    def test_delay_one_stage(self):
        """1 stage = same as PipeValid."""
        sig = Signature(unsigned(8))
        dut = Delay(sig, stages=1)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_delay_1.vcd")
        assert results == expected

    def test_delay_multi_stage(self):
        """N stages of delay."""
        sig = Signature(unsigned(8))
        dut = Delay(sig, stages=4)
        results = []
        expected = [0x01, 0x02, 0x03, 0x04, 0x05]

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
                 deadline_ns=100_000, vcd_name="test_delay_4.vcd")
        assert results == expected

    def test_delay_multi_stage_backpressure(self):
        """N stages with random backpressure."""
        sig = Signature(unsigned(8))
        dut = Delay(sig, stages=3)
        results = []
        expected = list(range(8))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=77)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=88)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_delay_3_bp.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

class TestStress:
    """Random stress tests."""

    def test_stress_random_pipe_valid(self):
        """Random valid/ready with many transfers through PipeValid."""
        sig = Signature(unsigned(8))
        dut = PipeValid(sig)
        results = []
        expected = [random.Random(42).randint(0, 255) for _ in range(50)]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=100)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=200)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_stress_pv.vcd")
        assert results == expected

    def test_stress_random_pipe_ready(self):
        """Random valid/ready with many transfers through PipeReady."""
        sig = Signature(unsigned(8))
        dut = PipeReady(sig)
        results = []
        expected = [random.Random(43).randint(0, 255) for _ in range(50)]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=101)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=201)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_stress_pr.vcd")
        assert results == expected

    def test_stress_random_full(self):
        """Random valid/ready with many transfers through full Buffer."""
        sig = Signature(unsigned(8))
        dut = Buffer(sig, pipe_valid=True, pipe_ready=True)
        results = []
        expected = [random.Random(44).randint(0, 255) for _ in range(50)]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=102)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=202)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_stress_full.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# Optional signal tests
# ---------------------------------------------------------------------------

class TestOptionalSignals:
    """Test that optional signals (param, keep) are buffered correctly."""

    def test_with_param(self):
        """param signal is buffered through PipeValid."""
        sig = Signature(unsigned(8), param_shape=unsigned(4))
        dut = PipeValid(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, param=0x5)
            await sender.send(ctx, 0xBB, param=0xA)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1)
            beat2 = await receiver.recv(ctx)
            results.append(beat2)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_param.vcd")

        assert results[0]["payload"] == 0xAA
        assert results[0]["param"] == 0x5
        assert results[1]["payload"] == 0xBB
        assert results[1]["param"] == 0xA

    def test_with_keep(self):
        """keep signal is buffered through PipeValid."""
        sig = Signature(unsigned(16), has_keep=True)
        dut = PipeValid(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x1234, keep=0x3)
            await sender.send(ctx, 0x5678, keep=0x1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1)
            beat2 = await receiver.recv(ctx)
            results.append(beat2)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_keep.vcd")

        assert results[0]["payload"] == 0x1234
        assert results[0]["keep"] == 0x3
        assert results[1]["payload"] == 0x5678
        assert results[1]["keep"] == 0x1

    def test_with_param_pipe_ready(self):
        """param signal is buffered through PipeReady."""
        sig = Signature(unsigned(8), param_shape=unsigned(4))
        dut = PipeReady(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, param=0x5)
            await sender.send(ctx, 0xBB, param=0xA)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1)
            beat2 = await receiver.recv(ctx)
            results.append(beat2)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_param_pr.vcd")

        assert results[0]["payload"] == 0xAA
        assert results[0]["param"] == 0x5
        assert results[1]["payload"] == 0xBB
        assert results[1]["param"] == 0xA

    def test_with_keep_pipe_ready(self):
        """keep signal is buffered through PipeReady."""
        sig = Signature(unsigned(16), has_keep=True)
        dut = PipeReady(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x1234, keep=0x3)
            await sender.send(ctx, 0x5678, keep=0x1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1)
            beat2 = await receiver.recv(ctx)
            results.append(beat2)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_keep_pr.vcd")

        assert results[0]["payload"] == 0x1234
        assert results[0]["keep"] == 0x3
        assert results[1]["payload"] == 0x5678
        assert results[1]["keep"] == 0x1

    def test_all_optional_signals(self):
        """All optional signals (first, last, param, keep) through PipeValid."""
        sig = Signature(unsigned(8), has_first_last=True,
                        param_shape=unsigned(4), has_keep=True)
        dut = PipeValid(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, first=1, last=0, param=0x3, keep=0x1)
            await sender.send(ctx, 0xBB, first=0, last=1, param=0x7, keep=0x1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1)
            beat2 = await receiver.recv(ctx)
            results.append(beat2)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_all_optional.vcd")

        assert results[0]["payload"] == 0xAA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[0]["param"] == 0x3
        assert results[0]["keep"] == 0x1
        assert results[1]["payload"] == 0xBB
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1
        assert results[1]["param"] == 0x7
        assert results[1]["keep"] == 0x1


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    """Test constructor parameter validation."""

    def test_buffer_rejects_non_signature(self):
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            Buffer("not a signature")

    def test_delay_rejects_non_signature(self):
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            Delay("not a signature")

    def test_delay_rejects_negative_stages(self):
        sig = Signature(unsigned(8))
        with pytest.raises(ValueError, match="stages must be >= 0"):
            Delay(sig, stages=-1)
