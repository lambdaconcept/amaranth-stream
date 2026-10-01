"""Tests for amaranth_stream.routing (StreamMux, StreamDemux, StreamGate, StreamSplitter, StreamJoiner)."""

import random
import pytest

from amaranth.hdl import unsigned
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.routing import StreamMux, StreamDemux, StreamGate, StreamSplitter, StreamJoiner
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name=None):
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


# ===========================================================================
# StreamMux tests
# ===========================================================================

class TestStreamMux:
    """Test StreamMux (N:1 multiplexer)."""

    def test_mux_basic(self):
        """Select between 2 inputs."""
        sig = Signature(unsigned(8))
        dut = StreamMux(sig, 2)
        results = []

        async def tb(ctx):
            sender0 = StreamSimSender(dut.get_input(0))
            sender1 = StreamSimSender(dut.get_input(1))
            receiver = StreamSimReceiver(dut.o_stream)

            # Select input 0
            ctx.set(dut.sel, 0)
            ctx.set(dut.get_input(0).payload, 0xAA)
            ctx.set(dut.get_input(0).valid, 1)
            ctx.set(dut.o_stream.ready, 1)
            _, _, valid_val, payload_val = await ctx.tick().sample(
                dut.o_stream.valid, dut.o_stream.payload)
            assert valid_val == 1
            assert payload_val == 0xAA
            ctx.set(dut.get_input(0).valid, 0)

            # Select input 1
            ctx.set(dut.sel, 1)
            ctx.set(dut.get_input(1).payload, 0xBB)
            ctx.set(dut.get_input(1).valid, 1)
            _, _, valid_val, payload_val = await ctx.tick().sample(
                dut.o_stream.valid, dut.o_stream.payload)
            assert valid_val == 1
            assert payload_val == 0xBB

        _run_sim(dut, tb)

    def test_mux_3_inputs(self):
        """Select between 3 inputs using BFMs."""
        sig = Signature(unsigned(8))
        dut = StreamMux(sig, 3)
        results = []

        async def tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)

            # Send from input 0
            ctx.set(dut.sel, 0)
            ctx.set(dut.get_input(0).payload, 0x10)
            ctx.set(dut.get_input(0).valid, 1)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])
            ctx.set(dut.get_input(0).valid, 0)

            # Send from input 1
            ctx.set(dut.sel, 1)
            ctx.set(dut.get_input(1).payload, 0x20)
            ctx.set(dut.get_input(1).valid, 1)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])
            ctx.set(dut.get_input(1).valid, 0)

            # Send from input 2
            ctx.set(dut.sel, 2)
            ctx.set(dut.get_input(2).payload, 0x30)
            ctx.set(dut.get_input(2).valid, 1)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])

        _run_sim(dut, tb)
        assert results == [0x10, 0x20, 0x30]

    def test_mux_switching(self):
        """Change sel during operation."""
        sig = Signature(unsigned(8))
        dut = StreamMux(sig, 2)
        results = []

        async def tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)

            for i in range(6):
                sel_val = i % 2
                ctx.set(dut.sel, sel_val)
                payload = 0x10 + i
                ctx.set(dut.get_input(sel_val).payload, payload)
                ctx.set(dut.get_input(sel_val).valid, 1)
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])
                ctx.set(dut.get_input(sel_val).valid, 0)

        _run_sim(dut, tb)
        assert results == [0x10, 0x11, 0x12, 0x13, 0x14, 0x15]

    def test_mux_backpressure(self):
        """Only selected input gets ready."""
        sig = Signature(unsigned(8))
        dut = StreamMux(sig, 2)

        async def tb(ctx):
            # Select input 0, assert output ready
            ctx.set(dut.sel, 0)
            ctx.set(dut.o_stream.ready, 1)
            _, _, ready0, ready1 = await ctx.tick().sample(
                dut.get_input(0).ready, dut.get_input(1).ready)
            assert ready0 == 1
            assert ready1 == 0

            # Select input 1
            ctx.set(dut.sel, 1)
            _, _, ready0, ready1 = await ctx.tick().sample(
                dut.get_input(0).ready, dut.get_input(1).ready)
            assert ready0 == 0
            assert ready1 == 1

            # Deassert output ready
            ctx.set(dut.o_stream.ready, 0)
            _, _, ready0, ready1 = await ctx.tick().sample(
                dut.get_input(0).ready, dut.get_input(1).ready)
            assert ready0 == 0
            assert ready1 == 0

        _run_sim(dut, tb)

    def test_mux_with_first_last(self):
        """first/last propagation through mux."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamMux(sig, 2)
        results = []

        async def tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)

            # Send a packet from input 0
            ctx.set(dut.sel, 0)
            inp = dut.get_input(0)

            # Beat 1: first=1, last=0
            ctx.set(inp.payload, 0xA1)
            ctx.set(inp.valid, 1)
            ctx.set(inp.first, 1)
            ctx.set(inp.last, 0)
            beat = await receiver.recv(ctx)
            results.append(beat)

            # Beat 2: first=0, last=1
            ctx.set(inp.payload, 0xA2)
            ctx.set(inp.first, 0)
            ctx.set(inp.last, 1)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, tb)
        assert results[0]["payload"] == 0xA1
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["payload"] == 0xA2
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

    def test_mux_stress(self):
        """Random valid/ready stress test."""
        sig = Signature(unsigned(8))
        dut = StreamMux(sig, 2)
        results = []
        expected = list(range(20))

        async def sender_tb(ctx):
            rng = random.Random(42)
            for val in expected:
                sel_val = rng.randint(0, 1)
                ctx.set(dut.sel, sel_val)
                sender = StreamSimSender(dut.get_input(sel_val), random_valid=True, seed=val)
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000)
        assert results == expected

    def test_mux_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamMux("not a sig", 2)
        sig = Signature(unsigned(8))
        with pytest.raises(ValueError, match="n must be >= 1"):
            StreamMux(sig, 0)


# ===========================================================================
# StreamDemux tests
# ===========================================================================

class TestStreamDemux:
    """Test StreamDemux (1:N demultiplexer)."""

    def test_demux_basic(self):
        """Route to 2 outputs."""
        sig = Signature(unsigned(8))
        dut = StreamDemux(sig, 2)
        results0 = []
        results1 = []

        async def tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            recv0 = StreamSimReceiver(dut.get_output(0))
            recv1 = StreamSimReceiver(dut.get_output(1))

            # Route to output 0
            ctx.set(dut.sel, 0)
            ctx.set(dut.i_stream.payload, 0xAA)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.get_output(0).ready, 1)
            ctx.set(dut.get_output(1).ready, 1)

            _, _, v0, p0, v1, p1 = await ctx.tick().sample(
                dut.get_output(0).valid, dut.get_output(0).payload,
                dut.get_output(1).valid, dut.get_output(1).payload)
            assert v0 == 1
            assert p0 == 0xAA
            assert v1 == 0  # Not selected

            # Route to output 1
            ctx.set(dut.sel, 1)
            ctx.set(dut.i_stream.payload, 0xBB)

            _, _, v0, p0, v1, p1 = await ctx.tick().sample(
                dut.get_output(0).valid, dut.get_output(0).payload,
                dut.get_output(1).valid, dut.get_output(1).payload)
            assert v0 == 0  # Not selected
            assert v1 == 1
            assert p1 == 0xBB

        _run_sim(dut, tb)

    def test_demux_switching(self):
        """Change sel during operation."""
        sig = Signature(unsigned(8))
        dut = StreamDemux(sig, 2)
        results = {0: [], 1: []}

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for i in range(6):
                ctx.set(dut.sel, i % 2)
                await sender.send(ctx, 0x10 + i)

        async def recv0_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0))
            for _ in range(3):
                beat = await recv.recv(ctx)
                results[0].append(beat["payload"])

        async def recv1_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(1))
            for _ in range(3):
                beat = await recv.recv(ctx)
                results[1].append(beat["payload"])

        _run_sim(dut, sender_tb, recv0_tb, recv1_tb, deadline_ns=100_000)
        assert results[0] == [0x10, 0x12, 0x14]
        assert results[1] == [0x11, 0x13, 0x15]

    def test_demux_backpressure(self):
        """Ready from selected output propagates to input."""
        sig = Signature(unsigned(8))
        dut = StreamDemux(sig, 2)

        async def tb(ctx):
            # Select output 0, only output 0 ready
            ctx.set(dut.sel, 0)
            ctx.set(dut.get_output(0).ready, 1)
            ctx.set(dut.get_output(1).ready, 0)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 1  # Ready from selected output 0

            # Select output 1, only output 0 ready
            ctx.set(dut.sel, 1)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 0  # Output 1 not ready

            # Make output 1 ready
            ctx.set(dut.get_output(1).ready, 1)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 1

        _run_sim(dut, tb)

    def test_demux_with_first_last(self):
        """first/last propagation through demux."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamDemux(sig, 2)
        results = []

        async def tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0))

            ctx.set(dut.sel, 0)

            # Beat 1: first=1, last=0
            ctx.set(dut.i_stream.payload, 0xA1)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.i_stream.first, 1)
            ctx.set(dut.i_stream.last, 0)
            beat = await recv.recv(ctx)
            results.append(beat)

            # Beat 2: first=0, last=1
            ctx.set(dut.i_stream.payload, 0xA2)
            ctx.set(dut.i_stream.first, 0)
            ctx.set(dut.i_stream.last, 1)
            beat = await recv.recv(ctx)
            results.append(beat)

        _run_sim(dut, tb)
        assert results[0]["payload"] == 0xA1
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["payload"] == 0xA2
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

    def test_demux_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamDemux("not a sig", 2)
        sig = Signature(unsigned(8))
        with pytest.raises(ValueError, match="n must be >= 1"):
            StreamDemux(sig, 0)


# ===========================================================================
# StreamGate tests
# ===========================================================================

class TestStreamGate:
    """Test StreamGate (enable/disable gate)."""

    def test_gate_enabled(self):
        """Pass through when enabled."""
        sig = Signature(unsigned(8))
        dut = StreamGate(sig)
        results = []
        expected = [0xAA, 0xBB, 0xCC]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            ctx.set(dut.en, 1)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb)
        assert results == expected

    def test_gate_disabled_backpressure(self):
        """Backpressure when disabled (discard=False)."""
        sig = Signature(unsigned(8))
        dut = StreamGate(sig, discard=False)

        async def tb(ctx):
            # Disable gate
            ctx.set(dut.en, 0)
            ctx.set(dut.i_stream.payload, 0xAA)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.o_stream.ready, 1)

            _, _, o_valid, i_ready = await ctx.tick().sample(
                dut.o_stream.valid, dut.i_stream.ready)
            assert o_valid == 0  # Output not valid
            assert i_ready == 0  # Input not ready (backpressure)

        _run_sim(dut, tb)

    def test_gate_disabled_discard(self):
        """Consume and discard when disabled (discard=True)."""
        sig = Signature(unsigned(8))
        dut = StreamGate(sig, discard=True)

        async def tb(ctx):
            # Disable gate
            ctx.set(dut.en, 0)
            ctx.set(dut.i_stream.payload, 0xAA)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.o_stream.ready, 1)

            _, _, o_valid, i_ready = await ctx.tick().sample(
                dut.o_stream.valid, dut.i_stream.ready)
            assert o_valid == 0  # Output not valid
            assert i_ready == 1  # Input ready (discarding)

        _run_sim(dut, tb)

    def test_gate_toggle(self):
        """Toggle enable during operation."""
        sig = Signature(unsigned(8))
        dut = StreamGate(sig, discard=False)
        results = []

        async def tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)

            # Enable: send a value
            ctx.set(dut.en, 1)
            ctx.set(dut.i_stream.payload, 0xAA)
            ctx.set(dut.i_stream.valid, 1)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])

            # Disable: verify backpressure
            ctx.set(dut.en, 0)
            ctx.set(dut.i_stream.payload, 0xBB)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.o_stream.ready, 1)
            _, _, o_valid, i_ready = await ctx.tick().sample(
                dut.o_stream.valid, dut.i_stream.ready)
            assert o_valid == 0
            assert i_ready == 0

            # Re-enable: send another value
            ctx.set(dut.en, 1)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])

        _run_sim(dut, tb)
        assert results == [0xAA, 0xBB]

    def test_gate_discard_toggle(self):
        """Toggle enable with discard=True."""
        sig = Signature(unsigned(8))
        dut = StreamGate(sig, discard=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Gate is disabled, this should be discarded
            ctx.set(dut.en, 0)
            await sender.send(ctx, 0xFF)  # Discarded
            # Enable gate
            ctx.set(dut.en, 1)
            await sender.send(ctx, 0xAA)
            await sender.send(ctx, 0xBB)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Should only receive the values sent while enabled
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=100_000)
        assert results == [0xAA, 0xBB]

    def test_gate_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamGate("not a sig")


# ===========================================================================
# StreamSplitter tests
# ===========================================================================

class TestStreamSplitter:
    """Test StreamSplitter (1:N broadcast/fanout)."""

    def test_splitter_basic(self):
        """Broadcast to 2 outputs."""
        sig = Signature(unsigned(8))
        dut = StreamSplitter(sig, 2)
        results0 = []
        results1 = []
        expected = [0xAA, 0xBB, 0xCC]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def recv0_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0))
            for _ in expected:
                beat = await recv.recv(ctx)
                results0.append(beat["payload"])

        async def recv1_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(1))
            for _ in expected:
                beat = await recv.recv(ctx)
                results1.append(beat["payload"])

        _run_sim(dut, sender_tb, recv0_tb, recv1_tb, deadline_ns=100_000)
        assert results0 == expected
        assert results1 == expected

    def test_splitter_3_outputs(self):
        """Broadcast to 3 outputs."""
        sig = Signature(unsigned(8))
        dut = StreamSplitter(sig, 3)
        results = {0: [], 1: [], 2: []}
        expected = [0x10, 0x20, 0x30]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def make_recv_tb(idx):
            async def recv_tb(ctx):
                recv = StreamSimReceiver(dut.get_output(idx))
                for _ in expected:
                    beat = await recv.recv(ctx)
                    results[idx].append(beat["payload"])
            return recv_tb

        # We need to create the coroutines properly
        async def recv0_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0))
            for _ in expected:
                beat = await recv.recv(ctx)
                results[0].append(beat["payload"])

        async def recv1_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(1))
            for _ in expected:
                beat = await recv.recv(ctx)
                results[1].append(beat["payload"])

        async def recv2_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(2))
            for _ in expected:
                beat = await recv.recv(ctx)
                results[2].append(beat["payload"])

        _run_sim(dut, sender_tb, recv0_tb, recv1_tb, recv2_tb, deadline_ns=100_000)
        for idx in range(3):
            assert results[idx] == expected

    def test_splitter_backpressure(self):
        """One slow output blocks all."""
        sig = Signature(unsigned(8))
        dut = StreamSplitter(sig, 2)

        async def tb(ctx):
            # Both outputs ready: input should be ready
            ctx.set(dut.get_output(0).ready, 1)
            ctx.set(dut.get_output(1).ready, 1)
            ctx.set(dut.i_stream.valid, 1)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 1

            # Output 1 not ready: input should not be ready
            ctx.set(dut.get_output(1).ready, 0)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 0

            # Output 0 not ready: input should not be ready
            ctx.set(dut.get_output(0).ready, 0)
            ctx.set(dut.get_output(1).ready, 1)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 0

        _run_sim(dut, tb)

    def test_splitter_with_first_last(self):
        """first/last propagation through splitter."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamSplitter(sig, 2)
        results0 = []
        results1 = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0xA1, 0xA2, 0xA3])

        async def recv0_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0))
            packet = await recv.recv_packet(ctx)
            results0.extend(packet)

        async def recv1_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(1))
            packet = await recv.recv_packet(ctx)
            results1.extend(packet)

        _run_sim(dut, sender_tb, recv0_tb, recv1_tb, deadline_ns=100_000)

        assert len(results0) == 3
        assert results0[0]["first"] == 1
        assert results0[2]["last"] == 1
        assert len(results1) == 3
        assert results1[0]["first"] == 1
        assert results1[2]["last"] == 1

    def test_splitter_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamSplitter("not a sig", 2)
        sig = Signature(unsigned(8))
        with pytest.raises(ValueError, match="n must be >= 1"):
            StreamSplitter(sig, 0)


# ===========================================================================
# StreamJoiner tests
# ===========================================================================

class TestStreamJoiner:
    """Test StreamJoiner (N:1 interleaved join)."""

    def test_joiner_basic(self):
        """Round-robin 2 inputs."""
        sig = Signature(unsigned(8))
        dut = StreamJoiner(sig, 2)
        results = []

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0))
            await sender.send(ctx, 0xA0)
            await sender.send(ctx, 0xA1)
            await sender.send(ctx, 0xA2)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1))
            await sender.send(ctx, 0xB0)
            await sender.send(ctx, 0xB1)
            await sender.send(ctx, 0xB2)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(6):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender0_tb, sender1_tb, receiver_tb, deadline_ns=100_000)
        # Round-robin: 0, 1, 0, 1, 0, 1
        assert results == [0xA0, 0xB0, 0xA1, 0xB1, 0xA2, 0xB2]

    def test_joiner_3_inputs(self):
        """Round-robin 3 inputs."""
        sig = Signature(unsigned(8))
        dut = StreamJoiner(sig, 3)
        results = []

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0))
            await sender.send(ctx, 0xA0)
            await sender.send(ctx, 0xA1)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1))
            await sender.send(ctx, 0xB0)
            await sender.send(ctx, 0xB1)

        async def sender2_tb(ctx):
            sender = StreamSimSender(dut.get_input(2))
            await sender.send(ctx, 0xC0)
            await sender.send(ctx, 0xC1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(6):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender0_tb, sender1_tb, sender2_tb, receiver_tb,
                 deadline_ns=100_000)
        # Round-robin: 0, 1, 2, 0, 1, 2
        assert results == [0xA0, 0xB0, 0xC0, 0xA1, 0xB1, 0xC1]

    def test_joiner_backpressure(self):
        """Backpressure handling — only current input gets ready."""
        sig = Signature(unsigned(8))
        dut = StreamJoiner(sig, 2)

        async def tb(ctx):
            # Initially sel=0, assert output ready
            ctx.set(dut.o_stream.ready, 1)
            ctx.set(dut.get_input(0).valid, 1)
            ctx.set(dut.get_input(0).payload, 0xAA)
            ctx.set(dut.get_input(1).valid, 1)
            ctx.set(dut.get_input(1).payload, 0xBB)

            _, _, r0, r1 = await ctx.tick().sample(
                dut.get_input(0).ready, dut.get_input(1).ready)
            assert r0 == 1  # Current input
            assert r1 == 0  # Not current

            # After transfer, sel advances to 1
            _, _, r0, r1 = await ctx.tick().sample(
                dut.get_input(0).ready, dut.get_input(1).ready)
            assert r0 == 0
            assert r1 == 1

        _run_sim(dut, tb)

    def test_joiner_stress(self):
        """Random valid/ready stress test."""
        sig = Signature(unsigned(8))
        dut = StreamJoiner(sig, 2)
        results = []
        n_per_input = 10
        expected_0 = list(range(0, n_per_input))
        expected_1 = list(range(100, 100 + n_per_input))

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0), random_valid=True, seed=10)
            for val in expected_0:
                await sender.send(ctx, val)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1), random_valid=True, seed=20)
            for val in expected_1:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=30)
            for _ in range(n_per_input * 2):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender0_tb, sender1_tb, receiver_tb, deadline_ns=500_000)

        # Verify round-robin interleaving
        got_0 = [results[i] for i in range(0, len(results), 2)]
        got_1 = [results[i] for i in range(1, len(results), 2)]
        assert got_0 == expected_0
        assert got_1 == expected_1

    def test_joiner_with_first_last(self):
        """first/last propagation through joiner."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamJoiner(sig, 2)
        results = []

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0))
            await sender.send(ctx, 0xA0, first=1, last=1)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1))
            await sender.send(ctx, 0xB0, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender0_tb, sender1_tb, receiver_tb, deadline_ns=100_000)
        assert results[0]["payload"] == 0xA0
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1
        assert results[1]["payload"] == 0xB0
        assert results[1]["first"] == 1
        assert results[1]["last"] == 1

    def test_joiner_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamJoiner("not a sig", 2)
        sig = Signature(unsigned(8))
        with pytest.raises(ValueError, match="n must be >= 1"):
            StreamJoiner(sig, 0)
