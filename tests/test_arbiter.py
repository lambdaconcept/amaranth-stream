"""Tests for amaranth_stream.arbiter (StreamArbiter, StreamDispatcher)."""

import random
import pytest

from amaranth.hdl import unsigned
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.arbiter import StreamArbiter, StreamDispatcher
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


# ===========================================================================
# StreamArbiter tests
# ===========================================================================

class TestStreamArbiter:
    """Test StreamArbiter (packet-aware N:1 arbiter)."""

    def test_arbiter_round_robin_basic(self):
        """2 inputs, round-robin: alternate between inputs."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamArbiter(sig, 2, round_robin=True)
        results = []

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0))
            # Send single-beat packets (first=1, last=1)
            await sender.send(ctx, 0xA0, first=1, last=1)
            await sender.send(ctx, 0xA1, first=1, last=1)
            await sender.send(ctx, 0xA2, first=1, last=1)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1))
            await sender.send(ctx, 0xB0, first=1, last=1)
            await sender.send(ctx, 0xB1, first=1, last=1)
            await sender.send(ctx, 0xB2, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(6):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender0_tb, sender1_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_arbiter_rr_basic.vcd")

        # Round-robin should alternate: 0, 1, 0, 1, 0, 1
        got_0 = [r for r in results if r & 0xF0 == 0xA0]
        got_1 = [r for r in results if r & 0xF0 == 0xB0]
        assert len(got_0) == 3, f"Expected 3 from input 0, got {len(got_0)}: {results}"
        assert len(got_1) == 3, f"Expected 3 from input 1, got {len(got_1)}: {results}"

    def test_arbiter_round_robin_3(self):
        """3 inputs, round-robin."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamArbiter(sig, 3, round_robin=True)
        results = []

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0))
            await sender.send(ctx, 0xA0, first=1, last=1)
            await sender.send(ctx, 0xA1, first=1, last=1)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1))
            await sender.send(ctx, 0xB0, first=1, last=1)
            await sender.send(ctx, 0xB1, first=1, last=1)

        async def sender2_tb(ctx):
            sender = StreamSimSender(dut.get_input(2))
            await sender.send(ctx, 0xC0, first=1, last=1)
            await sender.send(ctx, 0xC1, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(6):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender0_tb, sender1_tb, sender2_tb, receiver_tb,
                 deadline_ns=300_000, vcd_name="test_arbiter_rr_3.vcd")

        # All 6 beats should be received
        got_0 = [r for r in results if (r & 0xF0) == 0xA0]
        got_1 = [r for r in results if (r & 0xF0) == 0xB0]
        got_2 = [r for r in results if (r & 0xF0) == 0xC0]
        assert len(got_0) == 2
        assert len(got_1) == 2
        assert len(got_2) == 2

    def test_arbiter_priority(self):
        """Priority mode: lower index always wins."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamArbiter(sig, 2, round_robin=False)
        results = []

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0))
            # Send 3 single-beat packets
            for i in range(3):
                await sender.send(ctx, 0xA0 + i, first=1, last=1)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1))
            # Send 3 single-beat packets
            for i in range(3):
                await sender.send(ctx, 0xB0 + i, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(6):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender0_tb, sender1_tb, receiver_tb,
                 deadline_ns=300_000, vcd_name="test_arbiter_priority.vcd")

        # In priority mode, input 0 should be served first when both are valid
        # All 6 beats should be received
        assert len(results) == 6
        # Input 0's packets should appear before input 1's (mostly)
        got_0 = [r for r in results if (r & 0xF0) == 0xA0]
        got_1 = [r for r in results if (r & 0xF0) == 0xB0]
        assert len(got_0) == 3
        assert len(got_1) == 3

    def test_arbiter_packet_hold(self):
        """Grant held during multi-beat packet."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamArbiter(sig, 2, round_robin=True)
        results = []

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0))
            # Wait a cycle so input 1 gets the first grant (round-robin starts from 1)
            await ctx.tick()
            # Send a 3-beat packet
            await sender.send(ctx, 0xA0, first=1, last=0)
            await sender.send(ctx, 0xA1, first=0, last=0)
            await sender.send(ctx, 0xA2, first=0, last=1)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1))
            # Input 1 sends while input 0's packet is in progress
            # First, let input 0 start its packet
            await ctx.tick()
            await ctx.tick()
            # Now input 0 should be locked, so this waits
            await sender.send(ctx, 0xB0, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender0_tb, sender1_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_arbiter_pkt_hold.vcd")

        # The 3-beat packet from input 0 should be contiguous
        assert results[:3] == [0xA0, 0xA1, 0xA2], f"Got: {[hex(r) for r in results]}"
        assert results[3] == 0xB0

    def test_arbiter_backpressure(self):
        """Backpressure: only granted input gets ready."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamArbiter(sig, 2, round_robin=True)

        async def tb(ctx):
            # Both inputs valid, output ready
            ctx.set(dut.get_input(0).valid, 1)
            ctx.set(dut.get_input(0).payload, 0xAA)
            ctx.set(dut.get_input(0).first, 1)
            ctx.set(dut.get_input(0).last, 1)
            ctx.set(dut.get_input(1).valid, 1)
            ctx.set(dut.get_input(1).payload, 0xBB)
            ctx.set(dut.get_input(1).first, 1)
            ctx.set(dut.get_input(1).last, 1)
            ctx.set(dut.o_stream.ready, 1)

            _, _, r0, r1, grant = await ctx.tick().sample(
                dut.get_input(0).ready, dut.get_input(1).ready, dut.grant)

            # One should have ready, the other should not
            assert (r0 == 1) != (r1 == 1), f"r0={r0}, r1={r1}"

        _run_sim(dut, tb)

    def test_arbiter_stress(self):
        """Random valid/ready stress test."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamArbiter(sig, 2, round_robin=True)
        results = []
        n_packets = 5

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.get_input(0), random_valid=True, seed=10)
            for i in range(n_packets):
                await sender.send_packet(ctx, [0xA0 + i * 2, 0xA0 + i * 2 + 1])

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.get_input(1), random_valid=True, seed=20)
            for i in range(n_packets):
                await sender.send_packet(ctx, [0xB0 + i * 2, 0xB0 + i * 2 + 1])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=30)
            for _ in range(n_packets * 2 * 2):  # 2 inputs * n_packets * 2 beats
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender0_tb, sender1_tb, receiver_tb,
                 deadline_ns=1_000_000, vcd_name="test_arbiter_stress.vcd")

        # Verify all beats received
        assert len(results) == n_packets * 2 * 2

        # Verify packet integrity: consecutive beats from same source
        # should maintain first/last framing
        i = 0
        while i < len(results):
            beat = results[i]
            assert beat["first"] == 1, f"Beat {i} should be first: {beat}"
            if beat["last"] == 1:
                i += 1
            else:
                # Multi-beat packet: next beat should be from same source
                i += 1
                while i < len(results) and results[i]["first"] == 0:
                    if results[i]["last"] == 1:
                        i += 1
                        break
                    i += 1

    def test_arbiter_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        sig = Signature(unsigned(8), has_first_last=True)
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamArbiter("not a sig", 2)
        with pytest.raises(ValueError, match="has_first_last"):
            StreamArbiter(Signature(unsigned(8)), 2)
        with pytest.raises(ValueError, match="n must be >= 1"):
            StreamArbiter(sig, 0)


# ===========================================================================
# StreamDispatcher tests
# ===========================================================================

class TestStreamDispatcher:
    """Test StreamDispatcher (packet-aware 1:N dispatcher)."""

    def test_dispatcher_basic(self):
        """Route single-beat packets to different outputs."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamDispatcher(sig, 2)
        results0 = []
        results1 = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send to output 0
            ctx.set(dut.sel, 0)
            await sender.send(ctx, 0xAA, first=1, last=1)
            # Send to output 1
            ctx.set(dut.sel, 1)
            await sender.send(ctx, 0xBB, first=1, last=1)
            # Send to output 0 again
            ctx.set(dut.sel, 0)
            await sender.send(ctx, 0xCC, first=1, last=1)

        async def recv0_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0))
            beat = await recv.recv(ctx)
            results0.append(beat["payload"])
            beat = await recv.recv(ctx)
            results0.append(beat["payload"])

        async def recv1_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(1))
            beat = await recv.recv(ctx)
            results1.append(beat["payload"])

        _run_sim(dut, sender_tb, recv0_tb, recv1_tb,
                 deadline_ns=200_000, vcd_name="test_dispatcher_basic.vcd")

        assert results0 == [0xAA, 0xCC]
        assert results1 == [0xBB]

    def test_dispatcher_packet_hold(self):
        """sel latched during multi-beat packet."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamDispatcher(sig, 2)
        results0 = []
        results1 = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send a 3-beat packet to output 0
            ctx.set(dut.sel, 0)
            await sender.send(ctx, 0xA0, first=1, last=0)
            # Change sel mid-packet — should be ignored
            ctx.set(dut.sel, 1)
            await sender.send(ctx, 0xA1, first=0, last=0)
            await sender.send(ctx, 0xA2, first=0, last=1)
            # Now send to output 1
            ctx.set(dut.sel, 1)
            await sender.send(ctx, 0xB0, first=1, last=1)

        async def recv0_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0))
            for _ in range(3):
                beat = await recv.recv(ctx)
                results0.append(beat["payload"])

        async def recv1_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(1))
            beat = await recv.recv(ctx)
            results1.append(beat["payload"])

        _run_sim(dut, sender_tb, recv0_tb, recv1_tb,
                 deadline_ns=200_000, vcd_name="test_dispatcher_pkt_hold.vcd")

        assert results0 == [0xA0, 0xA1, 0xA2], f"Got: {[hex(r) for r in results0]}"
        assert results1 == [0xB0]

    def test_dispatcher_backpressure(self):
        """Ready from selected output propagates to input."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamDispatcher(sig, 2)

        async def tb(ctx):
            # Select output 0, only output 0 ready
            ctx.set(dut.sel, 0)
            ctx.set(dut.get_output(0).ready, 1)
            ctx.set(dut.get_output(1).ready, 0)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.i_stream.first, 1)
            ctx.set(dut.i_stream.last, 1)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 1

            # Select output 1, only output 0 ready
            ctx.set(dut.sel, 1)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 0

            # Make output 1 ready
            ctx.set(dut.get_output(1).ready, 1)
            _, _, i_ready = await ctx.tick().sample(dut.i_stream.ready)
            assert i_ready == 1

        _run_sim(dut, tb)

    def test_dispatcher_stress(self):
        """Random valid/ready stress test."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamDispatcher(sig, 2)
        results0 = []
        results1 = []
        n_packets = 5

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            rng = random.Random(99)
            for i in range(n_packets * 2):
                sel = i % 2
                ctx.set(dut.sel, sel)
                await sender.send_packet(ctx, [0x10 * sel + i, 0x10 * sel + i + 0x80])

        async def recv0_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(0), random_ready=True, seed=10)
            for _ in range(n_packets):
                pkt = await recv.recv_packet(ctx)
                results0.extend([b["payload"] for b in pkt])

        async def recv1_tb(ctx):
            recv = StreamSimReceiver(dut.get_output(1), random_ready=True, seed=20)
            for _ in range(n_packets):
                pkt = await recv.recv_packet(ctx)
                results1.extend([b["payload"] for b in pkt])

        _run_sim(dut, sender_tb, recv0_tb, recv1_tb,
                 deadline_ns=1_000_000, vcd_name="test_dispatcher_stress.vcd")

        assert len(results0) == n_packets * 2
        assert len(results1) == n_packets * 2

    def test_dispatcher_constructor_validation(self):
        """Constructor rejects invalid arguments."""
        sig = Signature(unsigned(8), has_first_last=True)
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamDispatcher("not a sig", 2)
        with pytest.raises(ValueError, match="has_first_last"):
            StreamDispatcher(Signature(unsigned(8)), 2)
        with pytest.raises(ValueError, match="n must be >= 1"):
            StreamDispatcher(sig, 0)
