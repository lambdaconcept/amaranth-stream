"""Tests for amaranth_stream.sim (StreamSimSender, StreamSimReceiver).

These tests use a simple passthrough component that connects a source stream
directly to a sink stream, exercising the BFMs with the Amaranth simulator.
"""

import pytest

from amaranth.hdl import Module, Signal, ClockDomain, unsigned
from amaranth.lib.wiring import Component, In, Out, connect
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Passthrough DUT
# ---------------------------------------------------------------------------

class Passthrough(Component):
    """Trivial passthrough: connects i_stream directly to o_stream in comb.

    Includes a sync domain so the simulator can add a clock.
    """

    def __init__(self, payload_width=8, *, has_first_last=False, param_shape=None, has_keep=False):
        sig = Signature(unsigned(payload_width),
                        has_first_last=has_first_last,
                        param_shape=param_shape,
                        has_keep=has_keep)
        members = {
            "i_stream": In(sig),
            "o_stream": Out(sig),
        }
        super().__init__(members)

    def elaborate(self, platform):
        m = Module()
        # Ensure the sync domain exists so the simulator can add a clock
        m.domains += ClockDomain("sync")

        # Wire input to output combinationally
        m.d.comb += [
            self.o_stream.payload.eq(self.i_stream.payload),
            self.o_stream.valid.eq(self.i_stream.valid),
            self.i_stream.ready.eq(self.o_stream.ready),
        ]
        if hasattr(self.i_stream, "first"):
            m.d.comb += [
                self.o_stream.first.eq(self.i_stream.first),
                self.o_stream.last.eq(self.i_stream.last),
            ]
        if hasattr(self.i_stream, "param"):
            m.d.comb += self.o_stream.param.eq(self.i_stream.param)
        if hasattr(self.i_stream, "keep"):
            m.d.comb += self.o_stream.keep.eq(self.i_stream.keep)
        return m


def _run_sim(dut, testbench, *, deadline_ns=10_000):
    """Helper to run a simulation with a single testbench."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    sim.add_testbench(testbench)
    with sim.write_vcd("test_sim.vcd"):
        sim.run_until(Period(ns=deadline_ns))


# ---------------------------------------------------------------------------
# Basic send / recv
# ---------------------------------------------------------------------------

class TestBasicSendRecv:
    """Test basic single-beat send and receive."""

    def test_single_beat(self):
        dut = Passthrough(8)

        async def testbench(ctx):
            # Drive payload and valid
            ctx.set(dut.i_stream.payload, 0xAB)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.o_stream.ready, 1)
            await ctx.tick()

            # After tick, both valid and ready were high => transfer happened
            payload = ctx.get(dut.o_stream.payload)
            assert payload == 0xAB

        _run_sim(dut, testbench)

    def test_sender_recv_bfm(self):
        """Test using BFMs with concurrent testbenches."""
        dut = Passthrough(8)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x42)
            await sender.send(ctx, 0xFF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1["payload"])
            beat2 = await receiver.recv(ctx)
            results.append(beat2["payload"])

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_bfm.vcd"):
            sim.run_until(Period(ns=10_000))

        assert results == [0x42, 0xFF]

    def test_multiple_beats(self):
        """Send and receive several beats."""
        dut = Passthrough(8)
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

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_multi.vcd"):
            sim.run_until(Period(ns=50_000))

        assert results == expected


# ---------------------------------------------------------------------------
# Packet send / recv
# ---------------------------------------------------------------------------

class TestPacketSendRecv:
    """Test send_packet / recv_packet with first/last framing."""

    def test_send_recv_packet(self):
        dut = Passthrough(8, has_first_last=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_packet.vcd"):
            sim.run_until(Period(ns=50_000))

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

    def test_single_beat_packet(self):
        """A single-beat packet has both first=1 and last=1."""
        dut = Passthrough(8, has_first_last=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0xAA])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_single_packet.vcd"):
            sim.run_until(Period(ns=10_000))

        assert len(results) == 1
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1


# ---------------------------------------------------------------------------
# expect_packet
# ---------------------------------------------------------------------------

class TestExpectPacket:
    """Test the expect_packet convenience method."""

    def test_expect_packet_pass(self):
        dut = Passthrough(8, has_first_last=True)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x01, 0x02, 0x03])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            await receiver.expect_packet(ctx, [0x01, 0x02, 0x03])

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_expect_pass.vcd"):
            sim.run_until(Period(ns=50_000))

    def test_expect_packet_fail_payload(self):
        """expect_packet should raise AssertionError on payload mismatch."""
        dut = Passthrough(8, has_first_last=True)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x01, 0x99, 0x03])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            with pytest.raises(AssertionError, match="Beat 1"):
                await receiver.expect_packet(ctx, [0x01, 0x02, 0x03])

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_expect_fail.vcd"):
            sim.run_until(Period(ns=50_000))


# ---------------------------------------------------------------------------
# Random valid / ready stress
# ---------------------------------------------------------------------------

class TestRandomStress:
    """Test random_valid and random_ready modes."""

    def test_random_valid(self):
        """Sender randomly delays valid; data should still arrive correctly."""
        dut = Passthrough(8)
        results = []
        expected = list(range(10))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_random_valid.vcd"):
            sim.run_until(Period(ns=100_000))

        assert results == expected

    def test_random_ready(self):
        """Receiver randomly delays ready; data should still arrive correctly."""
        dut = Passthrough(8)
        results = []
        expected = list(range(10))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=42)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_random_ready.vcd"):
            sim.run_until(Period(ns=100_000))

        assert results == expected

    def test_random_both(self):
        """Both sender and receiver randomly delay; data should still arrive."""
        dut = Passthrough(8)
        results = []
        expected = list(range(20))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=123)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=456)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        sim = Simulator(dut)
        sim.add_clock(Period(MHz=10))
        sim.add_testbench(sender_tb)
        sim.add_testbench(receiver_tb)
        with sim.write_vcd("test_random_both.vcd"):
            sim.run_until(Period(ns=200_000))

        assert results == expected
