"""Tests for amaranth_stream.fifo and amaranth_stream.cdc."""

import random
import pytest

from amaranth.hdl import unsigned
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.fifo import StreamFIFO, StreamAsyncFIFO
from amaranth_stream.cdc import StreamCDC
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_fifo.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


def _run_async_sim(dut, testbenches, *,
                   clocks=None, deadline_ns=100_000, vcd_name="test_async_fifo.vcd"):
    """Helper to run a simulation with multiple clock domains.

    Parameters
    ----------
    dut : Elaboratable
    testbenches : list of testbench functions
    clocks : dict of {domain_name: period}
    """
    sim = Simulator(dut)
    if clocks is None:
        clocks = {}
    for domain, period in clocks.items():
        sim.add_clock(period, domain=domain)
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


# ===========================================================================
# StreamFIFO tests
# ===========================================================================

class TestStreamFIFO:
    """Test StreamFIFO (synchronous FIFO with stream interfaces)."""

    def test_sync_fifo_basic(self):
        """Write and read data through a buffered FIFO."""
        sig = Signature(unsigned(8))
        dut = StreamFIFO(sig, depth=4, buffered=True)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_sync_fifo_basic.vcd")
        assert results == expected

    def test_sync_fifo_buffered(self):
        """Buffered mode (registered output) passes data correctly."""
        sig = Signature(unsigned(8))
        dut = StreamFIFO(sig, depth=8, buffered=True)
        results = []
        expected = list(range(8))

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
                 deadline_ns=100_000, vcd_name="test_sync_fifo_buffered.vcd")
        assert results == expected

    def test_sync_fifo_unbuffered(self):
        """Unbuffered mode (combinational output) passes data correctly."""
        sig = Signature(unsigned(8))
        dut = StreamFIFO(sig, depth=4, buffered=False)
        results = []
        expected = [0x10, 0x20, 0x30, 0x40]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_sync_fifo_unbuffered.vcd")
        assert results == expected

    def test_sync_fifo_full(self):
        """FIFO fills up and backpressures the sender."""
        sig = Signature(unsigned(8))
        depth = 4
        dut = StreamFIFO(sig, depth=depth, buffered=True)
        results = []
        # Send more data than the FIFO can hold at once
        expected = list(range(8))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Add some delay before starting to read, so FIFO fills up
            for _ in range(6):
                await ctx.tick()
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_sync_fifo_full.vcd")
        assert results == expected

    def test_sync_fifo_level(self):
        """Level signal tracks occupancy correctly."""
        sig = Signature(unsigned(8))
        depth = 4
        dut = StreamFIFO(sig, depth=depth, buffered=False)
        levels = []

        async def testbench(ctx):
            sender = StreamSimSender(dut.i_stream)
            receiver = StreamSimReceiver(dut.o_stream)

            # Initially empty
            await ctx.tick()
            levels.append(ctx.get(dut.level))

            # Write 3 entries
            await sender.send(ctx, 0x01)
            await sender.send(ctx, 0x02)
            await sender.send(ctx, 0x03)

            # Check level after writes
            await ctx.tick()
            levels.append(ctx.get(dut.level))

            # Read one entry
            beat = await receiver.recv(ctx)
            assert beat["payload"] == 0x01

            # Check level after read
            await ctx.tick()
            levels.append(ctx.get(dut.level))

        _run_sim(dut, testbench, deadline_ns=100_000, vcd_name="test_sync_fifo_level.vcd")
        assert levels[0] == 0  # initially empty
        assert levels[1] == 3  # after 3 writes
        assert levels[2] == 2  # after 1 read

    def test_sync_fifo_with_first_last(self):
        """first/last signals pass through the FIFO correctly."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamFIFO(sig, depth=4, buffered=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_sync_fifo_first_last.vcd")

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

    def test_sync_fifo_with_param(self):
        """param signal passes through the FIFO correctly."""
        sig = Signature(unsigned(8), param_shape=unsigned(4))
        dut = StreamFIFO(sig, depth=4, buffered=True)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_sync_fifo_param.vcd")

        assert results[0]["payload"] == 0xAA
        assert results[0]["param"] == 0x5
        assert results[1]["payload"] == 0xBB
        assert results[1]["param"] == 0xA

    def test_sync_fifo_with_keep(self):
        """keep signal passes through the FIFO correctly."""
        sig = Signature(unsigned(16), has_keep=True)
        dut = StreamFIFO(sig, depth=4, buffered=True)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_sync_fifo_keep.vcd")

        assert results[0]["payload"] == 0x1234
        assert results[0]["keep"] == 0x3
        assert results[1]["payload"] == 0x5678
        assert results[1]["keep"] == 0x1

    def test_sync_fifo_all_optional(self):
        """All optional signals (first, last, param, keep) pass through."""
        sig = Signature(unsigned(8), has_first_last=True,
                        param_shape=unsigned(4), has_keep=True)
        dut = StreamFIFO(sig, depth=4, buffered=True)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_sync_fifo_all_opt.vcd")

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

    def test_sync_fifo_stress(self):
        """Random valid/ready stress test through FIFO."""
        sig = Signature(unsigned(8))
        dut = StreamFIFO(sig, depth=8, buffered=True)
        results = []
        expected = [random.Random(55).randint(0, 255) for _ in range(50)]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=300)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=400)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_sync_fifo_stress.vcd")
        assert results == expected

    def test_sync_fifo_stress_unbuffered(self):
        """Random valid/ready stress test through unbuffered FIFO."""
        sig = Signature(unsigned(8))
        dut = StreamFIFO(sig, depth=8, buffered=False)
        results = []
        expected = [random.Random(56).randint(0, 255) for _ in range(50)]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=301)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=401)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_sync_fifo_stress_unbuf.vcd")
        assert results == expected


# ===========================================================================
# StreamFIFO constructor validation
# ===========================================================================

class TestStreamFIFOValidation:
    """Test StreamFIFO constructor parameter validation."""

    def test_rejects_non_signature(self):
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamFIFO("not a signature", depth=4)

    def test_rejects_negative_depth(self):
        sig = Signature(unsigned(8))
        with pytest.raises(ValueError, match="non-negative"):
            StreamFIFO(sig, depth=-1)


# ===========================================================================
# StreamAsyncFIFO tests
# ===========================================================================

class TestStreamAsyncFIFO:
    """Test StreamAsyncFIFO (asynchronous CDC FIFO with stream interfaces)."""

    def test_async_fifo_basic(self):
        """Basic cross-domain transfer through async FIFO."""
        sig = Signature(unsigned(8))
        dut = StreamAsyncFIFO(sig, depth=4, w_domain="fast", r_domain="slow")
        results = []
        expected = [0x01, 0x02, 0x03, 0x04]

        async def writer(ctx):
            sender = StreamSimSender(dut.i_stream, domain="fast")
            for val in expected:
                await sender.send(ctx, val)

        async def reader(ctx):
            receiver = StreamSimReceiver(dut.o_stream, domain="slow")
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_async_sim(
            dut,
            [writer, reader],
            clocks={"fast": Period(MHz=10), "slow": Period(MHz=7)},
            deadline_ns=200_000,
            vcd_name="test_async_fifo_basic.vcd",
        )
        assert results == expected

    def test_async_fifo_buffered(self):
        """Buffered async FIFO passes data correctly."""
        sig = Signature(unsigned(8))
        dut = StreamAsyncFIFO(sig, depth=8, w_domain="fast", r_domain="slow", buffered=True)
        results = []
        expected = list(range(8))

        async def writer(ctx):
            sender = StreamSimSender(dut.i_stream, domain="fast")
            for val in expected:
                await sender.send(ctx, val)

        async def reader(ctx):
            receiver = StreamSimReceiver(dut.o_stream, domain="slow")
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_async_sim(
            dut,
            [writer, reader],
            clocks={"fast": Period(MHz=12), "slow": Period(MHz=8)},
            deadline_ns=200_000,
            vcd_name="test_async_fifo_buffered.vcd",
        )
        assert results == expected

    def test_async_fifo_unbuffered(self):
        """Unbuffered async FIFO passes data correctly."""
        sig = Signature(unsigned(8))
        dut = StreamAsyncFIFO(sig, depth=4, w_domain="fast", r_domain="slow", buffered=False)
        results = []
        expected = [0xAA, 0xBB, 0xCC, 0xDD]

        async def writer(ctx):
            sender = StreamSimSender(dut.i_stream, domain="fast")
            for val in expected:
                await sender.send(ctx, val)

        async def reader(ctx):
            receiver = StreamSimReceiver(dut.o_stream, domain="slow")
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_async_sim(
            dut,
            [writer, reader],
            clocks={"fast": Period(MHz=10), "slow": Period(MHz=7)},
            deadline_ns=200_000,
            vcd_name="test_async_fifo_unbuffered.vcd",
        )
        assert results == expected

    def test_async_fifo_with_first_last(self):
        """first/last signals pass through async FIFO."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamAsyncFIFO(sig, depth=4, w_domain="fast", r_domain="slow")
        results = []

        async def writer(ctx):
            sender = StreamSimSender(dut.i_stream, domain="fast")
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def reader(ctx):
            receiver = StreamSimReceiver(dut.o_stream, domain="slow")
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_async_sim(
            dut,
            [writer, reader],
            clocks={"fast": Period(MHz=10), "slow": Period(MHz=7)},
            deadline_ns=200_000,
            vcd_name="test_async_fifo_first_last.vcd",
        )

        assert len(results) == 3
        assert results[0]["payload"] == 0x10
        assert results[0]["first"] == 1
        assert results[2]["payload"] == 0x30
        assert results[2]["last"] == 1

    def test_async_fifo_stress(self):
        """Random valid/ready stress test through async FIFO."""
        sig = Signature(unsigned(8))
        dut = StreamAsyncFIFO(sig, depth=8, w_domain="fast", r_domain="slow")
        results = []
        expected = [random.Random(77).randint(0, 255) for _ in range(20)]

        async def writer(ctx):
            sender = StreamSimSender(dut.i_stream, domain="fast",
                                     random_valid=True, seed=500)
            for val in expected:
                await sender.send(ctx, val)

        async def reader(ctx):
            receiver = StreamSimReceiver(dut.o_stream, domain="slow",
                                         random_ready=True, seed=600)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_async_sim(
            dut,
            [writer, reader],
            clocks={"fast": Period(MHz=10), "slow": Period(MHz=7)},
            deadline_ns=500_000,
            vcd_name="test_async_fifo_stress.vcd",
        )
        assert results == expected


# ===========================================================================
# StreamAsyncFIFO constructor validation
# ===========================================================================

class TestStreamAsyncFIFOValidation:
    """Test StreamAsyncFIFO constructor parameter validation."""

    def test_rejects_non_signature(self):
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamAsyncFIFO("not a signature", depth=4)

    def test_rejects_negative_depth(self):
        sig = Signature(unsigned(8))
        with pytest.raises(ValueError, match="non-negative"):
            StreamAsyncFIFO(sig, depth=-1)


# ===========================================================================
# StreamCDC tests
# ===========================================================================

class TestStreamCDC:
    """Test StreamCDC (automatic CDC selection)."""

    def test_cdc_same_domain(self):
        """Same domain uses Buffer internally."""
        sig = Signature(unsigned(8))
        dut = StreamCDC(sig, w_domain="sync", r_domain="sync")
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cdc_same.vcd")
        assert results == expected

    def test_cdc_different_domain(self):
        """Different domains use StreamAsyncFIFO internally."""
        sig = Signature(unsigned(8))
        dut = StreamCDC(sig, depth=8, w_domain="fast", r_domain="slow")
        results = []
        expected = [0x01, 0x02, 0x03, 0x04]

        async def writer(ctx):
            sender = StreamSimSender(dut.i_stream, domain="fast")
            for val in expected:
                await sender.send(ctx, val)

        async def reader(ctx):
            receiver = StreamSimReceiver(dut.o_stream, domain="slow")
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_async_sim(
            dut,
            [writer, reader],
            clocks={"fast": Period(MHz=10), "slow": Period(MHz=7)},
            deadline_ns=200_000,
            vcd_name="test_cdc_different.vcd",
        )
        assert results == expected

    def test_cdc_same_domain_with_first_last(self):
        """Same domain CDC preserves first/last signals."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamCDC(sig, w_domain="sync", r_domain="sync")
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cdc_same_fl.vcd")

        assert len(results) == 3
        assert results[0]["first"] == 1
        assert results[2]["last"] == 1

    def test_cdc_same_domain_latency(self):
        """Same-domain StreamCDC has 1-cycle latency (PipeValid)."""
        sig = Signature(unsigned(8))
        dut = StreamCDC(sig, w_domain="sync", r_domain="sync")
        latency_cycles = []

        async def testbench(ctx):
            # Hold ready high so data flows through immediately
            ctx.set(dut.o_stream.ready, 1)
            await ctx.tick()

            # Assert valid + payload on input
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.i_stream.payload, 0x42)
            await ctx.tick()

            # After 1 cycle, output should be valid (PipeValid = 1-cycle latency)
            o_valid = ctx.get(dut.o_stream.valid)
            o_payload = ctx.get(dut.o_stream.payload)
            latency_cycles.append((o_valid, o_payload))

        _run_sim(dut, testbench, vcd_name="test_cdc_same_latency.vcd")

        # PipeValid: data appears after exactly 1 cycle
        assert latency_cycles[0] == (1, 0x42), (
            f"Expected (valid=1, payload=0x42) after 1 cycle, "
            f"got {latency_cycles[0]}"
        )

    def test_cdc_same_domain_data(self):
        """Same-domain StreamCDC passes data correctly."""
        sig = Signature(unsigned(8))
        dut = StreamCDC(sig, w_domain="sync", r_domain="sync")
        results = []
        expected = [0x01, 0x02, 0x03, 0x04, 0x55, 0xAA, 0xFF, 0x00]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cdc_same_data.vcd")
        assert results == expected

    def test_cdc_same_domain_backpressure(self):
        """Same-domain StreamCDC handles backpressure correctly."""
        sig = Signature(unsigned(8))
        dut = StreamCDC(sig, w_domain="sync", r_domain="sync")
        results = []
        expected = [random.Random(88).randint(0, 255) for _ in range(30)]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=700)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=800)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_cdc_same_backpressure.vcd")
        assert results == expected


# ===========================================================================
# StreamCDC constructor validation
# ===========================================================================

class TestStreamCDCValidation:
    """Test StreamCDC constructor parameter validation."""

    def test_rejects_non_signature(self):
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamCDC("not a signature")
