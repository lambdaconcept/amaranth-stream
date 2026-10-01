"""Integration tests for amaranth-stream.

Multi-component tests that verify components work together correctly
across module boundaries.
"""

import random

from amaranth import Module
from amaranth.hdl import unsigned
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver
from amaranth_stream.buffer import PipeValid, PipeReady
from amaranth_stream.fifo import StreamFIFO
from amaranth_stream.converter import StreamConverter, Pack
from amaranth_stream.routing import StreamSplitter, StreamJoiner
from amaranth_stream.arbiter import StreamArbiter
from amaranth_stream.pipeline import Pipeline
from amaranth_stream.transform import StreamFilter, EndianSwap
from amaranth_stream.monitor import StreamMonitor
from amaranth_stream.packet import LastInserter, PacketFIFO


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sim(dut, *testbenches, deadline_ns=200_000, vcd_name="test_integration.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


def _connect_streams(src, dst):
    """Return a list of statements connecting two stream interfaces (src → dst)."""
    stmts = [
        dst.payload.eq(src.payload),
        dst.valid.eq(src.valid),
        src.ready.eq(dst.ready),
    ]
    if hasattr(src, "first") and hasattr(dst, "first"):
        stmts.append(dst.first.eq(src.first))
        stmts.append(dst.last.eq(src.last))
    if hasattr(src, "param") and hasattr(dst, "param"):
        stmts.append(dst.param.eq(src.param))
    if hasattr(src, "keep") and hasattr(dst, "keep"):
        stmts.append(dst.keep.eq(src.keep))
    return stmts


# ===========================================================================
# Test 1: Width converter roundtrip (8b → 32b → 8b)
# ===========================================================================


class TestWidthConverterRoundtrip:
    """8b→32b→8b converter roundtrip. Send 8 bytes, upsize to 32b,
    downsize back to 8b, verify all bytes match."""

    def test_width_converter_roundtrip(self):
        sig8 = Signature(unsigned(8))
        sig32 = Signature(unsigned(32))

        upsize = StreamConverter(sig8, sig32)
        downsize = StreamConverter(sig32, sig8)
        dut = Pipeline(upsize, downsize)

        results = []
        expected = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=300_000,
            vcd_name="test_integ_width_roundtrip.vcd",
        )
        assert results == expected, f"Expected {expected}, got {results}"


# ===========================================================================
# Test 2: Pipeline with FIFO and buffer
# ===========================================================================


class TestPipelineWithFifoAndBuffer:
    """Pipeline(PipeValid, StreamFIFO, PipeReady). Send data through,
    verify it arrives correctly."""

    def test_pipeline_with_fifo_and_buffer(self):
        sig = Signature(unsigned(8))

        pv = PipeValid(sig)
        fifo = StreamFIFO(sig, depth=8)
        pr = PipeReady(sig)
        dut = Pipeline(pv, fifo, pr)

        results = []
        expected = list(range(10))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=300_000,
            vcd_name="test_integ_pipeline_fifo_buf.vcd",
        )
        assert results == expected, f"Expected {expected}, got {results}"


# ===========================================================================
# Test 3: Mux/Arbiter pipeline
# ===========================================================================


class TestMuxArbiterPipeline:
    """Two sources → StreamArbiter → StreamFIFO → output.
    Send packets from both sources, verify all arrive."""

    def test_mux_arbiter_pipeline(self):
        sig = Signature(unsigned(8), has_first_last=True)

        # StreamArbiter has i_stream__0, i_stream__1 (not i_stream),
        # so we build a custom DUT instead of using Pipeline.
        class ArbiterFifoDUT(wiring.Component):
            def __init__(self):
                self._arbiter = StreamArbiter(sig, n=2, round_robin=True)
                self._fifo = StreamFIFO(sig, depth=16)
                super().__init__(
                    {
                        "i_stream__0": In(sig),
                        "i_stream__1": In(sig),
                        "o_stream": Out(sig),
                    }
                )

            def elaborate(self, platform):
                m = Module()
                m.submodules.arbiter = self._arbiter
                m.submodules.fifo = self._fifo

                # Inputs → Arbiter
                m.d.comb += _connect_streams(
                    wiring.flipped(self.i_stream__0), self._arbiter.i_stream__0
                )
                m.d.comb += _connect_streams(
                    wiring.flipped(self.i_stream__1), self._arbiter.i_stream__1
                )

                # Arbiter → FIFO
                m.d.comb += _connect_streams(
                    self._arbiter.o_stream, self._fifo.i_stream
                )

                # FIFO → Output
                m.d.comb += _connect_streams(
                    self._fifo.o_stream, wiring.flipped(self.o_stream)
                )

                return m

        dut = ArbiterFifoDUT()
        results = []
        src0_data = [0x10, 0x11, 0x12]  # packet from source 0
        src1_data = [0x20, 0x21, 0x22]  # packet from source 1

        async def sender0_tb(ctx):
            sender = StreamSimSender(dut.i_stream__0)
            await sender.send_packet(ctx, src0_data)

        async def sender1_tb(ctx):
            sender = StreamSimSender(dut.i_stream__1)
            # Small delay so source 0 gets first grant
            await ctx.tick()
            await ctx.tick()
            await sender.send_packet(ctx, src1_data)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(6):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender0_tb,
            sender1_tb,
            receiver_tb,
            deadline_ns=500_000,
            vcd_name="test_integ_arbiter_pipeline.vcd",
        )

        # All 6 values should arrive (order depends on arbitration)
        assert sorted(results) == sorted(src0_data + src1_data), (
            f"Expected all values {sorted(src0_data + src1_data)}, got {sorted(results)}"
        )


# ===========================================================================
# Test 4: Splitter → FIFOs → Joiner roundtrip
# ===========================================================================


class TestSplitterJoinerRoundtrip:
    """StreamSplitter(n=2) → two StreamFIFOs → StreamJoiner(n=2).
    Verify data integrity."""

    def test_splitter_joiner_roundtrip(self):
        sig = Signature(unsigned(8))

        class SplitJoinDUT(wiring.Component):
            def __init__(self):
                self._splitter = StreamSplitter(sig, n=2)
                self._fifo0 = StreamFIFO(sig, depth=8)
                self._fifo1 = StreamFIFO(sig, depth=8)
                self._joiner = StreamJoiner(sig, n=2)
                super().__init__(
                    {
                        "i_stream": In(sig),
                        "o_stream": Out(sig),
                    }
                )

            def elaborate(self, platform):
                m = Module()
                m.submodules.splitter = self._splitter
                m.submodules.fifo0 = self._fifo0
                m.submodules.fifo1 = self._fifo1
                m.submodules.joiner = self._joiner

                # Input → Splitter
                m.d.comb += _connect_streams(
                    wiring.flipped(self.i_stream), self._splitter.i_stream
                )

                # Splitter output 0 → FIFO 0
                m.d.comb += _connect_streams(
                    self._splitter.o_stream__0, self._fifo0.i_stream
                )

                # Splitter output 1 → FIFO 1
                m.d.comb += _connect_streams(
                    self._splitter.o_stream__1, self._fifo1.i_stream
                )

                # FIFO 0 → Joiner input 0
                m.d.comb += _connect_streams(
                    self._fifo0.o_stream, self._joiner.i_stream__0
                )

                # FIFO 1 → Joiner input 1
                m.d.comb += _connect_streams(
                    self._fifo1.o_stream, self._joiner.i_stream__1
                )

                # Joiner → Output
                m.d.comb += _connect_streams(
                    self._joiner.o_stream, wiring.flipped(self.o_stream)
                )

                return m

        dut = SplitJoinDUT()
        results = []
        expected = [0xAA, 0xBB, 0xCC, 0xDD]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Splitter broadcasts to both FIFOs, joiner round-robins.
            # Each input value goes to both FIFOs, so we get 2x values.
            for _ in range(len(expected) * 2):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=500_000,
            vcd_name="test_integ_split_join.vcd",
        )

        # Each value should appear exactly twice (once from each FIFO)
        for val in expected:
            assert results.count(val) == 2, (
                f"Expected value {val:#x} to appear 2 times, got {results.count(val)}"
            )


# ===========================================================================
# Test 5: Filter then Pack
# ===========================================================================


class TestFilterThenPack:
    """StreamFilter (pass even) → Pack(n=4). Filter stream, then pack
    remaining beats."""

    def test_filter_then_pack(self):
        sig = Signature(unsigned(8))

        # Filter: pass only even values
        filt = StreamFilter(sig, predicate=lambda m, p: ~p[0])
        pack = Pack(sig, n=4)
        dut = Pipeline(filt, pack)

        results = []
        # Send 0..15; even values are 0,2,4,6,8,10,12,14 (8 values)
        # Pack(n=4) groups them into 2 wide beats
        input_data = list(range(16))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_data:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # 8 even values / 4 = 2 packed beats
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=500_000,
            vcd_name="test_integ_filter_pack.vcd",
        )

        # First packed beat: [0, 2, 4, 6] = 0x06040200
        # Second packed beat: [8, 10, 12, 14] = 0x0E0C0A08
        assert len(results) == 2
        # Unpack and verify
        beat0_bytes = [(results[0] >> (i * 8)) & 0xFF for i in range(4)]
        beat1_bytes = [(results[1] >> (i * 8)) & 0xFF for i in range(4)]
        assert beat0_bytes == [0, 2, 4, 6], f"Got {beat0_bytes}"
        assert beat1_bytes == [8, 10, 12, 14], f"Got {beat1_bytes}"


# ===========================================================================
# Test 6: EndianSwap roundtrip
# ===========================================================================


class TestEndianSwapRoundtrip:
    """EndianSwap → EndianSwap. Double swap should be identity."""

    def test_endian_swap_roundtrip(self):
        sig = Signature(unsigned(32))

        from amaranth.hdl import ClockDomain

        # Build a custom DUT to avoid duplicate ClockDomain("sync") issues
        class DoubleSwapDUT(wiring.Component):
            def __init__(self):
                self._swap1 = EndianSwap(sig)
                self._swap2 = EndianSwap(sig)
                super().__init__(
                    {
                        "i_stream": In(sig),
                        "o_stream": Out(sig),
                    }
                )

            def elaborate(self, platform):
                m = Module()
                m.domains += ClockDomain("sync")
                m.submodules.swap1 = self._swap1
                m.submodules.swap2 = self._swap2

                # Input → Swap1
                m.d.comb += _connect_streams(
                    wiring.flipped(self.i_stream), self._swap1.i_stream
                )

                # Swap1 → Swap2
                m.d.comb += _connect_streams(self._swap1.o_stream, self._swap2.i_stream)

                # Swap2 → Output
                m.d.comb += _connect_streams(
                    self._swap2.o_stream, wiring.flipped(self.o_stream)
                )

                return m

        dut = DoubleSwapDUT()
        results = []
        expected = [0xDEADBEEF, 0x12345678, 0xCAFEBABE, 0x00FF00FF]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=200_000,
            vcd_name="test_integ_endian_roundtrip.vcd",
        )
        assert results == expected, f"Expected {expected}, got {results}"


# ===========================================================================
# Test 7: Monitor in pipeline
# ===========================================================================


class TestMonitorInPipeline:
    """StreamMonitor in a pipeline. Verify counters while data flows."""

    def test_monitor_in_pipeline(self):
        sig = Signature(unsigned(8))

        monitor = StreamMonitor(sig)
        dut = Pipeline(monitor)

        results = []
        expected = list(range(10))
        transfer_count_final = [0]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])
            # Read the transfer count after all transfers
            # Wait one more tick for counter to update
            await ctx.tick()
            _, _, tc = await ctx.tick().sample(monitor.transfer_count)
            transfer_count_final[0] = tc

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=300_000,
            vcd_name="test_integ_monitor_pipeline.vcd",
        )
        assert results == expected, f"Expected {expected}, got {results}"
        assert transfer_count_final[0] == len(expected), (
            f"Expected transfer_count={len(expected)}, got {transfer_count_final[0]}"
        )


# ===========================================================================
# Test 8: Full packet pipeline (LastInserter → PacketFIFO → output)
# ===========================================================================


class TestFullPacketPipeline:
    """LastInserter → PacketFIFO → output. Create fixed-size packets,
    buffer atomically, read out."""

    def test_full_packet_pipeline(self):
        sig = Signature(unsigned(8), has_first_last=True)

        inserter = LastInserter(sig, n=4)  # 4-beat packets
        pkt_fifo = PacketFIFO(sig, payload_depth=32, packet_depth=8)
        dut = Pipeline(inserter, pkt_fifo)

        results = []
        # Send 12 beats → 3 packets of 4 beats each
        input_data = list(range(12))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_data:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in input_data:
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=500_000,
            vcd_name="test_integ_packet_pipeline.vcd",
        )

        # Verify all payloads arrived
        payloads = [b["payload"] for b in results]
        assert payloads == input_data, f"Expected {input_data}, got {payloads}"

        # Verify packet framing: first on beats 0,4,8; last on beats 3,7,11
        for i, beat in enumerate(results):
            expected_first = 1 if (i % 4 == 0) else 0
            expected_last = 1 if (i % 4 == 3) else 0
            assert beat["first"] == expected_first, (
                f"Beat {i}: expected first={expected_first}, got {beat['first']}"
            )
            assert beat["last"] == expected_last, (
                f"Beat {i}: expected last={expected_last}, got {beat['last']}"
            )


# ===========================================================================
# Test 9: Stress multi-component pipeline
# ===========================================================================


class TestStressMultiComponent:
    """Complex pipeline with random valid/ready:
    PipeValid → StreamFIFO → PipeReady. 100 transfers with random stalls."""

    def test_stress_multi_component(self):
        sig = Signature(unsigned(8))

        pv = PipeValid(sig)
        fifo = StreamFIFO(sig, depth=16)
        pr = PipeReady(sig)
        dut = Pipeline(pv, fifo, pr)

        results = []
        rng = random.Random(12345)
        expected = [rng.randint(0, 255) for _ in range(100)]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=111)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=222)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=2_000_000,
            vcd_name="test_integ_stress.vcd",
        )
        assert results == expected, (
            f"Mismatch at {len(results)} results vs {len(expected)} expected"
        )
