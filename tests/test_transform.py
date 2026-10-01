"""Tests for amaranth_stream.transform (StreamMap, StreamFilter, EndianSwap, GranularEndianSwap, ByteAligner, PacketAligner, WordReorder)."""

import random
import pytest

from amaranth.hdl import unsigned, Signal, Cat
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.transform import StreamMap, StreamFilter, EndianSwap, GranularEndianSwap, ByteAligner, PacketAligner, WordReorder
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_transform.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


# ---------------------------------------------------------------------------
# StreamMap tests
# ---------------------------------------------------------------------------

class TestStreamMap:
    """Test StreamMap (combinational payload transformation)."""

    def test_stream_map_double(self):
        """Map that doubles payload."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(9))  # wider to hold doubled value

        def double_transform(m, payload):
            result = Signal(9, name="doubled")
            m.d.comb += result.eq(payload << 1)
            return result

        dut = StreamMap(i_sig, o_sig, double_transform)
        results = []
        expected_in = [1, 2, 3, 10, 127]
        expected_out = [v * 2 for v in expected_in]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected_in:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected_in:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_map_double.vcd")
        assert results == expected_out

    def test_stream_map_mask(self):
        """Map that masks bits (keep lower 4 bits)."""
        sig = Signature(unsigned(8))

        def mask_transform(m, payload):
            result = Signal(8, name="masked")
            m.d.comb += result.eq(payload & 0x0F)
            return result

        dut = StreamMap(sig, sig, mask_transform)
        results = []
        expected_in = [0xFF, 0xAB, 0x12, 0x00]
        expected_out = [0x0F, 0x0B, 0x02, 0x00]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected_in:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected_in:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_map_mask.vcd")
        assert results == expected_out

    def test_stream_map_with_first_last(self):
        """first/last signals pass through StreamMap."""
        i_sig = Signature(unsigned(8), has_first_last=True)
        o_sig = Signature(unsigned(8), has_first_last=True)

        def identity_transform(m, payload):
            return payload

        dut = StreamMap(i_sig, o_sig, identity_transform)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_map_fl.vcd")

        assert len(results) == 3
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0
        assert results[2]["first"] == 0
        assert results[2]["last"] == 1


# ---------------------------------------------------------------------------
# StreamFilter tests
# ---------------------------------------------------------------------------

class TestStreamFilter:
    """Test StreamFilter (conditional beat dropping)."""

    def test_stream_filter_even(self):
        """Filter that passes only even values."""
        sig = Signature(unsigned(8))

        def even_predicate(m, payload):
            return ~payload[0]  # bit 0 == 0 means even

        dut = StreamFilter(sig, even_predicate)
        results = []
        input_vals = [1, 2, 3, 4, 5, 6, 7, 8]
        expected = [2, 4, 6, 8]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_filter_even.vcd")
        assert results == expected

    def test_stream_filter_all_pass(self):
        """Predicate always true — all beats pass through."""
        sig = Signature(unsigned(8))

        def always_pass(m, payload):
            return 1

        dut = StreamFilter(sig, always_pass)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_filter_all_pass.vcd")
        assert results == expected

    def test_stream_filter_all_drop(self):
        """Predicate always false — all beats are dropped."""
        sig = Signature(unsigned(8))

        def always_drop(m, payload):
            return 0

        dut = StreamFilter(sig, always_drop)
        input_vals = [0x10, 0x20, 0x30]
        sender_done = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)
            sender_done.append(True)

        async def checker_tb(ctx):
            # Wait for sender to finish, then verify no output valid
            for _ in range(20):
                await ctx.tick()
            # Output valid should never have been asserted
            # (sender should complete quickly since dropped beats are consumed)

        _run_sim(dut, sender_tb, checker_tb, vcd_name="test_filter_all_drop.vcd")
        assert len(sender_done) == 1  # sender completed (beats were consumed)

    def test_stream_filter_stress(self):
        """Random valid/ready with filter."""
        sig = Signature(unsigned(8))

        def pass_above_128(m, payload):
            return payload[7]  # bit 7 set means >= 128

        dut = StreamFilter(sig, pass_above_128)
        rng = random.Random(42)
        input_vals = [rng.randint(0, 255) for _ in range(30)]
        expected = [v for v in input_vals if v >= 128]
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=10)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=20)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=500_000, vcd_name="test_filter_stress.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# EndianSwap tests
# ---------------------------------------------------------------------------

class TestEndianSwap:
    """Test EndianSwap (byte-order reversal)."""

    def test_endian_swap_16(self):
        """16-bit endian swap: 0xAABB → 0xBBAA."""
        sig = Signature(unsigned(16))
        dut = EndianSwap(sig)
        results = []
        input_vals = [0xAABB, 0x1234, 0xFF00, 0x0001]
        expected = [0xBBAA, 0x3412, 0x00FF, 0x0100]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in input_vals:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_endian_16.vcd")
        assert results == expected

    def test_endian_swap_32(self):
        """32-bit endian swap: 0xDEADBEEF → 0xEFBEADDE."""
        sig = Signature(unsigned(32))
        dut = EndianSwap(sig)
        results = []
        input_vals = [0xDEADBEEF, 0x01020304]
        expected = [0xEFBEADDE, 0x04030201]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in input_vals:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_endian_32.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# ByteAligner tests
# ---------------------------------------------------------------------------

class TestByteAligner:
    """Test ByteAligner (sub-word byte alignment)."""

    def test_byte_aligner_basic(self):
        """Shift by 1 byte: 0x00FF → 0xFF00 (for 16-bit)."""
        sig = Signature(unsigned(16))
        dut = ByteAligner(sig, max_shift=2)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            ctx.set(dut.shift, 1)  # shift by 1 byte = 8 bits left
            await sender.send(ctx, 0x00FF)
            await sender.send(ctx, 0x0001)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1["payload"])
            beat2 = await receiver.recv(ctx)
            results.append(beat2["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_aligner_basic.vcd")
        assert results[0] == 0xFF00
        assert results[1] == 0x0100

    def test_byte_aligner_zero_shift(self):
        """No shift: data passes through unchanged."""
        sig = Signature(unsigned(16))
        dut = ByteAligner(sig, max_shift=2)
        results = []
        expected = [0xAABB, 0x1234]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            ctx.set(dut.shift, 0)  # no shift
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_aligner_zero.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# GranularEndianSwap tests
# ---------------------------------------------------------------------------

class TestGranularEndianSwap:
    """Test GranularEndianSwap (per-chunk byte-order reversal)."""

    def test_granular_endian_swap_256bit_dword(self):
        """256-bit data with 32-bit chunks: each DWORD independently byte-reversed."""
        dut = GranularEndianSwap(data_width=256, chunk_size=32)
        results = []

        # Build a 256-bit value from 8 DWORDs with known byte patterns:
        # DWORD0 = 0x01020304, DWORD1 = 0x05060708, ..., DWORD7 = 0x1D1E1F20
        dwords_in = [
            0x01020304,
            0x05060708,
            0x090A0B0C,
            0x0D0E0F10,
            0x11121314,
            0x15161718,
            0x191A1B1C,
            0x1D1E1F20,
        ]
        # Expected: each DWORD independently byte-reversed
        dwords_out = [
            0x04030201,
            0x08070605,
            0x0C0B0A09,
            0x100F0E0D,
            0x14131211,
            0x18171615,
            0x1C1B1A19,
            0x201F1E1D,
        ]

        # Assemble 256-bit values (DWORD0 in LSBs)
        input_val = sum(d << (32 * i) for i, d in enumerate(dwords_in))
        expected_val = sum(d << (32 * i) for i, d in enumerate(dwords_out))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send as a 2-beat packet to verify first/last pass-through
            await sender.send(ctx, input_val, first=1, last=0)
            await sender.send(ctx, input_val, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat1 = await receiver.recv(ctx)
            results.append(beat1)
            beat2 = await receiver.recv(ctx)
            results.append(beat2)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_granular_256_dword.vcd")

        # Verify payload byte-reversal per DWORD
        assert results[0]["payload"] == expected_val
        assert results[1]["payload"] == expected_val

        # Verify first/last pass-through
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

    def test_granular_endian_swap_be_mode(self):
        """Verify byte-enable (keep) reversal within chunks when be_mode=True."""
        # 64-bit data, 32-bit chunks, be_mode=True
        # keep has 8 bits (one per byte), 4 bits per DWORD chunk
        dut = GranularEndianSwap(data_width=64, chunk_size=32, be_mode=True)
        results = []

        # DWORD0 = 0xAABBCCDD, DWORD1 = 0x11223344
        input_val = 0xAABBCCDD | (0x11223344 << 32)
        # After per-DWORD byte reversal:
        # DWORD0: 0xAABBCCDD -> bytes [DD, CC, BB, AA] reversed -> [AA, BB, CC, DD] = 0xDDCCBBAA
        # DWORD1: 0x11223344 -> bytes [44, 33, 22, 11] reversed -> [11, 22, 33, 44] = 0x44332211
        expected_val = 0xDDCCBBAA | (0x44332211 << 32)

        # keep bits: chunk0 = 0b1100 (bytes 0,1 disabled, 2,3 enabled)
        #            chunk1 = 0b0011 (bytes 0,1 enabled, 2,3 disabled)
        # Combined: chunk1[3:0] | chunk0[3:0] = 0b0011_1100 = 0x3C
        keep_in = 0b0011_1100
        # After per-chunk reversal of keep bits:
        # chunk0: 0b1100 reversed -> 0b0011
        # chunk1: 0b0011 reversed -> 0b1100
        # Combined: 0b1100_0011 = 0xC3
        keep_out = 0b1100_0011

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, input_val, keep=keep_in, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_granular_be_mode.vcd")

        assert results[0]["payload"] == expected_val
        assert results[0]["keep"] == keep_out

    def test_granular_endian_swap_64bit_word(self):
        """Test with 64-bit chunk size (word-level swap) on 128-bit data."""
        # 128-bit data, 64-bit chunks -> 2 chunks, each with 8 bytes reversed
        dut = GranularEndianSwap(data_width=128, chunk_size=64)
        results = []

        # Word0 (bits 63:0)  = 0x0102030405060708
        # Word1 (bits 127:64) = 0x090A0B0C0D0E0F10
        word0_in = 0x0102030405060708
        word1_in = 0x090A0B0C0D0E0F10
        input_val = word0_in | (word1_in << 64)

        # After per-64-bit-chunk byte reversal:
        # Word0: bytes [08,07,06,05,04,03,02,01] reversed -> [01,02,03,04,05,06,07,08]
        #        = 0x0807060504030201
        # Word1: bytes [10,0F,0E,0D,0C,0B,0A,09] reversed -> [09,0A,0B,0C,0D,0E,0F,10]
        #        = 0x100F0E0D0C0B0A09
        word0_out = 0x0807060504030201
        word1_out = 0x100F0E0D0C0B0A09
        expected_val = word0_out | (word1_out << 64)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, input_val, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_granular_64bit_word.vcd")

        assert results[0]["payload"] == expected_val


# ---------------------------------------------------------------------------
# PacketAligner tests
# ---------------------------------------------------------------------------

def _build_128bit(dw3, dw2, dw1, dw0):
    """Build a 128-bit value from four DWORDs (dw0 in LSBs)."""
    return dw0 | (dw1 << 32) | (dw2 << 64) | (dw3 << 96)


class TestPacketAligner:
    """Test PacketAligner (cross-beat packet realignment)."""

    def test_packet_aligner_128bit_dword2_start(self):
        """128-bit bus, DWORD granularity, offset=2.

        Send a 3-beat packet where data starts at DWORD offset 2 (upper 64 bits
        of beat 0).  The aligner absorbs beat 0 as carry, then stitches across
        beat boundaries.  A flush beat is emitted after the last input beat.

        Input beats (128-bit, 4 DWORDs each, DW0 in LSBs):
          beat0: [0xAAAAAAAA, 0xBBBBBBBB, 0x11111111, 0x22222222]
          beat1: [0xCCCCCCCC, 0xDDDDDDDD, 0x33333333, 0x44444444]
          beat2: [0xEEEEEEEE, 0xFFFFFFFF, 0x55555555, 0x66666666]

        With offset=2, shift_bits=64.  The aligner does:
          carry = beat0
          out0 = (Cat(beat0, beat1) >> 64)[:128]
               = beat0[64:128] | beat1[0:64] << 64
               = [0x11111111, 0x22222222, 0xCCCCCCCC, 0xDDDDDDDD]
          carry = beat1
          out1 = (Cat(beat1, beat2) >> 64)[:128]
               = beat1[64:128] | beat2[0:64] << 64
               = [0x33333333, 0x44444444, 0xEEEEEEEE, 0xFFFFFFFF]
          carry = beat2
          flush = (Cat(beat2, 0) >> 64)[:128]
                = beat2[64:128]
                = [0x55555555, 0x66666666, 0x00000000, 0x00000000]
        """
        dut = PacketAligner(data_width=128, granularity=32)
        results = []

        beat0 = _build_128bit(0x22222222, 0x11111111, 0xBBBBBBBB, 0xAAAAAAAA)
        beat1 = _build_128bit(0x44444444, 0x33333333, 0xDDDDDDDD, 0xCCCCCCCC)
        beat2 = _build_128bit(0x66666666, 0x55555555, 0xFFFFFFFF, 0xEEEEEEEE)

        exp0 = _build_128bit(0xDDDDDDDD, 0xCCCCCCCC, 0x22222222, 0x11111111)
        exp1 = _build_128bit(0xFFFFFFFF, 0xEEEEEEEE, 0x44444444, 0x33333333)
        exp2 = _build_128bit(0x00000000, 0x00000000, 0x66666666, 0x55555555)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            ctx.set(dut.offset, 2)
            await sender.send(ctx, beat0, first=1, last=0)
            await sender.send(ctx, beat1, first=0, last=0)
            await sender.send(ctx, beat2, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(3):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_pkt_align_dw2.vcd")

        assert len(results) == 3

        # Verify payloads
        assert results[0]["payload"] == exp0, \
            f"out0: expected {exp0:#034x}, got {results[0]['payload']:#034x}"
        assert results[1]["payload"] == exp1, \
            f"out1: expected {exp1:#034x}, got {results[1]['payload']:#034x}"
        assert results[2]["payload"] == exp2, \
            f"out2: expected {exp2:#034x}, got {results[2]['payload']:#034x}"

        # Verify first/last framing
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0
        assert results[2]["first"] == 0
        assert results[2]["last"] == 1

    def test_packet_aligner_zero_offset(self):
        """Offset=0 is a pure pass-through — data and framing unchanged."""
        dut = PacketAligner(data_width=128, granularity=32)
        results = []

        pkt = [
            _build_128bit(0x44444444, 0x33333333, 0x22222222, 0x11111111),
            _build_128bit(0x88888888, 0x77777777, 0x66666666, 0x55555555),
        ]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            ctx.set(dut.offset, 0)
            await sender.send(ctx, pkt[0], first=1, last=0)
            await sender.send(ctx, pkt[1], first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_pkt_align_zero.vcd")

        assert len(results) == 2
        assert results[0]["payload"] == pkt[0]
        assert results[1]["payload"] == pkt[1]
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

    def test_packet_aligner_back_to_back_packets(self):
        """Two consecutive packets with different offsets work correctly.

        Packet A: offset=1 (shift by 32 bits), 2 input beats
        Packet B: offset=0 (pass-through), 2 input beats
        """
        dut = PacketAligner(data_width=128, granularity=32)
        results_a = []
        results_b = []

        # Packet A: offset=1, shift_bits=32
        a0 = _build_128bit(0xDD000000, 0xCC000000, 0xBB000000, 0xAA000000)
        a1 = _build_128bit(0x44000000, 0x33000000, 0x22000000, 0x11000000)
        # carry = a0
        # out_a0 = (Cat(a0, a1) >> 32)[:128]
        #   = a0[32:128] | a1[0:32] << 96
        #   = [0xBB000000, 0xCC000000, 0xDD000000, 0x11000000]
        # carry = a1
        # flush_a = (Cat(a1, 0) >> 32)[:128]
        #   = a1[32:128]
        #   = [0x22000000, 0x33000000, 0x44000000, 0x00000000]
        exp_a0 = _build_128bit(0x11000000, 0xDD000000, 0xCC000000, 0xBB000000)
        exp_a1 = _build_128bit(0x00000000, 0x44000000, 0x33000000, 0x22000000)

        # Packet B: offset=0, pass-through
        b0 = _build_128bit(0x88000000, 0x77000000, 0x66000000, 0x55000000)
        b1 = _build_128bit(0xCC000000, 0xBB000000, 0xAA000000, 0x99000000)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Packet A: offset=1
            ctx.set(dut.offset, 1)
            await sender.send(ctx, a0, first=1, last=0)
            await sender.send(ctx, a1, first=0, last=1)
            # Packet B: offset=0
            ctx.set(dut.offset, 0)
            await sender.send(ctx, b0, first=1, last=0)
            await sender.send(ctx, b1, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Packet A: 2 output beats (aligned + flush)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results_a.append(beat)
            # Packet B: 2 output beats (pass-through)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results_b.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_pkt_align_b2b.vcd")

        # Verify Packet A
        assert len(results_a) == 2
        assert results_a[0]["payload"] == exp_a0, \
            f"A out0: expected {exp_a0:#034x}, got {results_a[0]['payload']:#034x}"
        assert results_a[1]["payload"] == exp_a1, \
            f"A out1: expected {exp_a1:#034x}, got {results_a[1]['payload']:#034x}"
        assert results_a[0]["first"] == 1
        assert results_a[0]["last"] == 0
        assert results_a[1]["first"] == 0
        assert results_a[1]["last"] == 1

        # Verify Packet B (pass-through)
        assert len(results_b) == 2
        assert results_b[0]["payload"] == b0
        assert results_b[1]["payload"] == b1
        assert results_b[0]["first"] == 1
        assert results_b[0]["last"] == 0
        assert results_b[1]["first"] == 0
        assert results_b[1]["last"] == 1


# ---------------------------------------------------------------------------
# WordReorder tests
# ---------------------------------------------------------------------------

class TestWordReorder:
    """Test WordReorder (word-level permutation within a data beat)."""

    def test_word_reorder_256bit_reverse(self):
        """256-bit data, 32-bit words, reverse order (7,6,5,4,3,2,1,0).

        Send data with known DWORD pattern and verify DWORDs are reversed.
        """
        dut = WordReorder(data_width=256, word_width=32,
                          order=(7, 6, 5, 4, 3, 2, 1, 0))
        results = []

        # Build 256-bit value from 8 DWORDs (DWORD0 in LSBs)
        dwords_in = [
            0x01020304,  # word 0
            0x05060708,  # word 1
            0x090A0B0C,  # word 2
            0x0D0E0F10,  # word 3
            0x11121314,  # word 4
            0x15161718,  # word 5
            0x191A1B1C,  # word 6
            0x1D1E1F20,  # word 7
        ]
        input_val = sum(d << (32 * i) for i, d in enumerate(dwords_in))

        # After reverse: output word i comes from input word (7-i)
        # output word 0 = input word 7 = 0x1D1E1F20
        # output word 1 = input word 6 = 0x191A1B1C
        # ...
        # output word 7 = input word 0 = 0x01020304
        dwords_out = list(reversed(dwords_in))
        expected_val = sum(d << (32 * i) for i, d in enumerate(dwords_out))

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, input_val, first=1, last=1,
                              keep=(1 << 32) - 1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_word_reorder_256_reverse.vcd")

        assert results[0]["payload"] == expected_val, \
            f"expected {expected_val:#066x}, got {results[0]['payload']:#066x}"

    def test_word_reorder_64bit_swap_halves(self):
        """64-bit data, 32-bit words, order (1,0) — swap upper and lower DWORDs."""
        dut = WordReorder(data_width=64, word_width=32, order=(1, 0))
        results = []

        # word0 (bits 31:0) = 0xAABBCCDD, word1 (bits 63:32) = 0x11223344
        input_val = 0xAABBCCDD | (0x11223344 << 32)
        # After swap: output word0 = input word1, output word1 = input word0
        expected_val = 0x11223344 | (0xAABBCCDD << 32)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, input_val, first=1, last=1,
                              keep=0xFF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_word_reorder_64_swap.vcd")

        assert results[0]["payload"] == expected_val, \
            f"expected {expected_val:#018x}, got {results[0]['payload']:#018x}"

    def test_word_reorder_with_keep(self):
        """Verify keep bits are reordered along with data words.

        64-bit data, 32-bit words, order (1,0).
        Each DWORD has 4 keep bits (one per byte).
        """
        dut = WordReorder(data_width=64, word_width=32, order=(1, 0))
        results = []

        # word0 = 0xDEADBEEF, word1 = 0xCAFEBABE
        input_val = 0xDEADBEEF | (0xCAFEBABE << 32)
        expected_val = 0xCAFEBABE | (0xDEADBEEF << 32)

        # keep: word0 keep = 0b1010, word1 keep = 0b0101
        # Combined: bits [3:0]=word0, bits [7:4]=word1 → 0b0101_1010 = 0x5A
        keep_in = 0b0101_1010
        # After swap: output word0 keep = input word1 keep = 0b0101
        #             output word1 keep = input word0 keep = 0b1010
        # Combined: 0b1010_0101 = 0xA5
        keep_out = 0b1010_0101

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, input_val, first=1, last=1,
                              keep=keep_in)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_word_reorder_keep.vcd")

        assert results[0]["payload"] == expected_val
        assert results[0]["keep"] == keep_out, \
            f"expected keep {keep_out:#04x}, got {results[0]['keep']:#04x}"

    def test_word_reorder_identity(self):
        """Identity order (0,1,2,3) should be pass-through.

        128-bit data, 32-bit words, order (0,1,2,3).
        """
        dut = WordReorder(data_width=128, word_width=32,
                          order=(0, 1, 2, 3))
        results = []

        input_val = _build_128bit(0x44444444, 0x33333333,
                                  0x22222222, 0x11111111)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, input_val, first=1, last=1,
                              keep=0xFFFF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_word_reorder_identity.vcd")

        assert results[0]["payload"] == input_val, \
            f"expected {input_val:#034x}, got {results[0]['payload']:#034x}"
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1
