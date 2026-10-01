"""Tests for amaranth_stream.converter (StreamConverter, StrideConverter,
Gearbox, StreamCast, Pack, Unpack)."""

import random
import pytest

from amaranth.hdl import unsigned, Shape
from amaranth.lib import data
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.converter import (
    StreamConverter,
    StrideConverter,
    Gearbox,
    StreamCast,
    Pack,
    Unpack,
    ByteEnableSerializer,
)
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


def _payload_int(val):
    """Convert a payload value to int, handling ArrayLayout views."""
    return int(val)


# ---------------------------------------------------------------------------
# StreamConverter tests
# ---------------------------------------------------------------------------

class TestStreamConverterUpsize:
    """Test StreamConverter upsizing (narrow → wide)."""

    def test_upsize_8_to_32(self):
        """4 narrow 8-bit beats → 1 wide 32-bit beat."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(32))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send 4 bytes: 0x11, 0x22, 0x33, 0x44
            for val in [0x11, 0x22, 0x33, 0x44]:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_upsize_8_to_32.vcd")

        # Bytes packed little-endian: 0x44332211
        assert results == [0x44332211]

    def test_upsize_8_to_16(self):
        """2 narrow 8-bit beats → 1 wide 16-bit beat."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(16))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in [0xAB, 0xCD]:
                await sender.send(ctx, val)
            for val in [0x12, 0x34]:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_upsize_8_to_16.vcd")

        assert results == [0xCDAB, 0x3412]

    def test_upsize_multiple_words(self):
        """Multiple upsized words in sequence."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(32))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Word 1: 0x04030201
            for val in [0x01, 0x02, 0x03, 0x04]:
                await sender.send(ctx, val)
            # Word 2: 0x08070605
            for val in [0x05, 0x06, 0x07, 0x08]:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_upsize_multi.vcd")

        assert results == [0x04030201, 0x08070605]


class TestStreamConverterDownsize:
    """Test StreamConverter downsizing (wide → narrow)."""

    def test_downsize_32_to_8(self):
        """1 wide 32-bit beat → 4 narrow 8-bit beats."""
        i_sig = Signature(unsigned(32))
        o_sig = Signature(unsigned(8))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x44332211)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_downsize_32_to_8.vcd")

        # Bytes extracted little-endian
        assert results == [0x11, 0x22, 0x33, 0x44]

    def test_downsize_16_to_8(self):
        """1 wide 16-bit beat → 2 narrow 8-bit beats."""
        i_sig = Signature(unsigned(16))
        o_sig = Signature(unsigned(8))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xCDAB)
            await sender.send(ctx, 0x3412)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_downsize_16_to_8.vcd")

        assert results == [0xAB, 0xCD, 0x12, 0x34]

    def test_downsize_multiple_words(self):
        """Multiple downsized words in sequence."""
        i_sig = Signature(unsigned(32))
        o_sig = Signature(unsigned(8))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x04030201)
            await sender.send(ctx, 0x08070605)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(8):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_downsize_multi.vcd")

        assert results == [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]


class TestStreamConverterIdentity:
    """Test StreamConverter identity (same width)."""

    def test_identity(self):
        """Same width: wire-through."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(8))
        dut = StreamConverter(i_sig, o_sig)
        results = []
        expected = [0x11, 0x22, 0x33, 0x44]

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
                 vcd_name="test_identity.vcd")

        assert results == expected


class TestStreamConverterFirstLast:
    """Test first/last propagation through StreamConverter."""

    def test_upsize_with_first_last(self):
        """first/last propagation during upsizing."""
        i_sig = Signature(unsigned(8), has_first_last=True)
        o_sig = Signature(unsigned(16), has_first_last=True)
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send 2 beats with first on beat 0, last on beat 1
            await sender.send(ctx, 0xAA, first=1, last=0)
            await sender.send(ctx, 0xBB, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_upsize_fl.vcd")

        assert len(results) == 1
        assert results[0]["payload"] == 0xBBAA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1

    def test_downsize_with_first_last(self):
        """first/last propagation during downsizing."""
        i_sig = Signature(unsigned(16), has_first_last=True)
        o_sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xBBAA, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_downsize_fl.vcd")

        assert len(results) == 2
        assert results[0]["payload"] == 0xAA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["payload"] == 0xBB
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1


class TestStreamConverterBackpressure:
    """Test backpressure handling in StreamConverter."""

    def test_upsize_backpressure(self):
        """Upsizing with random backpressure."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(32))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        # Send 3 words = 12 bytes
        input_data = list(range(1, 13))
        expected_words = [
            0x04030201,
            0x08070605,
            0x0C0B0A09,
        ]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            for val in input_data:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in range(3):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_upsize_bp.vcd")

        assert results == expected_words

    def test_downsize_backpressure(self):
        """Downsizing with random backpressure."""
        i_sig = Signature(unsigned(32))
        o_sig = Signature(unsigned(8))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            await sender.send(ctx, 0x04030201)
            await sender.send(ctx, 0x08070605)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in range(8):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_downsize_bp.vcd")

        assert results == [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]


class TestStreamConverterStress:
    """Stress tests for StreamConverter."""

    def test_stress_upsize(self):
        """Random valid/ready stress test for upsizing."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(32))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        rng = random.Random(123)
        # 10 words = 40 bytes
        input_bytes = [rng.randint(0, 255) for _ in range(40)]
        expected_words = []
        for i in range(0, 40, 4):
            word = (input_bytes[i] |
                    (input_bytes[i+1] << 8) |
                    (input_bytes[i+2] << 16) |
                    (input_bytes[i+3] << 24))
            expected_words.append(word)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=200)
            for val in input_bytes:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=300)
            for _ in range(10):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=500_000,
                 vcd_name="test_stress_upsize.vcd")

        assert results == expected_words

    def test_stress_downsize(self):
        """Random valid/ready stress test for downsizing."""
        i_sig = Signature(unsigned(32))
        o_sig = Signature(unsigned(8))
        dut = StreamConverter(i_sig, o_sig)
        results = []

        rng = random.Random(456)
        input_words = [rng.randint(0, 0xFFFFFFFF) for _ in range(10)]
        expected_bytes = []
        for word in input_words:
            expected_bytes.append(word & 0xFF)
            expected_bytes.append((word >> 8) & 0xFF)
            expected_bytes.append((word >> 16) & 0xFF)
            expected_bytes.append((word >> 24) & 0xFF)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=201)
            for val in input_words:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=301)
            for _ in range(40):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=500_000,
                 vcd_name="test_stress_downsize.vcd")

        assert results == expected_bytes


class TestStreamConverterValidation:
    """Test constructor validation for StreamConverter."""

    def test_rejects_non_integer_ratio(self):
        """Non-integer ratio raises ValueError."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(12))
        with pytest.raises(ValueError, match="integer multiple"):
            StreamConverter(i_sig, o_sig)

    def test_rejects_non_signature_input(self):
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamConverter("not a sig", Signature(unsigned(8)))

    def test_rejects_non_signature_output(self):
        with pytest.raises(TypeError, match="amaranth_stream.Signature"):
            StreamConverter(Signature(unsigned(8)), "not a sig")


# ---------------------------------------------------------------------------
# Gearbox tests
# ---------------------------------------------------------------------------

class TestGearbox:
    """Test Gearbox (non-integer ratio width converter)."""

    def test_gearbox_10_to_8(self):
        """10-bit input → 8-bit output (LCM=40, 4 in → 5 out)."""
        dut = Gearbox(10, 8)
        results = []

        # 4 x 10-bit values = 40 bits → 5 x 8-bit values
        # Input: 0x3FF, 0x000, 0x155, 0x2AA (10-bit values)
        # Pack as 40 bits: 0x3FF | (0x000 << 10) | (0x155 << 20) | (0x2AA << 30)
        # = 0x3FF + 0 + 0x15500000 + 0xAA800000000 (but only 40 bits)
        # Let's use simpler values
        input_vals = [0b0000000001, 0b0000000010, 0b0000000100, 0b0000001000]
        # 40 bits: 01_00000010_00000001_00000100_00001000 (reversed bit order)
        # Actually: val0[9:0] | val1[9:0] | val2[9:0] | val3[9:0]
        # = 0000000001 | 0000000010 | 0000000100 | 0000001000
        # As 40-bit number: 0000001000_0000000100_0000000010_0000000001
        # Split into 8-bit: 00000001, 00001000, 00000001, 00010000, 00000010
        # Let me compute properly:
        combined = (input_vals[0] |
                    (input_vals[1] << 10) |
                    (input_vals[2] << 20) |
                    (input_vals[3] << 30))
        expected = []
        for i in range(5):
            expected.append((combined >> (i * 8)) & 0xFF)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(5):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_gearbox_10_to_8.vcd")

        assert results == expected

    def test_gearbox_8_to_10(self):
        """8-bit input → 10-bit output (LCM=40, 5 in → 4 out)."""
        dut = Gearbox(8, 10)
        results = []

        # 5 x 8-bit values = 40 bits → 4 x 10-bit values
        input_vals = [0x01, 0x02, 0x03, 0x04, 0x05]
        combined = 0
        for i, v in enumerate(input_vals):
            combined |= v << (i * 8)
        expected = []
        for i in range(4):
            expected.append((combined >> (i * 10)) & 0x3FF)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_gearbox_8_to_10.vcd")

        assert results == expected

    def test_gearbox_same_width(self):
        """Same width gearbox acts as pass-through."""
        dut = Gearbox(8, 8)
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

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_gearbox_same.vcd")

        assert results == expected


class TestGearboxPacket:
    """Test Gearbox with has_first_last=True (packet-aware mode)."""

    def test_gearbox_packet_aware_10to8(self):
        """10-bit to 8-bit gearbox with first/last framing.

        Send 3 beats of 10-bit data (30 bits total) with first on beat 0
        and last on beat 2. Expect ceil(30/8) = 4 output beats (last one
        has only 6 valid bits, zero-padded to 8).
        """
        dut = Gearbox(10, 8, has_first_last=True)
        results = []

        # 3 x 10-bit values
        input_vals = [0x1AB, 0x2CD, 0x3EF]  # 10-bit values
        combined = 0
        for i, v in enumerate(input_vals):
            combined |= v << (i * 10)
        # 30 bits total → 4 x 8-bit outputs (last one is 6 bits, zero-padded)
        n_out = (30 + 7) // 8  # = 4
        expected = []
        for i in range(n_out):
            expected.append((combined >> (i * 8)) & 0xFF)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for i, val in enumerate(input_vals):
                first = 1 if i == 0 else 0
                last = 1 if i == len(input_vals) - 1 else 0
                await sender.send(ctx, val, first=first, last=last)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(n_out):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_gearbox_packet_10to8.vcd")

        # Verify data
        for i, (beat, exp) in enumerate(zip(results, expected)):
            assert beat["payload"] == exp, \
                f"Beat {i}: expected {exp:#04x}, got {beat['payload']:#04x}"

        # Verify first/last framing
        assert results[0]["first"] == 1, "First beat should have first=1"
        for i in range(1, n_out):
            assert results[i]["first"] == 0, f"Beat {i} should have first=0"
        for i in range(n_out - 1):
            assert results[i]["last"] == 0, f"Beat {i} should have last=0"
        assert results[-1]["last"] == 1, "Last beat should have last=1"

    def test_gearbox_packet_aware_flush(self):
        """Verify that on last, remaining bits are flushed as a partial beat.

        Send 1 beat of 10-bit data with first=1, last=1. This gives 10 bits,
        which produces 1 full 8-bit beat + 1 partial 2-bit beat (zero-padded).
        """
        dut = Gearbox(10, 8, has_first_last=True)
        results = []

        input_val = 0x2AB  # 10-bit: 0b10_1010_1011

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, input_val, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # 10 bits → 1 full byte + 1 partial byte (2 bits)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_gearbox_packet_flush.vcd")

        assert len(results) == 2
        # First byte: lower 8 bits of 0x2AB = 0xAB
        assert results[0]["payload"] == (input_val & 0xFF)
        # Second byte: upper 2 bits of 0x2AB = 0b10 = 0x02 (zero-padded)
        assert results[1]["payload"] == ((input_val >> 8) & 0xFF)

        # Framing
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

    def test_gearbox_packet_aware_back_to_back(self):
        """Two consecutive packets, verify they're correctly separated.

        Packet 1: 2 beats of 10-bit data (20 bits → 3 output bytes, last partial)
        Packet 2: 1 beat of 10-bit data (10 bits → 2 output bytes, last partial)
        """
        dut = Gearbox(10, 8, has_first_last=True)
        results_pkt1 = []
        results_pkt2 = []

        # Packet 1: 2 x 10-bit
        pkt1_vals = [0x0FF, 0x100]
        pkt1_combined = pkt1_vals[0] | (pkt1_vals[1] << 10)
        # 20 bits → 3 bytes (last byte has 4 valid bits)
        pkt1_expected = []
        for i in range(3):
            pkt1_expected.append((pkt1_combined >> (i * 8)) & 0xFF)

        # Packet 2: 1 x 10-bit
        pkt2_val = 0x3AB
        pkt2_expected = [pkt2_val & 0xFF, (pkt2_val >> 8) & 0xFF]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Packet 1
            await sender.send(ctx, pkt1_vals[0], first=1, last=0)
            await sender.send(ctx, pkt1_vals[1], first=0, last=1)
            # Packet 2
            await sender.send(ctx, pkt2_val, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Receive packet 1 (3 beats)
            for _ in range(3):
                beat = await receiver.recv(ctx)
                results_pkt1.append(beat)
            # Receive packet 2 (2 beats)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results_pkt2.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=300_000,
                 vcd_name="test_gearbox_packet_b2b.vcd")

        # Verify packet 1 data
        for i, (beat, exp) in enumerate(zip(results_pkt1, pkt1_expected)):
            assert beat["payload"] == exp, \
                f"Pkt1 beat {i}: expected {exp:#04x}, got {beat['payload']:#04x}"

        # Verify packet 1 framing
        assert results_pkt1[0]["first"] == 1
        assert results_pkt1[0]["last"] == 0
        assert results_pkt1[1]["first"] == 0
        assert results_pkt1[1]["last"] == 0
        assert results_pkt1[2]["first"] == 0
        assert results_pkt1[2]["last"] == 1

        # Verify packet 2 data
        for i, (beat, exp) in enumerate(zip(results_pkt2, pkt2_expected)):
            assert beat["payload"] == exp, \
                f"Pkt2 beat {i}: expected {exp:#04x}, got {beat['payload']:#04x}"

        # Verify packet 2 framing
        assert results_pkt2[0]["first"] == 1
        assert results_pkt2[0]["last"] == 0
        assert results_pkt2[1]["first"] == 0
        assert results_pkt2[1]["last"] == 1

    def test_gearbox_no_first_last_backward_compat(self):
        """Verify has_first_last=False (default) works exactly as before.

        Same test as test_gearbox_10_to_8 but explicitly checking backward
        compatibility.
        """
        dut = Gearbox(10, 8)  # default has_first_last=False
        results = []

        input_vals = [0b0000000001, 0b0000000010, 0b0000000100, 0b0000001000]
        combined = (input_vals[0] |
                    (input_vals[1] << 10) |
                    (input_vals[2] << 20) |
                    (input_vals[3] << 30))
        expected = []
        for i in range(5):
            expected.append((combined >> (i * 8)) & 0xFF)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(5):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_gearbox_no_fl_compat.vcd")

        assert results == expected

        # Verify no first/last fields in the output
        assert "first" not in results[0] if isinstance(results[0], dict) else True


# ---------------------------------------------------------------------------
# StreamCast tests
# ---------------------------------------------------------------------------

class TestStreamCast:
    """Test StreamCast (zero-cost bit reinterpretation)."""

    def test_cast_basic(self):
        """Reinterpret 16-bit unsigned as 16-bit unsigned (different shapes)."""
        i_sig = Signature(unsigned(16))
        o_sig = Signature(unsigned(16))
        dut = StreamCast(i_sig, o_sig)
        results = []
        expected = [0x1234, 0xABCD, 0x0000, 0xFFFF]

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
                 vcd_name="test_cast_basic.vcd")

        assert results == expected

    def test_cast_with_first_last(self):
        """first/last pass through cast."""
        i_sig = Signature(unsigned(8), has_first_last=True)
        o_sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamCast(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, first=1, last=0)
            await sender.send(ctx, 0xBB, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_cast_fl.vcd")

        assert results[0]["payload"] == 0xAA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["payload"] == 0xBB
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

    def test_cast_rejects_different_width(self):
        """StreamCast rejects different payload widths."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(16))
        with pytest.raises(ValueError, match="same payload bit width"):
            StreamCast(i_sig, o_sig)


# ---------------------------------------------------------------------------
# Pack tests
# ---------------------------------------------------------------------------

class TestPack:
    """Test Pack (collect N narrow beats into 1 wide beat)."""

    def test_pack_4(self):
        """Pack 4 x 8-bit beats into 1 x 32-bit beat."""
        i_sig = Signature(unsigned(8))
        dut = Pack(i_sig, 4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in [0x11, 0x22, 0x33, 0x44]:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_pack_4.vcd")

        # Packed as: elem[0] in LSBs
        assert results == [0x44332211]

    def test_pack_2(self):
        """Pack 2 x 8-bit beats into 1 x 16-bit beat."""
        i_sig = Signature(unsigned(8))
        dut = Pack(i_sig, 2)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in [0xAB, 0xCD, 0x12, 0x34]:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_pack_2.vcd")

        assert results == [0xCDAB, 0x3412]

    def test_pack_with_first_last(self):
        """Pack with first/last propagation."""
        i_sig = Signature(unsigned(8), has_first_last=True)
        dut = Pack(i_sig, 2)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, first=1, last=0)
            await sender.send(ctx, 0xBB, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_pack_fl.vcd")

        assert results[0]["payload"] == 0xBBAA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1

    def test_pack_backpressure(self):
        """Pack with random backpressure."""
        i_sig = Signature(unsigned(8))
        dut = Pack(i_sig, 4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            for val in [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_pack_bp.vcd")

        assert results == [0x04030201, 0x08070605]


# ---------------------------------------------------------------------------
# Unpack tests
# ---------------------------------------------------------------------------

class TestUnpack:
    """Test Unpack (split 1 wide beat into N narrow beats)."""

    def test_unpack_4(self):
        """Unpack 1 x 32-bit beat into 4 x 8-bit beats."""
        o_sig = Signature(unsigned(8))
        dut = Unpack(o_sig, 4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x44332211)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_unpack_4.vcd")

        assert results == [0x11, 0x22, 0x33, 0x44]

    def test_unpack_2(self):
        """Unpack 1 x 16-bit beat into 2 x 8-bit beats."""
        o_sig = Signature(unsigned(8))
        dut = Unpack(o_sig, 2)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xCDAB)
            await sender.send(ctx, 0x3412)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_unpack_2.vcd")

        assert results == [0xAB, 0xCD, 0x12, 0x34]

    def test_unpack_with_first_last(self):
        """Unpack with first/last propagation."""
        o_sig = Signature(unsigned(8), has_first_last=True)
        dut = Unpack(o_sig, 2)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xBBAA, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_unpack_fl.vcd")

        assert results[0]["payload"] == 0xAA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["payload"] == 0xBB
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

    def test_unpack_backpressure(self):
        """Unpack with random backpressure."""
        o_sig = Signature(unsigned(8))
        dut = Unpack(o_sig, 4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            await sender.send(ctx, 0x04030201)
            await sender.send(ctx, 0x08070605)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in range(8):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_unpack_bp.vcd")

        assert results == [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]


# ---------------------------------------------------------------------------
# Pack/Unpack roundtrip tests
# ---------------------------------------------------------------------------

class TestPackUnpackRoundtrip:
    """Test Pack → Unpack roundtrip."""

    def test_pack_unpack_roundtrip(self):
        """Pack 4 beats, then unpack back to 4 beats."""
        from amaranth import Module
        from amaranth.lib import wiring as wiring_lib
        from amaranth.lib.wiring import In, Out, connect

        class PackUnpack(wiring_lib.Component):
            def __init__(self):
                i_sig = Signature(unsigned(8))
                o_sig = Signature(unsigned(8))
                super().__init__({
                    "i_stream": In(i_sig),
                    "o_stream": Out(o_sig),
                })

            def elaborate(self, platform):
                m = Module()
                m.submodules.pack = pack_m = Pack(Signature(unsigned(8)), 4)
                m.submodules.unpack = unpack_m = Unpack(Signature(unsigned(8)), 4)

                connect(m, wiring_lib.flipped(self.i_stream), pack_m.i_stream)
                connect(m, pack_m.o_stream, unpack_m.i_stream)
                connect(m, unpack_m.o_stream, wiring_lib.flipped(self.o_stream))
                return m

        dut = PackUnpack()
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

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_pack_unpack_roundtrip.vcd")

        assert results == expected


# ---------------------------------------------------------------------------
# StrideConverter tests
# ---------------------------------------------------------------------------

class TestStrideConverter:
    """Test StrideConverter."""

    def test_stride_integer_ratio(self):
        """Integer ratio: 8→16 (same as StreamConverter)."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(16))
        dut = StrideConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in [0xAB, 0xCD]:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_stride_int.vcd")

        assert results == [0xCDAB]

    def test_stride_identity(self):
        """Same width: wire-through."""
        i_sig = Signature(unsigned(8))
        o_sig = Signature(unsigned(8))
        dut = StrideConverter(i_sig, o_sig)
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
                 vcd_name="test_stride_identity.vcd")

        assert results == expected

    def test_stride_non_integer_ratio(self):
        """Non-integer ratio: 6→4 (LCM=12, 2 in → 3 out)."""
        i_sig = Signature(unsigned(6))
        o_sig = Signature(unsigned(4))
        dut = StrideConverter(i_sig, o_sig)
        results = []

        # 2 x 6-bit inputs = 12 bits → 3 x 4-bit outputs
        input_vals = [0b110011, 0b010101]
        combined = input_vals[0] | (input_vals[1] << 6)
        expected = []
        for i in range(3):
            expected.append((combined >> (i * 4)) & 0xF)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in input_vals:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(3):
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_stride_non_int.vcd")

        assert results == expected

    def test_stride_converter_short_packet_upsize(self):
        """Short packet upsize: 8→32 with only 2 input beats (ratio=4).

        Sends a 2-beat packet (with last on beat 2) into a 4x upsize
        StrideConverter. The converter must not deadlock waiting for the
        remaining 2 beats. Instead it should zero-pad the unfilled slots
        and produce one output beat with correct data, zero-padding, and
        last asserted.
        """
        i_sig = Signature(unsigned(8), has_first_last=True)
        o_sig = Signature(unsigned(32), has_first_last=True)
        dut = StrideConverter(i_sig, o_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send only 2 beats for a 4x upsize ratio — short packet
            await sender.send(ctx, 0xAA, first=1, last=0)
            await sender.send(ctx, 0xBB, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=50_000,
                 vcd_name="test_stride_short_packet.vcd")

        # Should get 1 output beat
        assert len(results) == 1
        beat = results[0]
        # Data: byte 0 = 0xAA, byte 1 = 0xBB, bytes 2-3 = 0x00 (zero-padded)
        assert beat["payload"] == 0x0000BBAA, \
            f"Expected 0x0000BBAA, got {beat['payload']:#010x}"
        # first should be set (captured from first input beat)
        assert beat["first"] == 1
        # last should be set (propagated from the short packet's last)
        assert beat["last"] == 1


# ---------------------------------------------------------------------------
# StrideConverter param_shape tests
# ---------------------------------------------------------------------------

class TestStrideConverterParam:
    """Test StrideConverter param_shape propagation."""

    def test_stride_converter_param_propagation(self):
        """Upconvert 8→32 bit with param_shape=4.

        Send a packet where the first beat has param=0xA and subsequent
        beats have param=0x0.  Verify the output beat has param=0xA
        (captured from the first beat).
        """
        i_sig = Signature(unsigned(8), has_first_last=True)
        o_sig = Signature(unsigned(32), has_first_last=True)
        dut = StrideConverter(i_sig, o_sig, param_shape=4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # First beat: param=0xA, subsequent beats: param=0x0
            await sender.send(ctx, 0x11, first=1, last=0, param=0xA)
            await sender.send(ctx, 0x22, first=0, last=0, param=0x0)
            await sender.send(ctx, 0x33, first=0, last=0, param=0x0)
            await sender.send(ctx, 0x44, first=0, last=1, param=0x0)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_stride_param_up.vcd")

        assert len(results) == 1
        assert results[0]["payload"] == 0x44332211
        assert results[0]["param"] == 0xA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1

    def test_stride_converter_param_downconvert(self):
        """Downconvert 32→8 bit with param_shape=4.

        Send a beat with param=0xB.  Verify all 4 output beats have
        param=0xB.
        """
        i_sig = Signature(unsigned(32), has_first_last=True)
        o_sig = Signature(unsigned(8), has_first_last=True)
        dut = StrideConverter(i_sig, o_sig, param_shape=4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x44332211, first=1, last=1, param=0xB)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_stride_param_down.vcd")

        assert len(results) == 4
        assert results[0]["payload"] == 0x11
        assert results[1]["payload"] == 0x22
        assert results[2]["payload"] == 0x33
        assert results[3]["payload"] == 0x44
        # All output beats should carry param=0xB
        for i, beat in enumerate(results):
            assert beat["param"] == 0xB, \
                f"Beat {i}: expected param=0xB, got {beat['param']:#x}"

    def test_stride_converter_param_multi_packet(self):
        """Verify param changes correctly between packets.

        Send two packets with different param values during upconvert
        8→16 and verify each output beat carries the correct param.
        """
        i_sig = Signature(unsigned(8), has_first_last=True)
        o_sig = Signature(unsigned(16), has_first_last=True)
        dut = StrideConverter(i_sig, o_sig, param_shape=4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Packet 1: param=0xA
            await sender.send(ctx, 0xAA, first=1, last=0, param=0xA)
            await sender.send(ctx, 0xBB, first=0, last=1, param=0x0)
            # Packet 2: param=0xC
            await sender.send(ctx, 0xCC, first=1, last=0, param=0xC)
            await sender.send(ctx, 0xDD, first=0, last=1, param=0x0)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_stride_param_multi.vcd")

        assert len(results) == 2
        # Packet 1
        assert results[0]["payload"] == 0xBBAA
        assert results[0]["param"] == 0xA
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1
        # Packet 2
        assert results[1]["payload"] == 0xDDCC
        assert results[1]["param"] == 0xC
        assert results[1]["first"] == 1
        assert results[1]["last"] == 1


# ---------------------------------------------------------------------------
# ByteEnableSerializer tests
# ---------------------------------------------------------------------------

class TestByteEnableSerializer:
    """Test ByteEnableSerializer (wide stream with byte enables → narrow valid-only stream)."""

    def test_be_serializer_256to8_sparse(self):
        """256-bit input, 8-bit output with sparse byte enables.

        Only bytes 0, 3, 7, 31 are enabled — verify only those 4 bytes
        appear in the output stream with correct first/last framing.
        """
        dut = ByteEnableSerializer(256, 8)
        results = []

        # Build a 256-bit (32-byte) payload where each byte = its index
        payload = 0
        for i in range(32):
            payload |= i << (i * 8)

        # Enable only bytes 0, 3, 7, 31
        keep = 0
        for bit in [0, 3, 7, 31]:
            keep |= 1 << bit

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, payload, first=1, last=1, keep=keep)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_be_ser_256to8_sparse.vcd")

        assert len(results) == 4
        # Byte 0 = 0x00, byte 3 = 0x03, byte 7 = 0x07, byte 31 = 0x1F
        assert results[0]["payload"] == 0x00
        assert results[1]["payload"] == 0x03
        assert results[2]["payload"] == 0x07
        assert results[3]["payload"] == 0x1F

        # first on first valid byte, last on last valid byte
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0
        assert results[2]["first"] == 0
        assert results[2]["last"] == 0
        assert results[3]["first"] == 0
        assert results[3]["last"] == 1

    def test_be_serializer_all_valid(self):
        """All keep bits set — verify all bytes emitted in order."""
        dut = ByteEnableSerializer(32, 8)
        results = []

        payload = 0x44332211
        keep = 0xF  # all 4 bytes valid

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, payload, first=1, last=1, keep=keep)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_be_ser_all_valid.vcd")

        assert len(results) == 4
        assert results[0]["payload"] == 0x11
        assert results[1]["payload"] == 0x22
        assert results[2]["payload"] == 0x33
        assert results[3]["payload"] == 0x44

        # first/last framing
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[3]["first"] == 0
        assert results[3]["last"] == 1

    def test_be_serializer_multi_beat_packet(self):
        """2-beat packet with different keep patterns.

        Beat 1: 32-bit, keep=0b0101 (bytes 0 and 2 valid)
        Beat 2: 32-bit, keep=0b1010 (bytes 1 and 3 valid)
        Total: 4 output bytes across 2 input beats.
        """
        dut = ByteEnableSerializer(32, 8)
        results = []

        # Beat 1: payload = 0x44332211, keep = 0b0101 → bytes 0 (0x11) and 2 (0x33)
        # Beat 2: payload = 0x88776655, keep = 0b1010 → bytes 1 (0x66) and 3 (0x88)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x44332211, first=1, last=0, keep=0b0101)
            await sender.send(ctx, 0x88776655, first=0, last=1, keep=0b1010)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_be_ser_multi_beat.vcd")

        assert len(results) == 4
        # Beat 1 outputs: byte 0 = 0x11, byte 2 = 0x33
        assert results[0]["payload"] == 0x11
        assert results[1]["payload"] == 0x33
        # Beat 2 outputs: byte 1 = 0x66, byte 3 = 0x88
        assert results[2]["payload"] == 0x66
        assert results[3]["payload"] == 0x88

        # first on first valid byte of first beat, last on last valid byte of last beat
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0  # not last beat
        assert results[2]["first"] == 0  # not first beat
        assert results[2]["last"] == 0
        assert results[3]["first"] == 0
        assert results[3]["last"] == 1

    def test_be_serializer_backpressure(self):
        """Verify backpressure works correctly with random ready."""
        dut = ByteEnableSerializer(32, 8)
        results = []

        # Send 2 beats with all bytes valid
        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=42)
            await sender.send(ctx, 0x04030201, first=1, last=0, keep=0xF)
            await sender.send(ctx, 0x08070605, first=0, last=1, keep=0xF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=99)
            for _ in range(8):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=500_000,
                 vcd_name="test_be_ser_backpressure.vcd")

        assert len(results) == 8
        expected_payloads = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        for i, (beat, expected) in enumerate(zip(results, expected_payloads)):
            assert beat["payload"] == expected, \
                f"Beat {i}: expected {expected:#x}, got {beat['payload']:#x}"

        # first on first byte, last on last byte
        assert results[0]["first"] == 1
        assert results[7]["last"] == 1

    def test_be_serializer_all_keep_zero(self):
        """All keep bits zero — entire beat should be skipped.

        Send beat 1 with keep=0 (skip), then beat 2 with keep=0xF (all valid).
        Only beat 2's bytes should appear in output.
        """
        dut = ByteEnableSerializer(32, 8)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Beat with all keep=0 (should be skipped entirely)
            await sender.send(ctx, 0xDEADBEEF, first=1, last=0, keep=0x0)
            # Beat with all keep=1
            await sender.send(ctx, 0x44332211, first=0, last=1, keep=0xF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, deadline_ns=200_000,
                 vcd_name="test_be_ser_all_zero.vcd")

        assert len(results) == 4
        # Only beat 2's bytes should appear
        assert results[0]["payload"] == 0x11
        assert results[1]["payload"] == 0x22
        assert results[2]["payload"] == 0x33
        assert results[3]["payload"] == 0x44

    def test_be_serializer_validation(self):
        """Constructor validation."""
        # i_width not multiple of 8
        with pytest.raises(ValueError, match="multiple of 8"):
            ByteEnableSerializer(10, 8)
        # o_width not multiple of 8
        with pytest.raises(ValueError, match="multiple of 8"):
            ByteEnableSerializer(32, 10)
        # i_width < o_width
        with pytest.raises(ValueError, match="must be >="):
            ByteEnableSerializer(8, 16)
