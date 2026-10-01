"""Tests for amaranth_stream.packet."""

import pytest

from amaranth import *
from amaranth.hdl import unsigned
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.packet import (
    HeaderLayout,
    Packetizer,
    Depacketizer,
    PacketFIFO,
    PacketStatus,
    Stitcher,
    LastInserter,
    LastOnTimeout,
)
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_packet.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


# ===========================================================================
# HeaderLayout tests
# ===========================================================================

class TestHeaderLayout:
    """Test HeaderLayout declarative header definition."""

    def test_header_layout_basic(self):
        """Create a simple header layout and check properties."""
        hl = HeaderLayout({"type": (8, 0)})
        assert hl.byte_length == 1
        assert "type" in hl.fields
        sl = hl.struct_layout
        # StructLayout should have the field
        assert "type" in dict(sl)

    def test_header_layout_multi_field(self):
        """Multiple fields with different offsets."""
        hl = HeaderLayout({
            "src": (8, 0),
            "dst": (8, 1),
            "length": (16, 2),
        })
        assert hl.byte_length == 4  # offset 2 + ceil(16/8) = 4
        assert hl.fields["src"] == (8, 0, 0)
        assert hl.fields["dst"] == (8, 1, 0)
        assert hl.fields["length"] == (16, 2, 0)

    def test_header_layout_non_byte_aligned(self):
        """Field width not a multiple of 8."""
        hl = HeaderLayout({"flags": (3, 0)})
        assert hl.byte_length == 1  # ceil(3/8) = 1

    def test_header_layout_bit_offset(self):
        """PCIe-style fields with bit-level placement within byte regions."""
        hl = HeaderLayout({
            "fmt":  (2, 0, 5),   # 2 bits at byte_offset=0, bit_offset=5
            "type": (5, 0, 0),   # 5 bits at byte_offset=0, bit_offset=0
            "tc":   (3, 1, 1),   # 3 bits at byte_offset=1, bit_offset=1
        })
        # fmt: abs_bit = 0*8+5 = 5, occupies bits 5..6
        # type: abs_bit = 0*8+0 = 0, occupies bits 0..4
        # tc: abs_bit = 1*8+1 = 9, occupies bits 9..11
        assert hl.abs_bit_offset("fmt") == 5
        assert hl.abs_bit_offset("type") == 0
        assert hl.abs_bit_offset("tc") == 9
        # Fields stored as 3-tuples
        assert hl.fields["fmt"] == (2, 0, 5)
        assert hl.fields["type"] == (5, 0, 0)
        assert hl.fields["tc"] == (3, 1, 1)
        # byte_length: max(ceil((5+2)/8), ceil((0+5)/8), ceil((1+3)/8)+1)
        # fmt: 0 + ceil(7/8) = 1
        # type: 0 + ceil(5/8) = 1
        # tc: 1 + ceil(4/8) = 2
        assert hl.byte_length == 2

    def test_header_layout_bit_offset_default(self):
        """Omitting bit_offset defaults to 0 (backward compatible)."""
        hl = HeaderLayout({
            "src": (8, 0),
            "dst": (8, 1),
        })
        # Should be normalised to 3-tuples with bit_offset=0
        assert hl.fields["src"] == (8, 0, 0)
        assert hl.fields["dst"] == (8, 1, 0)
        assert hl.abs_bit_offset("src") == 0
        assert hl.abs_bit_offset("dst") == 8

    def test_header_layout_bit_offset_overlap(self):
        """Overlapping fields should raise ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            HeaderLayout({
                "a": (5, 0, 0),   # bits 0..4
                "b": (5, 0, 3),   # bits 3..7 — overlaps with 'a'
            })


# ===========================================================================
# Packetizer tests
# ===========================================================================

class TestPacketizer:
    """Test Packetizer — insert header beats before payload."""

    def test_packetizer_basic(self):
        """Insert a 1-byte header before a 3-beat payload (8-bit stream)."""
        hl = HeaderLayout({"type": (8, 0)})
        sig = Signature(unsigned(8), has_first_last=True)
        dut = Packetizer(hl, sig)
        results = []

        async def testbench(ctx):
            sender = StreamSimSender(dut.i_stream)
            receiver = StreamSimReceiver(dut.o_stream)

            # Set header
            ctx.set(dut.header.type, 0x42)

            # Send payload (3 beats)
            async def send_payload():
                await sender.send(ctx, 0xAA, first=1, last=0)
                await sender.send(ctx, 0xBB, first=0, last=0)
                await sender.send(ctx, 0xCC, first=0, last=1)

            async def recv_all():
                for _ in range(4):  # 1 header + 3 payload
                    beat = await receiver.recv(ctx)
                    results.append(beat)

            # Run sender and receiver concurrently by interleaving
            # We need to use separate testbenches for this
            pass

        # Use separate testbenches for sender and receiver
        results_list = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Set header
            ctx.set(dut.header.type, 0x42)
            # Wait a tick for header to be set
            await ctx.tick()
            # Send payload
            await sender.send(ctx, 0xAA, first=1, last=0)
            await sender.send(ctx, 0xBB, first=0, last=0)
            await sender.send(ctx, 0xCC, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):  # 1 header + 3 payload
                beat = await receiver.recv(ctx)
                results_list.append(beat)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_packetizer_basic.vcd")

        assert len(results_list) == 4
        # First beat is header
        assert results_list[0]["payload"] == 0x42
        assert results_list[0]["first"] == 1
        assert results_list[0]["last"] == 0
        # Payload beats
        assert results_list[1]["payload"] == 0xAA
        assert results_list[1]["first"] == 0
        assert results_list[2]["payload"] == 0xBB
        assert results_list[2]["first"] == 0
        assert results_list[3]["payload"] == 0xCC
        assert results_list[3]["last"] == 1

    def test_packetizer_multi_beat_header(self):
        """Header larger than payload width requires multiple beats."""
        # 2-byte header on an 8-bit stream = 2 header beats
        hl = HeaderLayout({
            "src": (8, 0),
            "dst": (8, 1),
        })
        sig = Signature(unsigned(8), has_first_last=True)
        dut = Packetizer(hl, sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            ctx.set(dut.header.src, 0x11)
            ctx.set(dut.header.dst, 0x22)
            await ctx.tick()
            await sender.send(ctx, 0xDD, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(3):  # 2 header + 1 payload
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_packetizer_multi_beat.vcd")

        assert len(results) == 3
        # First header beat (byte 0 = src)
        assert results[0]["payload"] == 0x11
        assert results[0]["first"] == 1
        # Second header beat (byte 1 = dst)
        assert results[1]["payload"] == 0x22
        assert results[1]["first"] == 0
        # Payload
        assert results[2]["payload"] == 0xDD
        assert results[2]["last"] == 1

    def test_packetizer_packed_96bit_header_64bit_bus(self):
        """Packed mode: 96-bit header on 64-bit bus merges header tail with payload."""
        # 12-byte (96-bit) header on a 64-bit stream
        # remainder = 96 % 64 = 32 bits of header in last beat
        # pack_bits = 64 - 32 = 32 bits of payload packed into last header beat
        hl = HeaderLayout({
            "dw0": (32, 0),
            "dw1": (32, 4),
            "dw2": (32, 8),
        })
        sig = Signature(unsigned(64), has_first_last=True)
        dut = Packetizer(hl, sig, packed=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Set header fields
            ctx.set(dut.header.dw0, 0x11111111)
            ctx.set(dut.header.dw1, 0x22222222)
            ctx.set(dut.header.dw2, 0x33333333)
            await ctx.tick()
            # Send 2 payload beats (64-bit each)
            await sender.send(ctx, 0xBBBBBBBBAAAAAAAA, first=1, last=0)
            await sender.send(ctx, 0xDDDDDDDDCCCCCCCC, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Expect 4 output beats:
            # Beat 0: full header [dw0|dw1]
            # Beat 1: packed [dw2|payload_lower32]
            # Beat 2: shifted payload
            # Beat 3: drain (last=1)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000,
                 vcd_name="test_packetizer_packed_96.vcd")

        assert len(results) == 4

        # Beat 0: first full header beat [dw0 | dw1] = 0x2222222211111111
        assert results[0]["payload"] == 0x2222222211111111, \
            f"Beat 0: expected 0x2222222211111111, got {results[0]['payload']:#018x}"
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0

        # Beat 1: packed beat [dw2(lower 32) | payload_beat0[0:32](upper 32)]
        # = 0xAAAAAAAA33333333
        assert results[1]["payload"] == 0xAAAAAAAA33333333, \
            f"Beat 1: expected 0xAAAAAAAA33333333, got {results[1]['payload']:#018x}"
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0

        # Beat 2: shifted payload [payload_beat0[32:64] | payload_beat1[0:32]]
        # = 0xCCCCCCCCBBBBBBBB
        assert results[2]["payload"] == 0xCCCCCCCCBBBBBBBB, \
            f"Beat 2: expected 0xCCCCCCCCBBBBBBBB, got {results[2]['payload']:#018x}"
        assert results[2]["first"] == 0
        assert results[2]["last"] == 0

        # Beat 3: drain [payload_beat1[32:64] | 0] = 0x00000000DDDDDDDD
        assert results[3]["payload"] == 0x00000000DDDDDDDD, \
            f"Beat 3: expected 0x00000000DDDDDDDD, got {results[3]['payload']:#018x}"
        assert results[3]["first"] == 0
        assert results[3]["last"] == 1

    def test_packetizer_bit_level_header(self):
        """Packetizer with bit-level header fields (PCIe-style sub-byte packing).

        Header layout (2 bytes = 16 bits on a 16-bit stream = 1 header beat):
          type: 5 bits at byte_offset=0, bit_offset=0  → bits [0:5]
          fmt:  2 bits at byte_offset=0, bit_offset=5  → bits [5:7]
          tc:   3 bits at byte_offset=1, bit_offset=0  → bits [8:11]
        """
        hl = HeaderLayout({
            "type": (5, 0, 0),   # bits 0..4
            "fmt":  (2, 0, 5),   # bits 5..6
            "tc":   (3, 1, 0),   # bits 8..10
        })
        sig = Signature(unsigned(16), has_first_last=True)
        dut = Packetizer(hl, sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Set header fields
            ctx.set(dut.header.type, 0b10101)   # 5 bits = 21
            ctx.set(dut.header.fmt, 0b01)       # 2 bits = 1
            ctx.set(dut.header.tc, 0b110)       # 3 bits = 6
            await ctx.tick()
            # Send 2 payload beats
            await sender.send(ctx, 0x1234, first=1, last=0)
            await sender.send(ctx, 0x5678, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # 1 header beat + 2 payload beats = 3 total
            for _ in range(3):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_packetizer_bit_level.vcd")

        assert len(results) == 3

        # Header beat: bits [0:5]=10101, bits [5:7]=01, bits [7]=0, bits [8:11]=110
        # = 0b_00000_110_0_01_10101 = 0b0000011000110101
        #   bit15..11=00000, bit10..8=110, bit7=0, bit6..5=01, bit4..0=10101
        expected_hdr = (0b10101) | (0b01 << 5) | (0b110 << 8)
        assert results[0]["payload"] == expected_hdr, \
            f"Header beat: expected {expected_hdr:#06x}, got {results[0]['payload']:#06x}"
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0

        # Payload beats pass through
        assert results[1]["payload"] == 0x1234
        assert results[2]["payload"] == 0x5678
        assert results[2]["last"] == 1


# ===========================================================================
# Depacketizer tests
# ===========================================================================

class TestDepacketizer:
    """Test Depacketizer — extract header from packet start."""

    def test_depacketizer_basic(self):
        """Extract 1-byte header, pass through payload."""
        hl = HeaderLayout({"type": (8, 0)})
        sig = Signature(unsigned(8), has_first_last=True)
        dut = Depacketizer(hl, sig)
        results = []
        header_val = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send header beat
            await sender.send(ctx, 0x42, first=1, last=0)
            # Send payload beats
            await sender.send(ctx, 0xAA, first=0, last=0)
            await sender.send(ctx, 0xBB, first=0, last=0)
            await sender.send(ctx, 0xCC, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(3):  # 3 payload beats (header stripped)
                beat = await receiver.recv(ctx)
                results.append(beat)
            # Read header output
            await ctx.tick()
            header_val.append(ctx.get(dut.header.type))

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_depacketizer_basic.vcd")

        assert len(results) == 3
        assert results[0]["payload"] == 0xAA
        assert results[0]["first"] == 1  # First payload beat after header
        assert results[1]["payload"] == 0xBB
        assert results[1]["first"] == 0
        assert results[2]["payload"] == 0xCC
        assert results[2]["last"] == 1
        assert header_val[0] == 0x42

    def test_depacketizer_multi_beat_header(self):
        """Multi-beat header extraction."""
        hl = HeaderLayout({
            "src": (8, 0),
            "dst": (8, 1),
        })
        sig = Signature(unsigned(8), has_first_last=True)
        dut = Depacketizer(hl, sig)
        results = []
        header_vals = {}

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send 2 header beats
            await sender.send(ctx, 0x11, first=1, last=0)
            await sender.send(ctx, 0x22, first=0, last=0)
            # Send payload
            await sender.send(ctx, 0xDD, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            beat = await receiver.recv(ctx)
            results.append(beat)
            # Read header
            await ctx.tick()
            header_vals["src"] = ctx.get(dut.header.src)
            header_vals["dst"] = ctx.get(dut.header.dst)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_depacketizer_multi_beat.vcd")

        assert len(results) == 1
        assert results[0]["payload"] == 0xDD
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1
        assert header_vals["src"] == 0x11
        assert header_vals["dst"] == 0x22

    def test_depacketizer_packed_96bit_header_64bit_bus(self):
        """Packed mode: extract header and realign payload from packed stream."""
        # 12-byte (96-bit) header on a 64-bit stream
        hl = HeaderLayout({
            "dw0": (32, 0),
            "dw1": (32, 4),
            "dw2": (32, 8),
        })
        sig = Signature(unsigned(64), has_first_last=True)
        dut = Depacketizer(hl, sig, packed=True)
        results = []
        header_vals = {}

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Send the packed stream (as produced by Packetizer packed mode):
            # Beat 0: full header [dw0|dw1] = 0x2222222211111111 (first=1)
            await sender.send(ctx, 0x2222222211111111, first=1, last=0)
            # Beat 1: packed [dw2|payload_lower32] = 0xAAAAAAAA33333333
            await sender.send(ctx, 0xAAAAAAAA33333333, first=0, last=0)
            # Beat 2: shifted payload = 0xCCCCCCCCBBBBBBBB
            await sender.send(ctx, 0xCCCCCCCCBBBBBBBB, first=0, last=0)
            # Beat 3: drain = 0x00000000DDDDDDDD (last=1)
            await sender.send(ctx, 0x00000000DDDDDDDD, first=0, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Should receive 2 reconstructed payload beats
            for _ in range(2):
                beat = await receiver.recv(ctx)
                results.append(beat)
            # Read header
            await ctx.tick()
            header_vals["dw0"] = ctx.get(dut.header.dw0)
            header_vals["dw1"] = ctx.get(dut.header.dw1)
            header_vals["dw2"] = ctx.get(dut.header.dw2)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000,
                 vcd_name="test_depacketizer_packed_96.vcd")

        assert len(results) == 2

        # Reconstructed payload beat 0: 0xBBBBBBBBAAAAAAAA
        assert results[0]["payload"] == 0xBBBBBBBBAAAAAAAA, \
            f"Payload beat 0: expected 0xBBBBBBBBAAAAAAAA, got {results[0]['payload']:#018x}"
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0

        # Reconstructed payload beat 1: 0xDDDDDDDDCCCCCCCC
        assert results[1]["payload"] == 0xDDDDDDDDCCCCCCCC, \
            f"Payload beat 1: expected 0xDDDDDDDDCCCCCCCC, got {results[1]['payload']:#018x}"
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1

        # Verify extracted header
        assert header_vals["dw0"] == 0x11111111, \
            f"Header dw0: expected 0x11111111, got {header_vals['dw0']:#010x}"
        assert header_vals["dw1"] == 0x22222222, \
            f"Header dw1: expected 0x22222222, got {header_vals['dw1']:#010x}"
        assert header_vals["dw2"] == 0x33333333, \
            f"Header dw2: expected 0x33333333, got {header_vals['dw2']:#010x}"


# ===========================================================================
# PacketFIFO tests
# ===========================================================================

class TestPacketFIFO:
    """Test PacketFIFO — atomic packet FIFO."""

    def test_packet_fifo_basic(self):
        """Write and read a complete packet."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PacketFIFO(sig, payload_depth=16, packet_depth=4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Wait for packet to be committed
            for _ in range(10):
                await ctx.tick()
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_pkt_fifo_basic.vcd")

        assert len(results) == 3
        assert results[0]["payload"] == 0x10
        assert results[0]["first"] == 1
        assert results[1]["payload"] == 0x20
        assert results[2]["payload"] == 0x30
        assert results[2]["last"] == 1

    def test_packet_fifo_multiple_packets(self):
        """Multiple packets buffered and read back."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PacketFIFO(sig, payload_depth=32, packet_depth=4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x01, 0x02])
            await sender.send_packet(ctx, [0x03, 0x04, 0x05])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Wait for packets to be committed
            for _ in range(15):
                await ctx.tick()
            pkt1 = await receiver.recv_packet(ctx)
            results.append(pkt1)
            pkt2 = await receiver.recv_packet(ctx)
            results.append(pkt2)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pkt_fifo_multi.vcd")

        assert len(results) == 2
        assert len(results[0]) == 2
        assert results[0][0]["payload"] == 0x01
        assert results[0][1]["payload"] == 0x02
        assert len(results[1]) == 3
        assert results[1][0]["payload"] == 0x03
        assert results[1][2]["payload"] == 0x05

    def test_packet_fifo_backpressure(self):
        """Backpressure handling — receiver delays."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PacketFIFO(sig, payload_depth=16, packet_depth=4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0xAA, 0xBB])
            await sender.send_packet(ctx, [0xCC, 0xDD])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=42)
            # Wait for packets
            for _ in range(15):
                await ctx.tick()
            pkt1 = await receiver.recv_packet(ctx)
            results.append(pkt1)
            pkt2 = await receiver.recv_packet(ctx)
            results.append(pkt2)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pkt_fifo_bp.vcd")

        assert len(results) == 2
        assert results[0][0]["payload"] == 0xAA
        assert results[0][1]["payload"] == 0xBB
        assert results[1][0]["payload"] == 0xCC
        assert results[1][1]["payload"] == 0xDD


# ===========================================================================
# PacketStatus tests
# ===========================================================================

class TestPacketStatus:
    """Test PacketStatus — packet boundary tracker."""

    def test_packet_status_basic(self):
        """in_packet tracks packet boundaries."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PacketStatus(sig)
        in_packet_values = []

        async def testbench(ctx):
            # Initially not in packet
            await ctx.tick()
            in_packet_values.append(ctx.get(dut.in_packet))

            # Simulate a transfer with first=1
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 1)
            ctx.set(dut.stream.first, 1)
            ctx.set(dut.stream.last, 0)
            ctx.set(dut.stream.payload, 0xAA)
            await ctx.tick()

            # After first, should be in packet
            ctx.set(dut.stream.first, 0)
            await ctx.tick()
            in_packet_values.append(ctx.get(dut.in_packet))

            # Middle beat
            ctx.set(dut.stream.payload, 0xBB)
            await ctx.tick()
            in_packet_values.append(ctx.get(dut.in_packet))

            # Last beat
            ctx.set(dut.stream.last, 1)
            ctx.set(dut.stream.payload, 0xCC)
            await ctx.tick()

            # After last, should not be in packet
            ctx.set(dut.stream.valid, 0)
            ctx.set(dut.stream.last, 0)
            await ctx.tick()
            in_packet_values.append(ctx.get(dut.in_packet))

        _run_sim(dut, testbench, vcd_name="test_pkt_status.vcd")

        assert in_packet_values[0] == 0  # Before packet
        assert in_packet_values[1] == 1  # In packet
        assert in_packet_values[2] == 1  # Still in packet
        assert in_packet_values[3] == 0  # After packet


# ===========================================================================
# Stitcher tests
# ===========================================================================

class TestStitcher:
    """Test Stitcher — group N consecutive packets into one."""

    def test_stitcher_basic(self):
        """Stitch 2 packets into 1."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = Stitcher(sig, n=2)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Packet 1: [0x01, 0x02]
            await sender.send_packet(ctx, [0x01, 0x02])
            # Packet 2: [0x03, 0x04]
            await sender.send_packet(ctx, [0x03, 0x04])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Should receive one stitched packet of 4 beats
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_stitcher_2.vcd")

        assert len(results) == 4
        assert results[0]["payload"] == 0x01
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[1]["payload"] == 0x02
        assert results[1]["first"] == 0
        assert results[1]["last"] == 0  # Suppressed
        assert results[2]["payload"] == 0x03
        assert results[2]["first"] == 0  # Suppressed
        assert results[2]["last"] == 0
        assert results[3]["payload"] == 0x04
        assert results[3]["first"] == 0
        assert results[3]["last"] == 1

    def test_stitcher_3(self):
        """Stitch 3 packets into 1."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = Stitcher(sig, n=3)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10])
            await sender.send_packet(ctx, [0x20])
            await sender.send_packet(ctx, [0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_stitcher_3.vcd")

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


# ===========================================================================
# LastInserter tests
# ===========================================================================

class TestLastInserter:
    """Test LastInserter — inject last=1 every N beats."""

    def test_last_inserter_basic(self):
        """Insert last every 4 beats."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = LastInserter(sig, n=4)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for i in range(8):
                await sender.send(ctx, i)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(8):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_last_inserter_4.vcd")

        assert len(results) == 8
        # Beat 0: first=1, last=0
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        # Beat 3: first=0, last=1
        assert results[3]["first"] == 0
        assert results[3]["last"] == 1
        # Beat 4: first=1, last=0 (new packet)
        assert results[4]["first"] == 1
        assert results[4]["last"] == 0
        # Beat 7: first=0, last=1
        assert results[7]["first"] == 0
        assert results[7]["last"] == 1

    def test_last_inserter_with_first(self):
        """First/last framing with n=2."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = LastInserter(sig, n=2)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for i in range(4):
                await sender.send(ctx, i + 1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in range(4):
                beat = await receiver.recv(ctx)
                results.append(beat)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=100_000, vcd_name="test_last_inserter_fl.vcd")

        assert len(results) == 4
        # Packet 1: beats 0,1
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[0]["payload"] == 1
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1
        assert results[1]["payload"] == 2
        # Packet 2: beats 2,3
        assert results[2]["first"] == 1
        assert results[2]["last"] == 0
        assert results[2]["payload"] == 3
        assert results[3]["first"] == 0
        assert results[3]["last"] == 1
        assert results[3]["payload"] == 4


# ===========================================================================
# LastOnTimeout tests
# ===========================================================================

class TestLastOnTimeout:
    """Test LastOnTimeout — inject last=1 after idle timeout."""

    def test_last_on_timeout_basic(self):
        """Timeout triggers last after idle cycles."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = LastOnTimeout(sig, timeout=3)
        results = []

        async def testbench(ctx):
            # Send first beat with first=1
            ctx.set(dut.i_stream.payload, 0xAA)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.i_stream.first, 1)
            ctx.set(dut.i_stream.last, 0)
            ctx.set(dut.o_stream.ready, 1)
            await ctx.tick()

            # Deassert valid (idle)
            ctx.set(dut.i_stream.valid, 0)
            ctx.set(dut.i_stream.first, 0)

            # Wait for timeout (3 idle cycles)
            for _ in range(4):
                await ctx.tick()

            # Now send another beat — should have force_last
            ctx.set(dut.i_stream.payload, 0xBB)
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.i_stream.last, 0)

            # Sample output
            _, _, last_val = await ctx.tick().sample(dut.o_stream.last)
            results.append(last_val)

        _run_sim(dut, testbench, vcd_name="test_last_timeout.vcd")

        # The beat after timeout should have last=1
        assert results[0] == 1

    def test_last_on_timeout_no_timeout(self):
        """Normal last passes through without timeout interference."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = LastOnTimeout(sig, timeout=10)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x01, 0x02, 0x03])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_last_no_timeout.vcd")

        assert len(results) == 3
        assert results[0]["payload"] == 0x01
        assert results[0]["first"] == 1
        assert results[2]["payload"] == 0x03
        assert results[2]["last"] == 1


# ===========================================================================
# PacketFIFO abort tests
# ===========================================================================

class TestPacketFIFOAbort:
    """Test PacketFIFO abort/drop-on-error functionality."""

    def test_packet_fifo_abort_mid_packet(self):
        """Write a partial packet, assert abort, verify data is discarded,
        then write a complete packet and verify it's received correctly."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PacketFIFO(sig, payload_depth=16, packet_depth=4, has_abort=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Write partial packet (will be aborted)
            await sender.send(ctx, 0xBA, first=1, last=0)
            await sender.send(ctx, 0xDB, first=0, last=0)
            # Assert abort
            ctx.set(dut.abort, 1)
            await ctx.tick()
            ctx.set(dut.abort, 0)
            await ctx.tick()
            # Write a complete packet
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Wait for the complete packet to be committed
            for _ in range(20):
                await ctx.tick()
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pkt_fifo_abort_mid.vcd")

        # Only the complete packet should be received
        assert len(results) == 3
        assert results[0]["payload"] == 0x10
        assert results[0]["first"] == 1
        assert results[1]["payload"] == 0x20
        assert results[2]["payload"] == 0x30
        assert results[2]["last"] == 1

    def test_packet_fifo_abort_preserves_committed(self):
        """Write a complete packet, start a second packet, abort it,
        verify the first packet is still readable."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PacketFIFO(sig, payload_depth=32, packet_depth=4, has_abort=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Write a complete packet (committed)
            await sender.send_packet(ctx, [0xAA, 0xBB, 0xCC])
            # Start a second packet (will be aborted)
            await sender.send(ctx, 0xDE, first=1, last=0)
            await sender.send(ctx, 0xAD, first=0, last=0)
            # Abort the second packet
            ctx.set(dut.abort, 1)
            await ctx.tick()
            ctx.set(dut.abort, 0)
            await ctx.tick()

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Wait for the first packet to be committed
            for _ in range(20):
                await ctx.tick()
            # Read the first (committed) packet
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pkt_fifo_abort_preserve.vcd")

        # The first committed packet should be intact
        assert len(results) == 3
        assert results[0]["payload"] == 0xAA
        assert results[0]["first"] == 1
        assert results[1]["payload"] == 0xBB
        assert results[2]["payload"] == 0xCC
        assert results[2]["last"] == 1

    def test_packet_fifo_abort_immediate_reuse(self):
        """Abort a packet and immediately start a new one,
        verify the new packet works correctly."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = PacketFIFO(sig, payload_depth=16, packet_depth=4, has_abort=True)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            # Write partial packet (will be aborted)
            await sender.send(ctx, 0xFF, first=1, last=0)
            await sender.send(ctx, 0xEE, first=0, last=0)
            await sender.send(ctx, 0xDD, first=0, last=0)
            # Abort
            ctx.set(dut.abort, 1)
            await ctx.tick()
            ctx.set(dut.abort, 0)
            # Immediately start a new packet (next cycle)
            await sender.send_packet(ctx, [0x01, 0x02])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            # Wait for the new packet to be committed
            for _ in range(20):
                await ctx.tick()
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb,
                 deadline_ns=200_000, vcd_name="test_pkt_fifo_abort_reuse.vcd")

        # Only the new packet should be received
        assert len(results) == 2
        assert results[0]["payload"] == 0x01
        assert results[0]["first"] == 1
        assert results[1]["payload"] == 0x02
        assert results[1]["last"] == 1
