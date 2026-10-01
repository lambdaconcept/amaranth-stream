"""Tests for amaranth_stream.adapter (SOPEOPAdapter, StreamToSOPEOP)."""

import pytest

from amaranth.hdl import unsigned, Signal
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.adapter import SOPEOPAdapter, StreamToSOPEOP
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_adapter.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


# ---------------------------------------------------------------------------
# SOPEOPAdapter tests
# ---------------------------------------------------------------------------

class TestSOPEOPAdapter:
    """Test SOPEOPAdapter (SOP/EOP → amaranth-stream)."""

    def test_sop_eop_adapter_256bit_dword_valid(self):
        """256-bit bus with DWORD granularity valid mask.

        Send a 2-beat packet with SOP on beat 1, EOP on beat 2.
        Verify first/last mapping, valid-mask to keep expansion,
        and data pass-through.
        """
        dut = SOPEOPAdapter(256, granularity="dword")
        results = []

        # 256 bits = 8 DWORDs → valid_mask is 8 bits
        # When all DWORDs valid: mask = 0xFF → keep should be 0xFFFFFFFF (32 bytes)
        # When lower 4 DWORDs valid: mask = 0x0F → keep should be 0x0000FFFF (lower 16 bytes)

        beat1_data = 0xDEADBEEF_CAFEBABE_12345678_9ABCDEF0_11111111_22222222_33333333_44444444
        beat2_data = 0xAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDD_EEEEEEEE_FFFFFFFF_00000000_11111111
        beat1_mask = 0xFF  # all 8 DWORDs valid
        beat2_mask = 0x0F  # lower 4 DWORDs valid

        async def driver_tb(ctx):
            # Wait a cycle for initialization
            await ctx.tick()

            # Beat 1: SOP, all DWORDs valid
            ctx.set(dut.sink.data, beat1_data)
            ctx.set(dut.sink.sop, 1)
            ctx.set(dut.sink.eop, 0)
            ctx.set(dut.sink.valid_mask, beat1_mask)
            ctx.set(dut.sink.valid, 1)

            # Wait for handshake
            while True:
                _, _, ready_val = await ctx.tick().sample(dut.sink.ready)
                if ready_val:
                    break

            # Beat 2: EOP, lower 4 DWORDs valid
            ctx.set(dut.sink.data, beat2_data)
            ctx.set(dut.sink.sop, 0)
            ctx.set(dut.sink.eop, 1)
            ctx.set(dut.sink.valid_mask, beat2_mask)
            ctx.set(dut.sink.valid, 1)

            while True:
                _, _, ready_val = await ctx.tick().sample(dut.sink.ready)
                if ready_val:
                    break

            ctx.set(dut.sink.valid, 0)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.source)

            beat1 = await receiver.recv(ctx)
            results.append(beat1)

            beat2 = await receiver.recv(ctx)
            results.append(beat2)

        _run_sim(dut, driver_tb, receiver_tb,
                 vcd_name="test_sop_eop_256_dword.vcd")

        # Beat 1: first=1, last=0, data passes through
        assert results[0]["first"] == 1
        assert results[0]["last"] == 0
        assert results[0]["payload"] == beat1_data
        # DWORD mask 0xFF → all 8 DWORDs valid → keep = 0xFFFFFFFF (32 bits all 1)
        assert results[0]["keep"] == 0xFFFFFFFF

        # Beat 2: first=0, last=1, lower 4 DWORDs valid
        assert results[1]["first"] == 0
        assert results[1]["last"] == 1
        assert results[1]["payload"] == beat2_data
        # DWORD mask 0x0F → lower 4 DWORDs valid → keep = 0x0000FFFF
        assert results[1]["keep"] == 0x0000FFFF

    def test_sop_eop_adapter_backpressure(self):
        """Verify backpressure works: deassert ready, verify data holds."""
        dut = SOPEOPAdapter(64, granularity="byte")
        results = []

        test_data = 0xDEADBEEFCAFEBABE

        async def driver_tb(ctx):
            await ctx.tick()

            # Present data on the SOP/EOP side
            ctx.set(dut.sink.data, test_data)
            ctx.set(dut.sink.sop, 1)
            ctx.set(dut.sink.eop, 1)
            ctx.set(dut.sink.valid_mask, 0xFF)  # 8 bytes all valid
            ctx.set(dut.sink.valid, 1)

            # Wait for handshake (receiver will delay ready)
            while True:
                _, _, ready_val = await ctx.tick().sample(dut.sink.ready)
                if ready_val:
                    break

            ctx.set(dut.sink.valid, 0)

        async def receiver_tb(ctx):
            # Deliberately delay ready for several cycles
            ctx.set(dut.source.ready, 0)
            for _ in range(5):
                await ctx.tick()

            # Now assert ready and receive
            ctx.set(dut.source.ready, 1)

            # Wait for valid
            while True:
                _, _, valid_val, payload_val, first_val, last_val = (
                    await ctx.tick().sample(
                        dut.source.valid, dut.source.payload,
                        dut.source.first, dut.source.last))
                if valid_val:
                    results.append({
                        "payload": payload_val,
                        "first": first_val,
                        "last": last_val,
                    })
                    break

            ctx.set(dut.source.ready, 0)

        _run_sim(dut, driver_tb, receiver_tb,
                 vcd_name="test_sop_eop_backpressure.vcd")

        # Data should still be correct despite backpressure delay
        assert len(results) == 1
        assert results[0]["payload"] == test_data
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1

    def test_sop_eop_adapter_dword_reorder(self):
        """Verify DWORD reordering when enabled.

        For a 128-bit (4 DWORD) bus:
        Input:  DWORD3 | DWORD2 | DWORD1 | DWORD0  (MSB to LSB)
        Output: DWORD0 | DWORD1 | DWORD2 | DWORD3  (reversed)
        """
        dut = SOPEOPAdapter(128, granularity="dword", dword_reorder=True)
        results = []

        # 128 bits = 4 DWORDs
        # Input data: 0x_DWORD3_DWORD2_DWORD1_DWORD0
        # In amaranth bit layout: bits[0:32]=DWORD0, bits[32:64]=DWORD1, etc.
        dword0 = 0x11111111
        dword1 = 0x22222222
        dword2 = 0x33333333
        dword3 = 0x44444444
        input_data = dword0 | (dword1 << 32) | (dword2 << 64) | (dword3 << 96)

        # After DWORD reorder (reverse): bits[0:32]=DWORD3, bits[32:64]=DWORD2, etc.
        expected_data = dword3 | (dword2 << 32) | (dword1 << 64) | (dword0 << 96)

        async def driver_tb(ctx):
            await ctx.tick()

            ctx.set(dut.sink.data, input_data)
            ctx.set(dut.sink.sop, 1)
            ctx.set(dut.sink.eop, 1)
            ctx.set(dut.sink.valid_mask, 0xF)  # 4 DWORDs all valid
            ctx.set(dut.sink.valid, 1)

            while True:
                _, _, ready_val = await ctx.tick().sample(dut.sink.ready)
                if ready_val:
                    break

            ctx.set(dut.sink.valid, 0)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.source)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, driver_tb, receiver_tb,
                 vcd_name="test_sop_eop_dword_reorder.vcd")

        assert results[0]["payload"] == expected_data
        assert results[0]["first"] == 1
        assert results[0]["last"] == 1

    def test_sop_eop_adapter_byte_granularity(self):
        """Byte granularity: valid_mask maps 1:1 to keep."""
        dut = SOPEOPAdapter(64, granularity="byte")
        results = []

        test_data = 0x0102030405060708
        test_mask = 0b10101010  # bytes 1,3,5,7 valid

        async def driver_tb(ctx):
            await ctx.tick()

            ctx.set(dut.sink.data, test_data)
            ctx.set(dut.sink.sop, 1)
            ctx.set(dut.sink.eop, 1)
            ctx.set(dut.sink.valid_mask, test_mask)
            ctx.set(dut.sink.valid, 1)

            while True:
                _, _, ready_val = await ctx.tick().sample(dut.sink.ready)
                if ready_val:
                    break

            ctx.set(dut.sink.valid, 0)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.source)
            beat = await receiver.recv(ctx)
            results.append(beat)

        _run_sim(dut, driver_tb, receiver_tb,
                 vcd_name="test_sop_eop_byte_gran.vcd")

        assert results[0]["payload"] == test_data
        assert results[0]["keep"] == test_mask  # 1:1 for byte granularity


# ---------------------------------------------------------------------------
# StreamToSOPEOP tests
# ---------------------------------------------------------------------------

class TestStreamToSOPEOP:
    """Test StreamToSOPEOP (amaranth-stream → SOP/EOP)."""

    def test_stream_to_sop_eop_basic(self):
        """Verify reverse direction: stream first/last → SOP/EOP, keep → valid_mask."""
        dut = StreamToSOPEOP(128, granularity="dword")
        results = []

        # 128 bits = 4 DWORDs, keep is 16 bits (16 bytes)
        beat1_data = 0x11111111_22222222_33333333_44444444
        beat1_keep = 0xFFFF  # all 16 bytes valid → dword mask = 0xF

        beat2_data = 0xAAAAAAAA_BBBBBBBB_CCCCCCCC_DDDDDDDD
        beat2_keep = 0x00FF  # lower 8 bytes valid → lower 2 DWORDs → dword mask = 0x3

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.sink)
            await sender.send(ctx, beat1_data, first=1, last=0, keep=beat1_keep)
            await sender.send(ctx, beat2_data, first=0, last=1, keep=beat2_keep)

        async def receiver_tb(ctx):
            # Manually receive from the SOP/EOP output
            ctx.set(dut.source.ready, 1)

            for _ in range(2):
                while True:
                    _, _, valid_val, data_val, sop_val, eop_val, mask_val = (
                        await ctx.tick().sample(
                            dut.source.valid, dut.source.data,
                            dut.source.sop, dut.source.eop,
                            dut.source.valid_mask))
                    if valid_val:
                        results.append({
                            "data": data_val,
                            "sop": sop_val,
                            "eop": eop_val,
                            "valid_mask": mask_val,
                        })
                        break

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_stream_to_sop_eop.vcd")

        # Beat 1: first→sop, not last→no eop
        assert results[0]["sop"] == 1
        assert results[0]["eop"] == 0
        assert results[0]["data"] == beat1_data
        assert results[0]["valid_mask"] == 0xF  # all 4 DWORDs valid

        # Beat 2: not first→no sop, last→eop
        assert results[1]["sop"] == 0
        assert results[1]["eop"] == 1
        assert results[1]["data"] == beat2_data
        assert results[1]["valid_mask"] == 0x3  # lower 2 DWORDs valid

    def test_stream_to_sop_eop_dword_reorder(self):
        """Verify DWORD reordering in reverse direction."""
        dut = StreamToSOPEOP(128, granularity="dword", dword_reorder=True)
        results = []

        dword0 = 0x11111111
        dword1 = 0x22222222
        dword2 = 0x33333333
        dword3 = 0x44444444
        input_data = dword0 | (dword1 << 32) | (dword2 << 64) | (dword3 << 96)

        # After DWORD reorder: bits[0:32]=DWORD3, bits[32:64]=DWORD2, etc.
        expected_data = dword3 | (dword2 << 32) | (dword1 << 64) | (dword0 << 96)

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.sink)
            await sender.send(ctx, input_data, first=1, last=1, keep=0xFFFF)

        async def receiver_tb(ctx):
            ctx.set(dut.source.ready, 1)

            while True:
                _, _, valid_val, data_val = (
                    await ctx.tick().sample(
                        dut.source.valid, dut.source.data))
                if valid_val:
                    results.append({"data": data_val})
                    break

        _run_sim(dut, sender_tb, receiver_tb,
                 vcd_name="test_stream_to_sop_eop_reorder.vcd")

        assert results[0]["data"] == expected_data


# ---------------------------------------------------------------------------
# Constructor validation tests
# ---------------------------------------------------------------------------

class TestAdapterValidation:
    """Test constructor parameter validation."""

    def test_invalid_data_width(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            SOPEOPAdapter(24)

    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="granularity"):
            SOPEOPAdapter(64, granularity="nibble")

    def test_stream_to_sop_eop_invalid_data_width(self):
        with pytest.raises(ValueError, match="multiple of 32"):
            StreamToSOPEOP(16)
