"""Tests for amaranth_stream.monitor (StreamMonitor, StreamChecker)."""

import pytest

from amaranth import *
from amaranth.hdl import unsigned, Signal
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out, connect
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import Signature
from amaranth_stream.monitor import StreamMonitor, StreamChecker, StreamProtocolChecker
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_monitor.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


# ---------------------------------------------------------------------------
# StreamMonitor tests
# ---------------------------------------------------------------------------

class TestStreamMonitor:
    """Test StreamMonitor (performance counters)."""

    def test_monitor_transfer_count(self):
        """Count completed transfers."""
        sig = Signature(unsigned(8))
        dut = StreamMonitor(sig)
        transfer_count = []

        async def tb(ctx):
            # Send 5 beats
            for i in range(5):
                ctx.set(dut.i_stream.payload, i)
                ctx.set(dut.i_stream.valid, 1)
                ctx.set(dut.o_stream.ready, 1)
                await ctx.tick()

            # Deassert before waiting for counter to update
            ctx.set(dut.i_stream.valid, 0)
            ctx.set(dut.o_stream.ready, 0)
            await ctx.tick()
            transfer_count.append(ctx.get(dut.transfer_count))

        _run_sim(dut, tb, vcd_name="test_mon_transfer.vcd")
        assert transfer_count[0] == 5

    def test_monitor_stall_count(self):
        """Count stall cycles (valid & ~ready)."""
        sig = Signature(unsigned(8))
        dut = StreamMonitor(sig)
        stall_count = []

        async def tb(ctx):
            # Create 3 stall cycles: valid=1, ready=0
            for _ in range(3):
                ctx.set(dut.i_stream.valid, 1)
                ctx.set(dut.o_stream.ready, 0)
                await ctx.tick()

            # Then do a transfer to end the stall
            ctx.set(dut.i_stream.valid, 1)
            ctx.set(dut.o_stream.ready, 1)
            await ctx.tick()

            # Wait for counter to update
            ctx.set(dut.i_stream.valid, 0)
            await ctx.tick()
            stall_count.append(ctx.get(dut.stall_count))

        _run_sim(dut, tb, vcd_name="test_mon_stall.vcd")
        assert stall_count[0] == 3

    def test_monitor_idle_count(self):
        """Count idle cycles (~valid & ready)."""
        sig = Signature(unsigned(8))
        dut = StreamMonitor(sig)
        idle_count = []

        async def tb(ctx):
            # Create 4 idle cycles: valid=0, ready=1
            for _ in range(4):
                ctx.set(dut.i_stream.valid, 0)
                ctx.set(dut.o_stream.ready, 1)
                await ctx.tick()

            # Wait for counter to update
            ctx.set(dut.o_stream.ready, 0)
            await ctx.tick()
            idle_count.append(ctx.get(dut.idle_count))

        _run_sim(dut, tb, vcd_name="test_mon_idle.vcd")
        assert idle_count[0] == 4

    def test_monitor_packet_count(self):
        """Count completed packets (valid & ready & last)."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = StreamMonitor(sig)
        packet_count = []

        async def tb(ctx):
            # Send 2 packets: [A, B, C(last)], [D, E(last)]
            for val, last in [(0xA, 0), (0xB, 0), (0xC, 1), (0xD, 0), (0xE, 1)]:
                ctx.set(dut.i_stream.payload, val)
                ctx.set(dut.i_stream.valid, 1)
                ctx.set(dut.i_stream.last, last)
                ctx.set(dut.o_stream.ready, 1)
                await ctx.tick()

            ctx.set(dut.i_stream.valid, 0)
            await ctx.tick()
            packet_count.append(ctx.get(dut.packet_count))

        _run_sim(dut, tb, vcd_name="test_mon_packet.vcd")
        assert packet_count[0] == 2

    def test_monitor_clear(self):
        """Clear resets all counters to 0."""
        sig = Signature(unsigned(8))
        dut = StreamMonitor(sig)
        counts_after_clear = []

        async def tb(ctx):
            # Generate some transfers
            for _ in range(3):
                ctx.set(dut.i_stream.valid, 1)
                ctx.set(dut.o_stream.ready, 1)
                await ctx.tick()

            # Wait for counters to update
            ctx.set(dut.i_stream.valid, 0)
            ctx.set(dut.o_stream.ready, 0)
            await ctx.tick()

            # Assert clear
            ctx.set(dut.clear, 1)
            await ctx.tick()
            ctx.set(dut.clear, 0)
            await ctx.tick()

            counts_after_clear.append(ctx.get(dut.transfer_count))
            counts_after_clear.append(ctx.get(dut.stall_count))
            counts_after_clear.append(ctx.get(dut.idle_count))

        _run_sim(dut, tb, vcd_name="test_mon_clear.vcd")
        assert counts_after_clear == [0, 0, 0]

    def test_monitor_passthrough(self):
        """Data passes through monitor unchanged."""
        sig = Signature(unsigned(8))
        dut = StreamMonitor(sig)
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

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_mon_passthrough.vcd")
        assert results == expected


# ---------------------------------------------------------------------------
# StreamChecker tests
# ---------------------------------------------------------------------------

class _CheckerTestHarness(wiring.Component):
    """Test harness that wraps a StreamChecker with a driveable stream."""

    def __init__(self, signature):
        self._stream_sig = signature
        super().__init__({
            "stream": Out(signature),
            "error": Out(1),
        })
        self._checker = StreamChecker(signature)

    def elaborate(self, platform):
        m = Module()
        m.submodules.checker = checker = self._checker

        # Connect our stream output to checker's stream input
        m.d.comb += [
            checker.stream.payload.eq(self.stream.payload),
            checker.stream.valid.eq(self.stream.valid),
            checker.stream.ready.eq(self.stream.ready),
        ]
        if self._stream_sig.has_first_last:
            m.d.comb += [
                checker.stream.first.eq(self.stream.first),
                checker.stream.last.eq(self.stream.last),
            ]
        m.d.comb += self.error.eq(checker.error)

        return m


class TestStreamChecker:
    """Test StreamChecker (protocol assertion checker)."""

    def test_checker_valid_stability(self):
        """Detect valid deassert violation (valid drops without transfer)."""
        sig = Signature(unsigned(8))
        dut = _CheckerTestHarness(sig)
        error_detected = []

        async def tb(ctx):
            # Assert valid with payload
            ctx.set(dut.stream.payload, 0x42)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 0)  # no transfer
            await ctx.tick()

            # Deassert valid without transfer — violation!
            ctx.set(dut.stream.valid, 0)
            await ctx.tick()

            # Wait for error to propagate
            await ctx.tick()
            error_detected.append(ctx.get(dut.error))

        _run_sim(dut, tb, vcd_name="test_checker_valid.vcd")
        assert error_detected[0] == 1

    def test_checker_payload_stability(self):
        """Detect payload change violation (payload changes while valid & ~ready)."""
        sig = Signature(unsigned(8))
        dut = _CheckerTestHarness(sig)
        error_detected = []

        async def tb(ctx):
            # Assert valid with payload
            ctx.set(dut.stream.payload, 0x42)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 0)  # no transfer
            await ctx.tick()

            # Change payload without transfer — violation!
            ctx.set(dut.stream.payload, 0x99)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 0)
            await ctx.tick()

            # Wait for error to propagate
            await ctx.tick()
            error_detected.append(ctx.get(dut.error))

        _run_sim(dut, tb, vcd_name="test_checker_payload.vcd")
        assert error_detected[0] == 1

    def test_checker_no_error(self):
        """No error on correct protocol usage."""
        sig = Signature(unsigned(8))
        dut = _CheckerTestHarness(sig)
        error_detected = []

        async def tb(ctx):
            # Correct: assert valid, then transfer happens
            ctx.set(dut.stream.payload, 0x42)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 1)  # transfer happens
            await ctx.tick()

            # Next beat
            ctx.set(dut.stream.payload, 0x99)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 1)
            await ctx.tick()

            # Deassert valid after transfer (OK)
            ctx.set(dut.stream.valid, 0)
            await ctx.tick()

            await ctx.tick()
            error_detected.append(ctx.get(dut.error))

        _run_sim(dut, tb, vcd_name="test_checker_ok.vcd")
        assert error_detected[0] == 0


# ---------------------------------------------------------------------------
# StreamProtocolChecker tests
# ---------------------------------------------------------------------------

class _ProtocolCheckerTestHarness(wiring.Component):
    """Test harness that wraps a StreamProtocolChecker with a driveable stream."""

    def __init__(self, signature):
        self._stream_sig = signature
        super().__init__({
            "stream": Out(signature),
            "error": Out(1),
            "error_code": Out(2),
        })
        self._checker = StreamProtocolChecker(signature)

    def elaborate(self, platform):
        m = Module()
        m.submodules.checker = checker = self._checker

        # Connect our stream output to checker's stream input
        m.d.comb += [
            checker.stream.payload.eq(self.stream.payload),
            checker.stream.valid.eq(self.stream.valid),
            checker.stream.ready.eq(self.stream.ready),
        ]
        if self._stream_sig.has_first_last:
            m.d.comb += [
                checker.stream.first.eq(self.stream.first),
                checker.stream.last.eq(self.stream.last),
            ]
        m.d.comb += [
            self.error.eq(checker.error),
            self.error_code.eq(checker.error_code),
        ]

        return m


class TestStreamProtocolChecker:
    """Test StreamProtocolChecker (protocol violation detection with error codes)."""

    def test_protocol_checker_valid_stream(self):
        """Send a well-formed packet, verify no errors."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = _ProtocolCheckerTestHarness(sig)
        errors = []

        async def tb(ctx):
            # Send a well-formed packet: first=1 on beat 0, last=1 on beat 2
            for i, (first, last) in enumerate([(1, 0), (0, 0), (0, 1)]):
                ctx.set(dut.stream.payload, 0x10 + i)
                ctx.set(dut.stream.valid, 1)
                ctx.set(dut.stream.ready, 1)
                ctx.set(dut.stream.first, first)
                ctx.set(dut.stream.last, last)
                await ctx.tick()
                errors.append(ctx.get(dut.error))

            # Deassert after transfer (valid protocol)
            ctx.set(dut.stream.valid, 0)
            ctx.set(dut.stream.ready, 0)
            await ctx.tick()
            errors.append(ctx.get(dut.error))

            # Another tick to be sure
            await ctx.tick()
            errors.append(ctx.get(dut.error))

        _run_sim(dut, tb, vcd_name="test_protchk_valid.vcd")
        assert all(e == 0 for e in errors), f"Unexpected errors: {errors}"

    def test_protocol_checker_first_without_last(self):
        """Assert first without preceding last, verify error_code=1."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = _ProtocolCheckerTestHarness(sig)
        error_code_detected = []

        async def tb(ctx):
            # First transfer: first=1, last=0 — valid start of packet
            ctx.set(dut.stream.payload, 0xAA)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 1)
            ctx.set(dut.stream.first, 1)
            ctx.set(dut.stream.last, 0)
            await ctx.tick()

            # Second transfer: first=0, last=0 — middle of packet, OK
            ctx.set(dut.stream.payload, 0xBB)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 1)
            ctx.set(dut.stream.first, 0)
            ctx.set(dut.stream.last, 0)
            await ctx.tick()

            # Third transfer: first=1, last=0 — VIOLATION! first without
            # preceding last (previous transfer had last=0)
            ctx.set(dut.stream.payload, 0xCC)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 1)
            ctx.set(dut.stream.first, 1)
            ctx.set(dut.stream.last, 0)
            await ctx.tick()

            # Check error on this cycle (the violation is combinational)
            error_code_detected.append(ctx.get(dut.error))
            error_code_detected.append(ctx.get(dut.error_code))

        _run_sim(dut, tb, vcd_name="test_protchk_first_no_last.vcd")
        assert error_code_detected[0] == 1, f"Expected error, got {error_code_detected[0]}"
        assert error_code_detected[1] == 1, f"Expected error_code=1, got {error_code_detected[1]}"

    def test_protocol_checker_valid_deassert(self):
        """Deassert valid while ready=0, verify error_code=2."""
        sig = Signature(unsigned(8))
        dut = _ProtocolCheckerTestHarness(sig)
        error_code_detected = []

        async def tb(ctx):
            # Assert valid with no ready (stall)
            ctx.set(dut.stream.payload, 0x42)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 0)
            await ctx.tick()

            # Deassert valid without transfer — violation!
            # After the tick, prev_valid=1, prev_ready=0 (stall).
            # Now set valid=0 and read the combinational error immediately.
            ctx.set(dut.stream.valid, 0)
            # The combinational check fires: prev_stall=1 & ~valid=1 → error
            error_code_detected.append(ctx.get(dut.error))
            error_code_detected.append(ctx.get(dut.error_code))

        _run_sim(dut, tb, vcd_name="test_protchk_valid_deassert.vcd")
        assert error_code_detected[0] == 1, f"Expected error, got {error_code_detected[0]}"
        assert error_code_detected[1] == 2, f"Expected error_code=2, got {error_code_detected[1]}"

    def test_protocol_checker_payload_change(self):
        """Change payload while valid=1 and ready=0, verify error_code=3."""
        sig = Signature(unsigned(8))
        dut = _ProtocolCheckerTestHarness(sig)
        error_code_detected = []

        async def tb(ctx):
            # Assert valid with payload, no ready (stall)
            ctx.set(dut.stream.payload, 0x42)
            ctx.set(dut.stream.valid, 1)
            ctx.set(dut.stream.ready, 0)
            await ctx.tick()

            # Change payload without transfer — violation!
            # After the tick, prev_valid=1, prev_ready=0 (stall),
            # prev_payload=0x42.  Now change payload and read error.
            ctx.set(dut.stream.payload, 0x99)
            # The combinational check fires: prev_stall=1 & payload!=prev_payload
            error_code_detected.append(ctx.get(dut.error))
            error_code_detected.append(ctx.get(dut.error_code))

        _run_sim(dut, tb, vcd_name="test_protchk_payload_change.vcd")
        assert error_code_detected[0] == 1, f"Expected error, got {error_code_detected[0]}"
        assert error_code_detected[1] == 3, f"Expected error_code=3, got {error_code_detected[1]}"
