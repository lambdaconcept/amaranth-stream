"""Tests for amaranth_stream._base (Signature, Interface, core_to_extended)."""

import math
import pytest

from amaranth.hdl import Shape, Signal, Const, unsigned, signed
from amaranth.lib.data import StructLayout
from amaranth.lib import wiring

from amaranth_stream._base import Signature, Interface, core_to_extended


# ---------------------------------------------------------------------------
# Signature creation
# ---------------------------------------------------------------------------


class TestSignatureCreation:
    """Test Signature construction with various parameter combinations."""

    def test_basic_int_shape(self):
        sig = Signature(8)
        assert sig.payload_shape == 8
        assert sig.payload_width == 8
        assert "payload" in sig.members
        assert "valid" in sig.members
        assert "ready" in sig.members

    def test_unsigned_shape(self):
        sig = Signature(unsigned(16))
        assert sig.payload_width == 16

    def test_signed_shape(self):
        sig = Signature(signed(12))
        assert sig.payload_width == 12

    def test_struct_layout_shape(self):
        layout = StructLayout({"r": 8, "g": 8, "b": 8})
        sig = Signature(layout)
        assert sig.payload_shape is layout
        assert sig.payload_width == 24

    def test_default_flags(self):
        sig = Signature(8)
        assert sig.always_valid is False
        assert sig.always_ready is False
        assert sig.has_first_last is False
        assert sig.param_shape is None
        assert sig.has_keep is False

    def test_always_valid(self):
        sig = Signature(8, always_valid=True)
        assert sig.always_valid is True
        assert sig.always_ready is False

    def test_always_ready(self):
        sig = Signature(8, always_ready=True)
        assert sig.always_valid is False
        assert sig.always_ready is True

    def test_always_valid_and_ready(self):
        sig = Signature(8, always_valid=True, always_ready=True)
        assert sig.always_valid is True
        assert sig.always_ready is True


# ---------------------------------------------------------------------------
# Optional members
# ---------------------------------------------------------------------------


class TestSignatureOptionalMembers:
    """Test has_first_last, param_shape, has_keep."""

    def test_has_first_last(self):
        sig = Signature(8, has_first_last=True)
        assert sig.has_first_last is True
        assert "first" in sig.members
        assert "last" in sig.members

    def test_no_first_last_by_default(self):
        sig = Signature(8)
        assert "first" not in sig.members
        assert "last" not in sig.members

    def test_param_shape_int(self):
        sig = Signature(8, param_shape=4)
        assert sig.param_shape == 4
        assert "param" in sig.members

    def test_param_shape_unsigned(self):
        sig = Signature(8, param_shape=unsigned(6))
        assert sig.param_shape == unsigned(6)
        assert "param" in sig.members

    def test_param_shape_struct(self):
        layout = StructLayout({"cmd": 4, "flags": 2})
        sig = Signature(8, param_shape=layout)
        assert sig.param_shape is layout
        assert "param" in sig.members

    def test_no_param_by_default(self):
        sig = Signature(8)
        assert "param" not in sig.members

    def test_has_keep_8bit(self):
        sig = Signature(8, has_keep=True)
        assert sig.has_keep is True
        assert "keep" in sig.members
        # 8 bits / 8 = 1 byte => keep width = 1
        keep_shape = Shape.cast(sig.members["keep"].shape)
        assert keep_shape.width == 1

    def test_has_keep_32bit(self):
        sig = Signature(unsigned(32), has_keep=True)
        keep_shape = Shape.cast(sig.members["keep"].shape)
        assert keep_shape.width == 4  # 32 / 8 = 4

    def test_has_keep_24bit(self):
        sig = Signature(unsigned(24), has_keep=True)
        keep_shape = Shape.cast(sig.members["keep"].shape)
        assert keep_shape.width == 3  # 24 / 8 = 3

    def test_has_keep_10bit(self):
        sig = Signature(unsigned(10), has_keep=True)
        keep_shape = Shape.cast(sig.members["keep"].shape)
        assert keep_shape.width == 2  # ceil(10 / 8) = 2

    def test_no_keep_by_default(self):
        sig = Signature(8)
        assert "keep" not in sig.members

    def test_all_options(self):
        sig = Signature(
            unsigned(32),
            always_valid=True,
            always_ready=True,
            has_first_last=True,
            param_shape=4,
            has_keep=True,
        )
        expected_members = {
            "payload",
            "valid",
            "ready",
            "first",
            "last",
            "param",
            "keep",
        }
        assert set(sig.members.keys()) == expected_members


# ---------------------------------------------------------------------------
# Member directions
# ---------------------------------------------------------------------------


class TestSignatureMemberDirections:
    """Verify Out/In directions of members."""

    def test_payload_is_out(self):
        sig = Signature(8)
        assert sig.members["payload"].flow == wiring.Out

    def test_valid_is_out(self):
        sig = Signature(8)
        assert sig.members["valid"].flow == wiring.Out

    def test_ready_is_in(self):
        sig = Signature(8)
        assert sig.members["ready"].flow == wiring.In

    def test_first_is_out(self):
        sig = Signature(8, has_first_last=True)
        assert sig.members["first"].flow == wiring.Out

    def test_last_is_out(self):
        sig = Signature(8, has_first_last=True)
        assert sig.members["last"].flow == wiring.Out

    def test_param_is_out(self):
        sig = Signature(8, param_shape=4)
        assert sig.members["param"].flow == wiring.Out

    def test_keep_is_out(self):
        sig = Signature(8, has_keep=True)
        assert sig.members["keep"].flow == wiring.Out


# ---------------------------------------------------------------------------
# Interface creation
# ---------------------------------------------------------------------------


class TestInterface:
    """Test Interface creation and properties."""

    def test_create_from_signature(self):
        sig = Signature(8)
        intf = sig.create()
        assert isinstance(intf, Interface)
        assert intf.signature is sig

    def test_payload_is_signal(self):
        sig = Signature(8)
        intf = sig.create()
        assert isinstance(intf.payload, Signal)

    def test_valid_is_signal(self):
        sig = Signature(8)
        intf = sig.create()
        assert isinstance(intf.valid, Signal)

    def test_ready_is_signal(self):
        sig = Signature(8)
        intf = sig.create()
        assert isinstance(intf.ready, Signal)

    def test_p_shortcut(self):
        sig = Signature(8)
        intf = sig.create()
        assert intf.p is intf.payload

    def test_always_valid_const(self):
        sig = Signature(8, always_valid=True)
        intf = sig.create()
        assert isinstance(intf.valid, Const)

    def test_always_ready_const(self):
        sig = Signature(8, always_ready=True)
        intf = sig.create()
        assert isinstance(intf.ready, Const)

    def test_first_last_present(self):
        sig = Signature(8, has_first_last=True)
        intf = sig.create()
        assert isinstance(intf.first, Signal)
        assert isinstance(intf.last, Signal)

    def test_param_present(self):
        sig = Signature(8, param_shape=4)
        intf = sig.create()
        assert hasattr(intf, "param")

    def test_keep_present(self):
        sig = Signature(unsigned(32), has_keep=True)
        intf = sig.create()
        assert hasattr(intf, "keep")

    def test_type_error_on_wrong_signature(self):
        with pytest.raises(TypeError, match="stream.Signature"):
            Interface(wiring.Signature({"a": wiring.Out(1)}))

    def test_repr(self):
        sig = Signature(8)
        intf = sig.create()
        r = repr(intf)
        assert "stream.Interface" in r
        assert "payload=" in r
        assert "valid=" in r
        assert "ready=" in r


# ---------------------------------------------------------------------------
# Signature equality
# ---------------------------------------------------------------------------


class TestSignatureEquality:
    """Test __eq__ for Signature."""

    def test_equal_basic(self):
        assert Signature(8) == Signature(8)

    def test_not_equal_shape(self):
        assert Signature(8) != Signature(16)

    def test_not_equal_always_valid(self):
        assert Signature(8) != Signature(8, always_valid=True)

    def test_not_equal_always_ready(self):
        assert Signature(8) != Signature(8, always_ready=True)

    def test_not_equal_first_last(self):
        assert Signature(8) != Signature(8, has_first_last=True)

    def test_not_equal_param(self):
        assert Signature(8) != Signature(8, param_shape=4)

    def test_not_equal_keep(self):
        assert Signature(8) != Signature(8, has_keep=True)

    def test_equal_all_options(self):
        kwargs = dict(
            always_valid=True,
            always_ready=True,
            has_first_last=True,
            param_shape=4,
            has_keep=True,
        )
        assert Signature(unsigned(32), **kwargs) == Signature(unsigned(32), **kwargs)

    def test_not_equal_to_non_signature(self):
        assert Signature(8) != "not a signature"

    def test_not_equal_to_wiring_signature(self):
        assert Signature(8) != wiring.Signature({"payload": wiring.Out(8)})


# ---------------------------------------------------------------------------
# Signature repr
# ---------------------------------------------------------------------------


class TestSignatureRepr:
    """Test __repr__ for Signature."""

    def test_basic_repr(self):
        assert repr(Signature(8)) == "stream.Signature(8)"

    def test_unsigned_repr(self):
        r = repr(Signature(unsigned(16)))
        assert "stream.Signature(unsigned(16))" == r

    def test_always_valid_repr(self):
        r = repr(Signature(8, always_valid=True))
        assert "always_valid=True" in r

    def test_always_ready_repr(self):
        r = repr(Signature(8, always_ready=True))
        assert "always_ready=True" in r

    def test_first_last_repr(self):
        r = repr(Signature(8, has_first_last=True))
        assert "has_first_last=True" in r

    def test_param_repr(self):
        r = repr(Signature(8, param_shape=4))
        assert "param_shape=4" in r

    def test_keep_repr(self):
        r = repr(Signature(8, has_keep=True))
        assert "has_keep=True" in r


# ---------------------------------------------------------------------------
# Signature is a wiring.Signature subclass
# ---------------------------------------------------------------------------


class TestSignatureIsWiringSubclass:
    """Verify that our Signature is a proper wiring.Signature subclass."""

    def test_isinstance(self):
        sig = Signature(8)
        assert isinstance(sig, wiring.Signature)

    def test_has_members(self):
        sig = Signature(8)
        assert hasattr(sig, "members")

    def test_can_flip(self):
        sig = Signature(8)
        flipped = sig.flip()
        # Flipped members should have reversed directions
        assert flipped.members["payload"].flow == wiring.In
        assert flipped.members["valid"].flow == wiring.In
        assert flipped.members["ready"].flow == wiring.Out


# ---------------------------------------------------------------------------
# core_to_extended
# ---------------------------------------------------------------------------


class TestCoreToExtended:
    """Test the core_to_extended helper."""

    def test_basic_conversion(self):
        from amaranth.lib.stream import Signature as CoreSignature

        core = CoreSignature(16)
        ext = core_to_extended(core)
        assert isinstance(ext, Signature)
        assert ext.payload_width == 16

    def test_preserves_always_valid(self):
        from amaranth.lib.stream import Signature as CoreSignature

        core = CoreSignature(8, always_valid=True)
        ext = core_to_extended(core)
        assert ext.always_valid is True

    def test_preserves_always_ready(self):
        from amaranth.lib.stream import Signature as CoreSignature

        core = CoreSignature(8, always_ready=True)
        ext = core_to_extended(core)
        assert ext.always_ready is True

    def test_no_extra_members(self):
        from amaranth.lib.stream import Signature as CoreSignature

        core = CoreSignature(8)
        ext = core_to_extended(core)
        assert ext.has_first_last is False
        assert ext.param_shape is None
        assert ext.has_keep is False

    def test_type_error_on_wrong_input(self):
        with pytest.raises(TypeError, match="amaranth.lib.stream.Signature"):
            core_to_extended(Signature(8))

    def test_type_error_on_string(self):
        with pytest.raises(TypeError):
            core_to_extended("not a signature")


# ---------------------------------------------------------------------------
# connect_streams
# ---------------------------------------------------------------------------

from amaranth import Module
from amaranth.hdl import unsigned
from amaranth.lib.wiring import In, Out
from amaranth.sim import Simulator
from tests._compat import Period

from amaranth_stream._base import connect_streams
from amaranth_stream.sim import StreamSimSender, StreamSimReceiver


def _run_sim(dut, *testbenches, deadline_ns=50_000, vcd_name="test_connect.vcd"):
    """Helper to run a simulation with one or more testbenches."""
    sim = Simulator(dut)
    sim.add_clock(Period(MHz=10))
    for tb in testbenches:
        sim.add_testbench(tb)
    with sim.write_vcd(vcd_name):
        sim.run_until(Period(ns=deadline_ns))


from amaranth.hdl import ClockDomain


class _ConnectBridge(wiring.Component):
    """Trivial component that connects i_stream → o_stream via connect_streams."""

    def __init__(self, src_sig, dst_sig=None, *, exclude=None):
        self._src_sig = src_sig
        self._dst_sig = dst_sig if dst_sig is not None else src_sig
        self._exclude = exclude
        super().__init__(
            {
                "i_stream": In(self._src_sig),
                "o_stream": Out(self._dst_sig),
            }
        )

    def elaborate(self, platform):
        m = Module()
        # Add a sync domain so the simulator can add a clock
        m.domains += ClockDomain("sync")
        m.d.comb += connect_streams(
            wiring.flipped(self.i_stream),
            wiring.flipped(self.o_stream),
            exclude=self._exclude,
        )
        return m


class TestConnectStreamsBasic:
    """test_connect_streams_basic — connect two simple streams, send data, verify."""

    def test_basic_data_transfer(self):
        sig = Signature(unsigned(8))
        dut = _ConnectBridge(sig)
        results = []
        expected = [0x10, 0x20, 0x30, 0x40, 0x50]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_basic.vcd")
        assert results == expected

    def test_basic_backpressure(self):
        sig = Signature(unsigned(8))
        dut = _ConnectBridge(sig)
        results = []
        expected = [0xAA, 0xBB, 0xCC, 0xDD]

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream, random_valid=True, seed=11)
            for val in expected:
                await sender.send(ctx, val)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream, random_ready=True, seed=22)
            for _ in expected:
                beat = await receiver.recv(ctx)
                results.append(beat["payload"])

        _run_sim(
            dut,
            sender_tb,
            receiver_tb,
            deadline_ns=100_000,
            vcd_name="test_cs_basic_bp.vcd",
        )
        assert results == expected


class TestConnectStreamsWithFirstLast:
    """test_connect_streams_with_first_last — verify first/last propagation."""

    def test_first_last_propagation(self):
        sig = Signature(unsigned(8), has_first_last=True)
        dut = _ConnectBridge(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send_packet(ctx, [0x10, 0x20, 0x30])

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            packet = await receiver.recv_packet(ctx)
            results.extend(packet)

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_first_last.vcd")

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


class TestConnectStreamsWithKeepParam:
    """test_connect_streams_with_keep_param — verify keep and param propagation."""

    def test_param_propagation(self):
        sig = Signature(unsigned(8), param_shape=unsigned(4))
        dut = _ConnectBridge(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, param=0x5)
            await sender.send(ctx, 0xBB, param=0xA)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_param.vcd")

        assert results[0]["payload"] == 0xAA
        assert results[0]["param"] == 0x5
        assert results[1]["payload"] == 0xBB
        assert results[1]["param"] == 0xA

    def test_keep_propagation(self):
        sig = Signature(unsigned(16), has_keep=True)
        dut = _ConnectBridge(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x1234, keep=0x3)
            await sender.send(ctx, 0x5678, keep=0x1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_keep.vcd")

        assert results[0]["payload"] == 0x1234
        assert results[0]["keep"] == 0x3
        assert results[1]["payload"] == 0x5678
        assert results[1]["keep"] == 0x1

    def test_all_optional_signals(self):
        sig = Signature(
            unsigned(8), has_first_last=True, param_shape=unsigned(4), has_keep=True
        )
        dut = _ConnectBridge(sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, first=1, last=0, param=0x3, keep=0x1)
            await sender.send(ctx, 0xBB, first=0, last=1, param=0x7, keep=0x1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_all_opt.vcd")

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


class TestConnectStreamsExclude:
    """test_connect_streams_exclude — verify exclude parameter works."""

    def test_exclude_param(self):
        """Excluding param should leave it unconnected (default 0)."""
        sig = Signature(unsigned(8), param_shape=unsigned(4))
        dut = _ConnectBridge(sig, exclude={"param"})
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xAA, param=0xF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_exclude.vcd")

        assert results[0]["payload"] == 0xAA
        # param should be 0 (unconnected, default)
        assert results[0]["param"] == 0

    def test_exclude_first_last(self):
        """Excluding first and last should leave them unconnected."""
        sig = Signature(unsigned(8), has_first_last=True)
        dut = _ConnectBridge(sig, exclude={"first", "last"})
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x42, first=1, last=1)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_exclude_fl.vcd")

        assert results[0]["payload"] == 0x42
        assert results[0]["first"] == 0
        assert results[0]["last"] == 0


class TestConnectStreamsMismatchedOptional:
    """test_connect_streams_mismatched_optional — graceful handling of mismatched fields."""

    def test_src_has_first_last_dst_does_not(self):
        """src has first/last but dst doesn't — should connect without error."""
        src_sig = Signature(unsigned(8), has_first_last=True)
        dst_sig = Signature(unsigned(8))
        dut = _ConnectBridge(src_sig, dst_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xDE, first=1, last=1)
            await sender.send(ctx, 0xAD)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_mismatch1.vcd")

        assert results[0]["payload"] == 0xDE
        assert results[1]["payload"] == 0xAD
        # dst has no first/last, so they shouldn't be in the result
        assert "first" not in results[0]
        assert "last" not in results[0]

    def test_dst_has_first_last_src_does_not(self):
        """dst has first/last but src doesn't — should connect without error."""
        src_sig = Signature(unsigned(8))
        dst_sig = Signature(unsigned(8), has_first_last=True)
        dut = _ConnectBridge(src_sig, dst_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0xBE)
            await sender.send(ctx, 0xEF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_mismatch2.vcd")

        assert results[0]["payload"] == 0xBE
        assert results[1]["payload"] == 0xEF
        # first/last should be 0 (unconnected defaults)
        assert results[0]["first"] == 0
        assert results[0]["last"] == 0

    def test_src_has_param_dst_has_keep(self):
        """src has param, dst has keep — neither should be connected."""
        src_sig = Signature(unsigned(8), param_shape=unsigned(4))
        dst_sig = Signature(unsigned(8), has_keep=True)
        dut = _ConnectBridge(src_sig, dst_sig)
        results = []

        async def sender_tb(ctx):
            sender = StreamSimSender(dut.i_stream)
            await sender.send(ctx, 0x42, param=0xF)

        async def receiver_tb(ctx):
            receiver = StreamSimReceiver(dut.o_stream)
            results.append(await receiver.recv(ctx))

        _run_sim(dut, sender_tb, receiver_tb, vcd_name="test_cs_mismatch3.vcd")

        assert results[0]["payload"] == 0x42
        # keep should be 0 (unconnected)
        assert results[0]["keep"] == 0
