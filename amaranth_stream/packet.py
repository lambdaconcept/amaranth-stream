"""Packet processing components for amaranth-stream.

Provides :class:`HeaderLayout`, :class:`Packetizer`, :class:`Depacketizer`,
:class:`PacketFIFO`, :class:`PacketStatus`, :class:`Stitcher`,
:class:`LastInserter`, and :class:`LastOnTimeout` for packet-level stream
processing.
"""

import math

from amaranth import *
from amaranth.hdl import Shape, Signal, unsigned
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out
from amaranth.lib.data import StructLayout
from amaranth.lib.fifo import SyncFIFO, SyncFIFOBuffered

from ._base import Signature as StreamSignature

__all__ = [
    "HeaderLayout",
    "Packetizer",
    "Depacketizer",
    "PacketFIFO",
    "PacketStatus",
    "Stitcher",
    "LastInserter",
    "LastOnTimeout",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_forward_signals(sig):
    """Return the list of forward signal names for a stream signature."""
    names = ["payload"]
    if sig.has_first_last:
        names.extend(["first", "last"])
    if sig.param_shape is not None:
        names.append("param")
    if sig.has_keep:
        names.append("keep")
    return names


def _stream_data_width(sig):
    """Calculate the total data width for packing all stream fields into a FIFO word."""
    fields = []
    payload_width = Shape.cast(sig.payload_shape).width
    fields.append(("payload", payload_width))

    if sig.has_first_last:
        fields.append(("first", 1))
        fields.append(("last", 1))

    if sig.param_shape is not None:
        param_width = Shape.cast(sig.param_shape).width
        fields.append(("param", param_width))

    if sig.has_keep:
        keep_width = max(1, math.ceil(payload_width / 8))
        fields.append(("keep", keep_width))

    total_width = sum(w for _, w in fields)
    return total_width, fields


# ===========================================================================
# HeaderLayout
# ===========================================================================

class HeaderLayout:
    """Declarative header definition.

    Parameters
    ----------
    fields : :class:`dict`
        Mapping of field name → ``(width, byte_offset)`` or
        ``(width, byte_offset, bit_offset)``.  When *bit_offset* is
        omitted it defaults to 0.

    Properties
    ----------
    struct_layout : :class:`StructLayout`
        StructLayout for the header.
    byte_length : :class:`int`
        Total header bytes.
    fields : :class:`dict`
        The normalised fields dict (values are always 3-tuples
        ``(width, byte_offset, bit_offset)``).
    """

    def __init__(self, fields):
        # Normalise to 3-tuples (width, byte_offset, bit_offset)
        normalised = {}
        for name, value in dict(fields).items():
            if len(value) == 2:
                width, byte_offset = value
                bit_offset = 0
            elif len(value) == 3:
                width, byte_offset, bit_offset = value
            else:
                raise ValueError(
                    f"Field '{name}': expected (width, byte_offset) or "
                    f"(width, byte_offset, bit_offset), got {len(value)}-tuple")
            normalised[name] = (width, byte_offset, bit_offset)
        self._fields = normalised

        # Validate that fields don't overlap within the same byte region
        self._validate_no_overlap()

        # Calculate byte_length from max(byte_offset + ceil((bit_offset + width) / 8))
        max_end = 0
        for name, (width, byte_offset, bit_offset) in self._fields.items():
            end = byte_offset + math.ceil((bit_offset + width) / 8)
            if end > max_end:
                max_end = end
        self._byte_length = max_end

    def _validate_no_overlap(self):
        """Check that no two fields occupy the same bits."""
        # Build a list of (abs_start_bit, abs_end_bit, name) and check for overlaps
        intervals = []
        for name, (width, byte_offset, bit_offset) in self._fields.items():
            start = byte_offset * 8 + bit_offset
            end = start + width
            intervals.append((start, end, name))

        # Sort by start bit and check for overlaps
        intervals.sort()
        for i in range(len(intervals) - 1):
            _, end_a, name_a = intervals[i]
            start_b, _, name_b = intervals[i + 1]
            if end_a > start_b:
                raise ValueError(
                    f"Fields '{name_a}' and '{name_b}' overlap: "
                    f"'{name_a}' ends at bit {end_a}, "
                    f"'{name_b}' starts at bit {start_b}")

    @property
    def struct_layout(self):
        """Return a StructLayout for the header fields."""
        return StructLayout({name: width for name, (width, _, _) in self._fields.items()})

    @property
    def byte_length(self):
        """Total header size in bytes."""
        return self._byte_length

    @property
    def fields(self):
        """The normalised fields dict (3-tuples)."""
        return self._fields

    def abs_bit_offset(self, field_name):
        """Return the absolute bit position for *field_name*.

        Computed as ``byte_offset * 8 + bit_offset``.
        """
        width, byte_offset, bit_offset = self._fields[field_name]
        return byte_offset * 8 + bit_offset


# ===========================================================================
# Packetizer
# ===========================================================================

class Packetizer(wiring.Component):
    """Inserts header beats before payload. Latches header on first.

    Parameters
    ----------
    header_layout : :class:`HeaderLayout`
        Header definition.
    payload_signature : :class:`~amaranth_stream.Signature`
        Stream signature (must have has_first_last=True).
    packed : :class:`bool`
        If ``True``, the last header beat is filled with the start of
        payload data when the header does not perfectly align to the bus
        width.  Subsequent payload beats are shifted accordingly.
        Default ``False`` preserves the original behaviour.

    Ports
    -----
    i_stream : In(payload_signature)
        Payload data input.
    o_stream : Out(payload_signature)
        Header + payload data output.
    header : In(header_layout.struct_layout)
        Header fields to insert.
    """

    def __init__(self, header_layout, payload_signature, *, packed=False):
        if not isinstance(payload_signature, StreamSignature):
            raise TypeError(
                f"Expected amaranth_stream.Signature, got {type(payload_signature).__name__}")
        if not payload_signature.has_first_last:
            raise ValueError("payload_signature must have has_first_last=True")
        self._header_layout = header_layout
        self._payload_sig = payload_signature
        self._payload_width = Shape.cast(payload_signature.payload_shape).width
        self._packed = packed

        # Calculate number of header beats
        header_bits = header_layout.byte_length * 8
        self._header_beats = math.ceil(header_bits / self._payload_width)
        self._header_bits = header_bits

        super().__init__({
            "i_stream": In(payload_signature),
            "o_stream": Out(payload_signature),
            "header": In(header_layout.struct_layout),
        })

    def elaborate(self, platform):
        m = Module()

        payload_width = self._payload_width
        header_beats = self._header_beats
        header_layout = self._header_layout
        header_bits = self._header_bits

        # Latch the header value
        header_latched = Signal(header_layout.byte_length * 8, name="header_latched")

        # Beat counter for header emission
        beat_cnt = Signal(range(max(header_beats, 1)), name="hdr_beat_cnt")

        if not self._packed or (header_bits % payload_width == 0):
            # ---- Original (non-packed) behaviour ----
            with m.FSM(name="packetizer"):
                with m.State("HEADER"):
                    # Latch header on entry (first beat)
                    with m.If(beat_cnt == 0):
                        # Pack header fields into a flat bit vector
                        offset = 0
                        for name, (width, byte_off, bit_off) in header_layout.fields.items():
                            abs_bit = byte_off * 8 + bit_off
                            m.d.sync += header_latched[abs_bit:abs_bit + width].eq(
                                getattr(self.header, name))
                            offset += width

                    # Emit header beats from latched value
                    # For the first beat, use the header port directly (latched not ready yet)
                    with m.If(beat_cnt == 0):
                        # Build header bits directly from input
                        header_flat = Signal(header_bits, name="header_flat")
                        offset = 0
                        for name, (width, byte_off, bit_off) in header_layout.fields.items():
                            abs_bit = byte_off * 8 + bit_off
                            m.d.comb += header_flat[abs_bit:abs_bit + width].eq(
                                getattr(self.header, name))
                        m.d.comb += self.o_stream.payload.eq(
                            header_flat[0:payload_width])
                    with m.Else():
                        # Subsequent beats from latched value
                        m.d.comb += self.o_stream.payload.eq(
                            header_latched.word_select(beat_cnt, payload_width))

                    m.d.comb += [
                        self.o_stream.valid.eq(1),
                        self.o_stream.first.eq(beat_cnt == 0),
                        self.o_stream.last.eq(0),
                        self.i_stream.ready.eq(0),  # Don't consume payload during header
                    ]

                    # Advance on transfer
                    with m.If(self.o_stream.valid & self.o_stream.ready):
                        with m.If(beat_cnt == header_beats - 1):
                            m.d.sync += beat_cnt.eq(0)
                            m.next = "PAYLOAD"
                        with m.Else():
                            m.d.sync += beat_cnt.eq(beat_cnt + 1)

                with m.State("PAYLOAD"):
                    # Pass through payload beats
                    m.d.comb += [
                        self.o_stream.payload.eq(self.i_stream.payload),
                        self.o_stream.valid.eq(self.i_stream.valid),
                        self.o_stream.first.eq(0),
                        self.o_stream.last.eq(self.i_stream.last),
                        self.i_stream.ready.eq(self.o_stream.ready),
                    ]

                    # On last beat, go back to HEADER
                    with m.If(self.i_stream.valid & self.i_stream.ready & self.i_stream.last):
                        m.next = "HEADER"
        else:
            # ---- Packed mode: merge tail of header with start of payload ----
            remainder = header_bits % payload_width   # header bits in last header beat
            pack_bits = payload_width - remainder      # payload bits packed into last header beat

            # Carry register: holds the upper bits of the previous payload beat
            # that couldn't fit into the current output beat.
            carry = Signal(remainder, name="carry")

            # Number of full header beats (those that are 100% header)
            full_hdr_beats = header_beats - 1  # last header beat is the packed one

            # Build a flat header signal from the input ports (for beat 0)
            header_flat = Signal(header_bits, name="header_flat_p")
            for name, (width, byte_off, bit_off) in header_layout.fields.items():
                abs_bit = byte_off * 8 + bit_off
                m.d.comb += header_flat[abs_bit:abs_bit + width].eq(
                    getattr(self.header, name))

            init_state = "HEADER" if full_hdr_beats > 0 else "PACK"

            with m.FSM(name="packetizer", init=init_state):
                if full_hdr_beats > 0:
                    with m.State("HEADER"):
                        # Latch header on entry (first beat)
                        with m.If(beat_cnt == 0):
                            for name, (width, byte_off, bit_off) in header_layout.fields.items():
                                abs_bit = byte_off * 8 + bit_off
                                m.d.sync += header_latched[abs_bit:abs_bit + width].eq(
                                    getattr(self.header, name))

                        # Emit full header beats
                        with m.If(beat_cnt == 0):
                            m.d.comb += self.o_stream.payload.eq(
                                header_flat[0:payload_width])
                        with m.Else():
                            m.d.comb += self.o_stream.payload.eq(
                                header_latched.word_select(beat_cnt, payload_width))

                        m.d.comb += [
                            self.o_stream.valid.eq(1),
                            self.o_stream.first.eq(beat_cnt == 0),
                            self.o_stream.last.eq(0),
                            self.i_stream.ready.eq(0),
                        ]

                        with m.If(self.o_stream.valid & self.o_stream.ready):
                            with m.If(beat_cnt == full_hdr_beats - 1):
                                m.d.sync += beat_cnt.eq(0)
                                m.next = "PACK"
                            with m.Else():
                                m.d.sync += beat_cnt.eq(beat_cnt + 1)

                with m.State("PACK"):
                    # Last header beat: lower `remainder` bits = header tail,
                    # upper `pack_bits` bits = start of payload.
                    # We need to consume from i_stream here.
                    hdr_tail = Signal(remainder, name="hdr_tail")
                    if full_hdr_beats > 0:
                        m.d.comb += hdr_tail.eq(
                            header_latched[full_hdr_beats * payload_width:
                                           full_hdr_beats * payload_width + remainder])
                    else:
                        # No full header beats — use header_flat directly
                        m.d.comb += hdr_tail.eq(header_flat[:remainder])
                        # Also latch header for potential future use
                        for name, (width, byte_off, bit_off) in header_layout.fields.items():
                            abs_bit = byte_off * 8 + bit_off
                            m.d.sync += header_latched[abs_bit:abs_bit + width].eq(
                                getattr(self.header, name))

                    m.d.comb += [
                        self.o_stream.payload[:remainder].eq(hdr_tail),
                        self.o_stream.payload[remainder:].eq(
                            self.i_stream.payload[:pack_bits]),
                        self.o_stream.valid.eq(self.i_stream.valid),
                        self.o_stream.first.eq(1 if full_hdr_beats == 0 else 0),
                        self.o_stream.last.eq(0),  # Never last — carry still needs flushing
                        self.i_stream.ready.eq(self.o_stream.ready),
                    ]

                    with m.If(self.i_stream.valid & self.i_stream.ready):
                        # Store the upper bits of this payload beat as carry
                        m.d.sync += carry.eq(
                            self.i_stream.payload[pack_bits:])
                        with m.If(self.i_stream.last):
                            # Payload ended — flush carry in DRAIN
                            m.next = "DRAIN"
                        with m.Else():
                            m.next = "PAYLOAD"

                with m.State("PAYLOAD"):
                    # Each output beat: lower `remainder` bits from carry,
                    # upper `pack_bits` bits from current input payload.
                    m.d.comb += [
                        self.o_stream.payload[:remainder].eq(carry),
                        self.o_stream.payload[remainder:].eq(
                            self.i_stream.payload[:pack_bits]),
                        self.o_stream.valid.eq(self.i_stream.valid),
                        self.o_stream.first.eq(0),
                        self.o_stream.last.eq(0),  # Never last — carry still needs flushing
                        self.i_stream.ready.eq(self.o_stream.ready),
                    ]

                    with m.If(self.i_stream.valid & self.i_stream.ready):
                        m.d.sync += carry.eq(
                            self.i_stream.payload[pack_bits:])
                        with m.If(self.i_stream.last):
                            # Payload ended — flush carry in DRAIN
                            m.next = "DRAIN"

                with m.State("DRAIN"):
                    # Emit the final partial beat with carry data in the lower
                    # bits.  Upper bits are zero-padded.
                    m.d.comb += [
                        self.o_stream.payload[:remainder].eq(carry),
                        self.o_stream.payload[remainder:].eq(0),
                        self.o_stream.valid.eq(1),
                        self.o_stream.first.eq(0),
                        self.o_stream.last.eq(1),
                    ]
                    m.d.comb += self.i_stream.ready.eq(0)

                    with m.If(self.o_stream.valid & self.o_stream.ready):
                        m.next = init_state

        return m


# ===========================================================================
# Depacketizer
# ===========================================================================

class Depacketizer(wiring.Component):
    """Extracts header from packet start. Strips header beats.

    Parameters
    ----------
    header_layout : :class:`HeaderLayout`
        Header definition.
    payload_signature : :class:`~amaranth_stream.Signature`
        Stream signature.
    packed : :class:`bool`
        If ``True``, the last header beat contains packed payload data
        that must be extracted and realigned.  Must match the ``packed``
        setting used by the corresponding :class:`Packetizer`.
        Default ``False`` preserves the original behaviour.

    Ports
    -----
    i_stream : In(payload_signature)
        Header + payload data input.
    o_stream : Out(payload_signature)
        Payload only output.
    header : Out(header_layout.struct_layout)
        Extracted header fields.
    header_raw : Out(unsigned(header_layout.byte_length * 8))
        Raw header bits in wire order (byte 0 in bits [7:0]). Useful for
        checksum verification over the exact received header image.
    """

    def __init__(self, header_layout, payload_signature, *, packed=False):
        if not isinstance(payload_signature, StreamSignature):
            raise TypeError(
                f"Expected amaranth_stream.Signature, got {type(payload_signature).__name__}")
        if not payload_signature.has_first_last:
            raise ValueError("payload_signature must have has_first_last=True")
        self._header_layout = header_layout
        self._payload_sig = payload_signature
        self._payload_width = Shape.cast(payload_signature.payload_shape).width
        self._packed = packed

        header_bits = header_layout.byte_length * 8
        self._header_beats = math.ceil(header_bits / self._payload_width)
        self._header_bits = header_bits

        super().__init__({
            "i_stream": In(payload_signature),
            "o_stream": Out(payload_signature),
            "header": Out(header_layout.struct_layout),
            "header_raw": Out(unsigned(header_bits)),
        })

    def elaborate(self, platform):
        m = Module()

        payload_width = self._payload_width
        header_beats = self._header_beats
        header_layout = self._header_layout
        header_bits = self._header_bits

        # Storage for header bits
        header_store = Signal(header_layout.byte_length * 8, name="header_store")

        # Beat counter for header consumption
        beat_cnt = Signal(range(max(header_beats, 1)), name="hdr_beat_cnt")

        # Track if this is the first payload beat after header
        first_payload = Signal(1, name="first_payload")

        if not self._packed or (header_bits % payload_width == 0):
            # ---- Original (non-packed) behaviour ----
            with m.FSM(name="depacketizer"):
                with m.State("HEADER"):
                    m.d.comb += [
                        self.i_stream.ready.eq(1),  # Consume header beats
                        self.o_stream.valid.eq(0),   # Don't output during header
                    ]

                    with m.If(self.i_stream.valid & self.i_stream.ready):
                        # Store header beat
                        m.d.sync += header_store.word_select(beat_cnt, payload_width).eq(
                            self.i_stream.payload)

                        with m.If(beat_cnt == header_beats - 1):
                            m.d.sync += [
                                beat_cnt.eq(0),
                                first_payload.eq(1),
                            ]
                            m.next = "PAYLOAD"
                        with m.Else():
                            m.d.sync += beat_cnt.eq(beat_cnt + 1)

                        # Runt packet (ends during, or exactly at the end of,
                        # the header): drop it and resynchronize.
                        with m.If(self.i_stream.last):
                            m.d.sync += beat_cnt.eq(0)
                            m.next = "HEADER"

                with m.State("PAYLOAD"):
                    # Pass through payload beats
                    m.d.comb += [
                        self.o_stream.payload.eq(self.i_stream.payload),
                        self.o_stream.valid.eq(self.i_stream.valid),
                        self.o_stream.first.eq(first_payload),
                        self.o_stream.last.eq(self.i_stream.last),
                        self.i_stream.ready.eq(self.o_stream.ready),
                    ]

                    with m.If(self.i_stream.valid & self.i_stream.ready):
                        m.d.sync += first_payload.eq(0)

                        with m.If(self.i_stream.last):
                            m.next = "HEADER"
        else:
            # ---- Packed mode: extract payload from packed header beat ----
            remainder = header_bits % payload_width   # header bits in last header beat
            pack_bits = payload_width - remainder      # payload bits packed into last header beat

            # Number of full header beats (100% header)
            full_hdr_beats = header_beats - 1

            # Carry register: holds payload bits extracted from the packed beat
            # or the upper bits of the previous payload beat.
            # Width is pack_bits because that's how many payload bits we extract
            # from each input beat.
            carry = Signal(pack_bits, name="carry")

            init_state = "HEADER" if full_hdr_beats > 0 else "UNPACK"

            with m.FSM(name="depacketizer", init=init_state):
                if full_hdr_beats > 0:
                    with m.State("HEADER"):
                        m.d.comb += [
                            self.i_stream.ready.eq(1),
                            self.o_stream.valid.eq(0),
                        ]

                        with m.If(self.i_stream.valid & self.i_stream.ready):
                            # Store full header beats
                            m.d.sync += header_store.word_select(beat_cnt, payload_width).eq(
                                self.i_stream.payload)

                            with m.If(beat_cnt == full_hdr_beats - 1):
                                m.d.sync += beat_cnt.eq(0)
                                m.next = "UNPACK"
                            with m.Else():
                                m.d.sync += beat_cnt.eq(beat_cnt + 1)

                            # Runt packet: drop it and resynchronize.
                            with m.If(self.i_stream.last):
                                m.d.sync += beat_cnt.eq(0)
                                m.next = "HEADER"

                with m.State("UNPACK"):
                    # Packed header beat: lower `remainder` bits = header tail,
                    # upper `pack_bits` bits = start of payload.
                    # Consume this beat, store header tail and payload carry.
                    m.d.comb += [
                        self.i_stream.ready.eq(1),
                        self.o_stream.valid.eq(0),
                    ]

                    with m.If(self.i_stream.valid & self.i_stream.ready):
                        # Store header tail bits
                        m.d.sync += header_store[
                            full_hdr_beats * payload_width:
                            full_hdr_beats * payload_width + remainder
                        ].eq(self.i_stream.payload[:remainder])
                        # Store payload bits as carry (upper pack_bits of this beat)
                        m.d.sync += carry.eq(
                            self.i_stream.payload[remainder:remainder + pack_bits])
                        m.d.sync += first_payload.eq(1)
                        m.next = "PAYLOAD"
                        # Runt packet: ends on the packed beat; drop it and
                        # resynchronize.
                        with m.If(self.i_stream.last):
                            m.d.sync += beat_cnt.eq(0)
                            m.next = init_state

                with m.State("PAYLOAD"):
                    # Reconstruct original payload from shifted data.
                    #
                    # Packetizer output format:
                    #   PACK beat:    [hdr_tail(remainder bits) | orig_payload[0:pack_bits]]
                    #   PAYLOAD beat: [prev_carry(remainder bits) | orig_payload[0:pack_bits]]
                    #   DRAIN beat:   [prev_carry(remainder bits) | zeros(pack_bits)]
                    #
                    # Where prev_carry = previous_orig_payload[pack_bits:payload_width]
                    #
                    # Depacketizer carry (pack_bits wide) holds the lower portion
                    # of the original payload beat.  The upper portion comes from
                    # the lower `remainder` bits of the current input beat.
                    #
                    # output = Cat(carry, input[0:remainder])
                    #        = pack_bits + remainder = payload_width  ✓
                    # new_carry = input[remainder:payload_width]  (pack_bits wide)

                    m.d.comb += [
                        self.o_stream.payload.eq(
                            Cat(carry, self.i_stream.payload[:remainder])),
                        self.o_stream.valid.eq(self.i_stream.valid),
                        self.o_stream.first.eq(first_payload),
                        self.o_stream.last.eq(self.i_stream.last),
                        self.i_stream.ready.eq(self.o_stream.ready),
                    ]

                    with m.If(self.i_stream.valid & self.i_stream.ready):
                        m.d.sync += first_payload.eq(0)
                        # Update carry with the upper pack_bits of this input beat
                        m.d.sync += carry.eq(
                            self.i_stream.payload[remainder:])

                        with m.If(self.i_stream.last):
                            m.next = init_state

        # Extract header fields from stored bits
        for name, (width, byte_off, bit_off) in header_layout.fields.items():
            abs_bit = byte_off * 8 + bit_off
            m.d.comb += getattr(self.header, name).eq(
                header_store[abs_bit:abs_bit + width])

        m.d.comb += self.header_raw.eq(header_store)

        return m


# ===========================================================================
# PacketFIFO
# ===========================================================================

class PacketFIFO(wiring.Component):
    """Atomic packet FIFO. Only releases complete packets.

    The payload storage is read synchronously (one prefetch register in front
    of the output) so that it maps to FPGA block RAM; the packet length FIFO
    is tiny and stays in distributed RAM. A committed packet becomes visible
    on ``o_stream`` two cycles after its last beat was accepted.

    Parameters
    ----------
    signature : :class:`~amaranth_stream.Signature`
        Stream signature (must have has_first_last=True).
    payload_depth : :class:`int`
        Max payload beats.
    packet_depth : :class:`int`
        Max buffered packets (default 16).
    has_abort : :class:`bool`
        If ``True``, add an ``abort`` input signal that discards the
        current in-progress packet and resets the write pointer to the
        start of that packet.  Previously committed packets are not
        affected.  Default ``False``.

    Ports
    -----
    i_stream : In(signature)
        Input stream.
    o_stream : Out(signature)
        Output stream.
    packet_count : Out(range(packet_depth + 1))
        Number of complete packets available.
    abort : In(1)
        Assert mid-packet to discard the current packet (only present
        when ``has_abort=True``).
    """

    def __init__(self, signature, payload_depth, packet_depth=16, *, has_abort=False):
        if not isinstance(signature, StreamSignature):
            raise TypeError(
                f"Expected amaranth_stream.Signature, got {type(signature).__name__}")
        if not signature.has_first_last:
            raise ValueError("signature must have has_first_last=True")
        self._stream_sig = signature
        self._payload_depth = payload_depth
        self._packet_depth = packet_depth
        self._has_abort = has_abort

        ports = {
            "i_stream": In(signature),
            "o_stream": Out(signature),
            "packet_count": Out(range(packet_depth + 1)),
        }
        if has_abort:
            ports["abort"] = In(1)

        super().__init__(ports)

    def elaborate(self, platform):
        m = Module()

        sig = self._stream_sig
        total_width, fields = _stream_data_width(sig)

        if not self._has_abort:
            return self._elaborate_standard(m, sig, total_width, fields)
        else:
            return self._elaborate_with_abort(m, sig, total_width, fields)

    def _elaborate_standard(self, m, sig, total_width, fields):
        """Original PacketFIFO implementation (no abort support)."""
        # Data FIFO: stores packed stream data. Buffered (synchronous memory
        # read + output register) so that the storage maps to block RAM.
        data_fifo = SyncFIFOBuffered(width=total_width, depth=self._payload_depth)
        m.submodules.data_fifo = data_fifo

        # Packet length FIFO: stores beat counts for complete packets
        len_width = max(1, (self._payload_depth).bit_length())
        pkt_fifo = SyncFIFO(width=len_width, depth=self._packet_depth)
        m.submodules.pkt_fifo = pkt_fifo

        # --- Write side ---
        # Pack stream fields into data word
        pack_signals = []
        for name, _width in fields:
            pack_signals.append(getattr(self.i_stream, name))

        # Beat counter for current packet being written
        wr_beat_cnt = Signal(range(self._payload_depth + 1), name="wr_beat_cnt")

        # Write to data FIFO
        w_transfer = Signal(name="w_transfer")
        m.d.comb += [
            data_fifo.w_data.eq(Cat(*pack_signals)),
            w_transfer.eq(self.i_stream.valid & self.i_stream.ready),
            data_fifo.w_en.eq(w_transfer),
        ]

        # Accept input when data FIFO has space
        m.d.comb += self.i_stream.ready.eq(data_fifo.w_rdy)

        # Track beats and commit packets
        with m.If(w_transfer):
            m.d.sync += wr_beat_cnt.eq(wr_beat_cnt + 1)
            with m.If(self.i_stream.last):
                # Commit: write beat count to packet FIFO
                m.d.comb += [
                    pkt_fifo.w_data.eq(wr_beat_cnt + 1),
                    pkt_fifo.w_en.eq(1),
                ]
                m.d.sync += wr_beat_cnt.eq(0)

        # --- Read side ---
        # Read beat counter for current packet being read
        rd_beat_cnt = Signal(range(self._payload_depth + 1), name="rd_beat_cnt")
        rd_pkt_len = Signal(len_width, name="rd_pkt_len")
        reading = Signal(name="reading")

        # Unpack data FIFO output
        offset = 0
        for name, width in fields:
            m.d.comb += getattr(self.o_stream, name).eq(data_fifo.r_data[offset:offset + width])
            offset += width

        # Output valid only when we have a packet to read
        m.d.comb += self.o_stream.valid.eq(reading & data_fifo.r_rdy)

        # Override first/last from the FIFO data with our own framing
        m.d.comb += [
            self.o_stream.first.eq(reading & (rd_beat_cnt == 0)),
            self.o_stream.last.eq(reading & (rd_beat_cnt == rd_pkt_len - 1)),
        ]

        r_transfer = Signal(name="r_transfer")
        m.d.comb += [
            r_transfer.eq(self.o_stream.valid & self.o_stream.ready),
            data_fifo.r_en.eq(r_transfer),
        ]

        with m.If(~reading):
            # Not currently reading a packet — check if one is available
            with m.If(pkt_fifo.r_rdy):
                m.d.sync += [
                    rd_pkt_len.eq(pkt_fifo.r_data),
                    rd_beat_cnt.eq(0),
                    reading.eq(1),
                ]
                m.d.comb += pkt_fifo.r_en.eq(1)
        with m.Else():
            # Currently reading a packet
            with m.If(r_transfer):
                with m.If(rd_beat_cnt == rd_pkt_len - 1):
                    # Finished this packet
                    m.d.sync += reading.eq(0)
                with m.Else():
                    m.d.sync += rd_beat_cnt.eq(rd_beat_cnt + 1)

        # Packet count
        m.d.comb += self.packet_count.eq(pkt_fifo.level)

        return m

    def _elaborate_with_abort(self, m, sig, total_width, fields):
        """PacketFIFO with abort support using manual memory management."""
        from amaranth.lib.memory import Memory

        depth = self._payload_depth
        addr_width = max(1, (depth - 1).bit_length())

        # --- Data memory (replaces SyncFIFO for data) ---
        # Synchronous read port (block-RAM friendly): words are prefetched
        # into the memory's output register one beat ahead of the consumer.
        data_mem = Memory(shape=unsigned(total_width), depth=depth, init=[])
        m.submodules.data_mem = data_mem
        wr_port = data_mem.write_port()
        rd_port = data_mem.read_port(domain="sync")

        # Packet length FIFO: stores beat counts for complete packets
        len_width = max(1, (self._payload_depth).bit_length())
        pkt_fifo = SyncFIFO(width=len_width, depth=self._packet_depth)
        m.submodules.pkt_fifo = pkt_fifo

        # --- Write pointers ---
        # wr_ptr: current write position (advances with each beat)
        # commit_ptr: position at start of current packet (restored on abort)
        # fetch_ptr: next committed word to prefetch into the output register
        wr_ptr = Signal(addr_width, name="wr_ptr")
        commit_ptr = Signal(addr_width, name="commit_ptr")
        fetch_ptr = Signal(addr_width, name="fetch_ptr")

        # Level tracking: number of committed beats in the buffer
        # (committed means the packet has been fully written and committed)
        committed_level = Signal(range(depth + 1), name="committed_level")

        # --- Write side ---
        pack_signals = []
        for name, _width in fields:
            pack_signals.append(getattr(self.i_stream, name))

        # Beat counter for current packet being written
        wr_beat_cnt = Signal(range(self._payload_depth + 1), name="wr_beat_cnt")

        # Calculate available space: depth - (wr_ptr - rd_ptr) mod depth
        # We use the full/empty distinction via a level counter
        wr_level = Signal(range(depth + 1), name="wr_level")
        # wr_level tracks total beats written (committed + uncommitted)
        w_rdy = Signal(name="w_rdy")
        m.d.comb += w_rdy.eq(wr_level < depth)

        w_transfer = Signal(name="w_transfer")
        m.d.comb += w_transfer.eq(self.i_stream.valid & self.i_stream.ready)

        # Write port connections
        m.d.comb += [
            wr_port.addr.eq(wr_ptr),
            wr_port.data.eq(Cat(*pack_signals)),
            wr_port.en.eq(w_transfer),
        ]

        # Accept input when buffer has space
        m.d.comb += self.i_stream.ready.eq(w_rdy)

        # Track beats, commit packets, handle abort
        with m.If(self.abort):
            # Abort: roll back write pointer to commit_ptr
            m.d.sync += [
                wr_ptr.eq(commit_ptr),
                wr_beat_cnt.eq(0),
                wr_level.eq(committed_level),
            ]
        with m.Elif(w_transfer):
            # Advance write pointer
            with m.If(wr_ptr == depth - 1):
                m.d.sync += wr_ptr.eq(0)
            with m.Else():
                m.d.sync += wr_ptr.eq(wr_ptr + 1)

            m.d.sync += wr_beat_cnt.eq(wr_beat_cnt + 1)

            with m.If(self.i_stream.last):
                # Commit: update commit_ptr and write beat count to pkt FIFO
                # commit_ptr = wr_ptr + 1 (next position after last beat)
                with m.If(wr_ptr == depth - 1):
                    m.d.sync += commit_ptr.eq(0)
                with m.Else():
                    m.d.sync += commit_ptr.eq(wr_ptr + 1)

                m.d.comb += [
                    pkt_fifo.w_data.eq(wr_beat_cnt + 1),
                    pkt_fifo.w_en.eq(1),
                ]
                m.d.sync += [
                    wr_beat_cnt.eq(0),
                    committed_level.eq(committed_level + wr_beat_cnt + 1),
                ]

        # Update wr_level: tracks total used slots (committed + uncommitted)
        # On write (non-abort): wr_level += 1
        # On read: wr_level -= 1 (and committed_level -= 1)
        # On abort: wr_level is reset to committed_level (handled above)
        r_transfer = Signal(name="r_transfer")

        with m.If(~self.abort):
            with m.If(w_transfer & ~r_transfer):
                m.d.sync += wr_level.eq(wr_level + 1)
            with m.Elif(~w_transfer & r_transfer):
                m.d.sync += wr_level.eq(wr_level - 1)
            # If both, wr_level stays the same

        # Also update committed_level on read
        with m.If(r_transfer):
            # If we're also committing in the same cycle, net effect is
            # committed_level += (wr_beat_cnt + 1) - 1 = wr_beat_cnt
            # But the commit case above already sets committed_level,
            # so we handle the read decrement separately.
            # We need to be careful about simultaneous commit + read.
            with m.If(w_transfer & self.i_stream.last):
                # Simultaneous commit and read: committed_level += (pkt_len - 1)
                m.d.sync += committed_level.eq(
                    committed_level + wr_beat_cnt + 1 - 1)
            with m.Else():
                m.d.sync += committed_level.eq(committed_level - 1)

        # --- Read side ---
        rd_beat_cnt = Signal(range(self._payload_depth + 1), name="rd_beat_cnt")
        rd_pkt_len = Signal(len_width, name="rd_pkt_len")
        reading = Signal(name="reading")

        # Prefetch: the memory output register holds the next word to be
        # consumed (`out_valid`). Only committed words are ever fetched
        # (`avail` counts committed words not yet fetched), so an abort can
        # never invalidate the register.
        avail = Signal(range(depth + 1), name="avail")
        out_valid = Signal(name="out_valid")
        do_fetch = Signal(name="do_fetch")
        commit = Signal(name="commit")
        m.d.comb += [
            commit.eq(~self.abort & w_transfer & self.i_stream.last),
            do_fetch.eq((avail != 0) & (~out_valid | r_transfer)),
            rd_port.addr.eq(fetch_ptr),
            rd_port.en.eq(do_fetch),
        ]
        m.d.sync += avail.eq(avail + Mux(commit, wr_beat_cnt + 1, 0) - do_fetch)
        with m.If(do_fetch):
            m.d.sync += out_valid.eq(1)
            with m.If(fetch_ptr == depth - 1):
                m.d.sync += fetch_ptr.eq(0)
            with m.Else():
                m.d.sync += fetch_ptr.eq(fetch_ptr + 1)
        with m.Elif(r_transfer):
            m.d.sync += out_valid.eq(0)

        # Unpack data from the memory output register
        offset = 0
        for name, width in fields:
            m.d.comb += getattr(self.o_stream, name).eq(rd_port.data[offset:offset + width])
            offset += width

        # Output valid only when we have a packet to read and its next word
        # has been fetched
        m.d.comb += self.o_stream.valid.eq(reading & out_valid)

        # Override first/last with our own framing
        m.d.comb += [
            self.o_stream.first.eq(reading & (rd_beat_cnt == 0)),
            self.o_stream.last.eq(reading & (rd_beat_cnt == rd_pkt_len - 1)),
        ]

        m.d.comb += r_transfer.eq(self.o_stream.valid & self.o_stream.ready)

        with m.If(~reading):
            # Not currently reading — check if a packet is available
            with m.If(pkt_fifo.r_rdy):
                m.d.sync += [
                    rd_pkt_len.eq(pkt_fifo.r_data),
                    rd_beat_cnt.eq(0),
                    reading.eq(1),
                ]
                m.d.comb += pkt_fifo.r_en.eq(1)
        with m.Else():
            # Currently reading a packet
            with m.If(r_transfer):
                with m.If(rd_beat_cnt == rd_pkt_len - 1):
                    # Finished this packet
                    m.d.sync += reading.eq(0)
                with m.Else():
                    m.d.sync += rd_beat_cnt.eq(rd_beat_cnt + 1)

        # Packet count
        m.d.comb += self.packet_count.eq(pkt_fifo.level)

        return m


# ===========================================================================
# PacketStatus
# ===========================================================================

class PacketStatus(wiring.Component):
    """Packet boundary tracker. Tap-only (non-consuming).

    Parameters
    ----------
    signature : :class:`~amaranth_stream.Signature`
        Stream signature.

    Ports
    -----
    stream : In(signature)
        Tap-only stream (does NOT drive ready).
    in_packet : Out(1)
        High when inside a packet.
    """

    def __init__(self, signature):
        if not isinstance(signature, StreamSignature):
            raise TypeError(
                f"Expected amaranth_stream.Signature, got {type(signature).__name__}")
        if not signature.has_first_last:
            raise ValueError("signature must have has_first_last=True")
        self._stream_sig = signature
        super().__init__({
            "stream": In(signature),
            "in_packet": Out(1),
        })

    def elaborate(self, platform):
        m = Module()

        transfer = Signal(name="transfer")
        m.d.comb += transfer.eq(self.stream.valid & self.stream.ready)

        in_pkt = Signal(name="in_pkt")

        with m.If(transfer & self.stream.first):
            m.d.sync += in_pkt.eq(1)
        with m.If(transfer & self.stream.last):
            m.d.sync += in_pkt.eq(0)

        m.d.comb += self.in_packet.eq(in_pkt)

        # NOTE: We do NOT drive self.stream.ready — this is a passive tap.

        return m


# ===========================================================================
# Stitcher
# ===========================================================================

class Stitcher(wiring.Component):
    """Groups N consecutive packets into one by suppressing intermediate first/last.

    Parameters
    ----------
    signature : :class:`~amaranth_stream.Signature`
        Stream signature (must have has_first_last=True).
    n : :class:`int`
        Number of packets to stitch.

    Ports
    -----
    i_stream : In(signature)
        Input stream.
    o_stream : Out(signature)
        Output stream with stitched packets.
    """

    def __init__(self, signature, n):
        if not isinstance(signature, StreamSignature):
            raise TypeError(
                f"Expected amaranth_stream.Signature, got {type(signature).__name__}")
        if not signature.has_first_last:
            raise ValueError("signature must have has_first_last=True")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self._stream_sig = signature
        self._n = n
        super().__init__({
            "i_stream": In(signature),
            "o_stream": Out(signature),
        })

    def elaborate(self, platform):
        m = Module()

        sig = self._stream_sig
        n = self._n

        # Packet counter (0 to n-1)
        pkt_cnt = Signal(range(n), name="pkt_cnt")

        # Pass through payload and handshake
        m.d.comb += [
            self.o_stream.payload.eq(self.i_stream.payload),
            self.o_stream.valid.eq(self.i_stream.valid),
            self.i_stream.ready.eq(self.o_stream.ready),
        ]

        # Pass through optional signals
        if sig.param_shape is not None:
            m.d.comb += self.o_stream.param.eq(self.i_stream.param)
        if sig.has_keep:
            m.d.comb += self.o_stream.keep.eq(self.i_stream.keep)

        # First: only on the first beat of the first packet
        m.d.comb += self.o_stream.first.eq(
            self.i_stream.first & (pkt_cnt == 0))

        # Last: only on the last beat of the Nth packet
        m.d.comb += self.o_stream.last.eq(
            self.i_stream.last & (pkt_cnt == n - 1))

        # Count packets on last beat transfer
        transfer = Signal(name="transfer")
        m.d.comb += transfer.eq(self.i_stream.valid & self.i_stream.ready)

        with m.If(transfer & self.i_stream.last):
            with m.If(pkt_cnt == n - 1):
                m.d.sync += pkt_cnt.eq(0)
            with m.Else():
                m.d.sync += pkt_cnt.eq(pkt_cnt + 1)

        return m


# ===========================================================================
# LastInserter
# ===========================================================================

class LastInserter(wiring.Component):
    """Injects last=1 every N beats (fixed-size packet creation).

    Parameters
    ----------
    signature : :class:`~amaranth_stream.Signature`
        Stream signature (must have has_first_last=True).
    n : :class:`int`
        Packet length in beats.

    Ports
    -----
    i_stream : In(signature)
        Input stream.
    o_stream : Out(signature)
        Output stream with injected first/last.
    """

    def __init__(self, signature, n):
        if not isinstance(signature, StreamSignature):
            raise TypeError(
                f"Expected amaranth_stream.Signature, got {type(signature).__name__}")
        if not signature.has_first_last:
            raise ValueError("signature must have has_first_last=True")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self._stream_sig = signature
        self._n = n
        super().__init__({
            "i_stream": In(signature),
            "o_stream": Out(signature),
        })

    def elaborate(self, platform):
        m = Module()

        sig = self._stream_sig
        n = self._n

        # Beat counter (0 to n-1)
        beat_cnt = Signal(range(n), name="beat_cnt")

        # Pass through payload and handshake
        m.d.comb += [
            self.o_stream.payload.eq(self.i_stream.payload),
            self.o_stream.valid.eq(self.i_stream.valid),
            self.i_stream.ready.eq(self.o_stream.ready),
        ]

        # Pass through optional signals
        if sig.param_shape is not None:
            m.d.comb += self.o_stream.param.eq(self.i_stream.param)
        if sig.has_keep:
            m.d.comb += self.o_stream.keep.eq(self.i_stream.keep)

        # first on beat 0, last on beat n-1
        m.d.comb += [
            self.o_stream.first.eq(beat_cnt == 0),
            self.o_stream.last.eq(beat_cnt == n - 1),
        ]

        # Count beats on transfer
        transfer = Signal(name="transfer")
        m.d.comb += transfer.eq(self.i_stream.valid & self.i_stream.ready)

        with m.If(transfer):
            with m.If(beat_cnt == n - 1):
                m.d.sync += beat_cnt.eq(0)
            with m.Else():
                m.d.sync += beat_cnt.eq(beat_cnt + 1)

        return m


# ===========================================================================
# LastOnTimeout
# ===========================================================================

class LastOnTimeout(wiring.Component):
    """Injects last=1 after configurable idle timeout.

    Parameters
    ----------
    signature : :class:`~amaranth_stream.Signature`
        Stream signature (must have has_first_last=True).
    timeout : :class:`int`
        Idle cycles before injecting last.

    Ports
    -----
    i_stream : In(signature)
        Input stream.
    o_stream : Out(signature)
        Output stream with timeout-injected last.
    """

    def __init__(self, signature, timeout):
        if not isinstance(signature, StreamSignature):
            raise TypeError(
                f"Expected amaranth_stream.Signature, got {type(signature).__name__}")
        if not signature.has_first_last:
            raise ValueError("signature must have has_first_last=True")
        if timeout < 1:
            raise ValueError(f"timeout must be >= 1, got {timeout}")
        self._stream_sig = signature
        self._timeout = timeout
        super().__init__({
            "i_stream": In(signature),
            "o_stream": Out(signature),
        })

    def elaborate(self, platform):
        m = Module()

        sig = self._stream_sig
        timeout = self._timeout

        # Idle counter
        idle_cnt = Signal(range(timeout + 1), name="idle_cnt")
        # Whether we're inside a packet (seen first, not yet seen last)
        in_packet = Signal(name="in_packet")
        # Force last on next transfer
        force_last = Signal(name="force_last")

        # Pass through payload and handshake
        m.d.comb += [
            self.o_stream.payload.eq(self.i_stream.payload),
            self.o_stream.valid.eq(self.i_stream.valid),
            self.i_stream.ready.eq(self.o_stream.ready),
        ]

        # Pass through optional signals
        if sig.param_shape is not None:
            m.d.comb += self.o_stream.param.eq(self.i_stream.param)
        if sig.has_keep:
            m.d.comb += self.o_stream.keep.eq(self.i_stream.keep)

        # first passes through
        m.d.comb += self.o_stream.first.eq(self.i_stream.first)

        # last: pass through OR force
        m.d.comb += self.o_stream.last.eq(self.i_stream.last | force_last)

        transfer = Signal(name="transfer")
        m.d.comb += transfer.eq(self.i_stream.valid & self.i_stream.ready)

        # Track packet state and idle counter
        with m.If(transfer):
            m.d.sync += idle_cnt.eq(0)
            with m.If(self.i_stream.first):
                m.d.sync += in_packet.eq(1)
            with m.If(self.i_stream.last | force_last):
                m.d.sync += [
                    in_packet.eq(0),
                    force_last.eq(0),
                ]
        with m.Elif(in_packet):
            # No transfer this cycle, increment idle counter
            with m.If(idle_cnt < timeout):
                m.d.sync += idle_cnt.eq(idle_cnt + 1)
            with m.If(idle_cnt == timeout - 1):
                m.d.sync += force_last.eq(1)

        return m
