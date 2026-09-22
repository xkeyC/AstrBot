"""Minimal protobuf codec for the Mumble protocol.

Messages are plain dicts keyed by field name. A schema maps field numbers to
``Field`` descriptions; only the scalar kinds Mumble uses are supported.
Unknown fields are skipped on decode, absent fields are omitted on encode.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

VARINT = 0
FIXED64 = 1
LENGTH = 2
FIXED32 = 5

# kind -> wire type
_WIRE = {
    "uint32": VARINT,
    "uint64": VARINT,
    "int32": VARINT,
    "bool": VARINT,
    "enum": VARINT,
    "string": LENGTH,
    "bytes": LENGTH,
    "message": LENGTH,
    "float": FIXED32,
}


@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    repeated: bool = False
    schema: Schema | None = None  # for kind == "message"


Schema = dict[int, Field]


class DecodeError(ValueError):
    pass


def encode_varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_varint(data: bytes | memoryview, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise DecodeError("truncated varint")
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7
        if shift >= 70:
            raise DecodeError("varint too long")


def _encode_value(field: Field, value: Any) -> bytes:
    kind = field.kind
    if kind in ("uint32", "uint64", "int32", "enum"):
        return encode_varint(int(value))
    if kind == "bool":
        return b"\x01" if value else b"\x00"
    if kind == "float":
        return struct.pack("<f", value)
    if kind == "string":
        raw = value.encode("utf-8")
    elif kind == "bytes":
        raw = bytes(value)
    elif kind == "message":
        assert field.schema is not None
        raw = encode(field.schema, value)
    else:
        raise ValueError(f"unsupported kind {kind}")
    return encode_varint(len(raw)) + raw


def encode(schema: Schema, message: dict[str, Any]) -> bytes:
    out = bytearray()
    for number, field in schema.items():
        value = message.get(field.name)
        if value is None:
            continue
        tag = encode_varint(number << 3 | _WIRE[field.kind])
        for item in value if field.repeated else (value,):
            out += tag + _encode_value(field, item)
    return bytes(out)


def _decode_scalar(field: Field, raw: int) -> Any:
    if field.kind == "bool":
        return bool(raw)
    if field.kind == "int32":
        raw &= 0xFFFFFFFF
        return raw - (1 << 32) if raw & 0x80000000 else raw
    if field.kind == "uint32":
        return raw & 0xFFFFFFFF
    return raw


def decode(schema: Schema, data: bytes | memoryview) -> dict[str, Any]:
    data = memoryview(data)
    message: dict[str, Any] = {}
    pos = 0
    while pos < len(data):
        key, pos = decode_varint(data, pos)
        number, wire = key >> 3, key & 7
        field = schema.get(number)
        if wire == VARINT:
            raw, pos = decode_varint(data, pos)
            values = [raw]
        elif wire == FIXED64:
            if pos + 8 > len(data):
                raise DecodeError("truncated fixed64")
            values = [bytes(data[pos : pos + 8])]
            pos += 8
        elif wire == FIXED32:
            if pos + 4 > len(data):
                raise DecodeError("truncated fixed32")
            values = [bytes(data[pos : pos + 4])]
            pos += 4
        elif wire == LENGTH:
            length, pos = decode_varint(data, pos)
            if pos + length > len(data):
                raise DecodeError("truncated field")
            chunk = data[pos : pos + length]
            pos += length
            values = [chunk]
        else:
            raise DecodeError(f"unsupported wire type {wire}")
        if field is None:
            continue
        expected = _WIRE[field.kind]
        if wire == LENGTH and expected in (VARINT, FIXED32):
            # packed repeated scalars
            values = _unpack(field, values[0])
        elif wire != expected:
            raise DecodeError(f"field {field.name}: wire type {wire}")
        decoded = [_decode_value(field, value) for value in values]
        if field.repeated:
            message.setdefault(field.name, []).extend(decoded)
        elif decoded:
            message[field.name] = decoded[-1]
    return message


def _unpack(field: Field, chunk: memoryview) -> list[Any]:
    values: list[Any] = []
    pos = 0
    while pos < len(chunk):
        if _WIRE[field.kind] == FIXED32:
            if pos + 4 > len(chunk):
                raise DecodeError("truncated packed fixed32")
            values.append(bytes(chunk[pos : pos + 4]))
            pos += 4
        else:
            raw, pos = decode_varint(chunk, pos)
            values.append(raw)
    return values


def _decode_value(field: Field, value: Any) -> Any:
    kind = field.kind
    if kind == "float":
        return struct.unpack("<f", value)[0]
    if kind == "string":
        return bytes(value).decode("utf-8", errors="replace")
    if kind == "bytes":
        return bytes(value)
    if kind == "message":
        assert field.schema is not None
        return decode(field.schema, value)
    return _decode_scalar(field, value)
