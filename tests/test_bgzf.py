"""Bgzipped pileups are split into ranges of whole contigs that reproduce the file exactly."""
import random
import zlib

from epirefine.sites import _blocks, is_bgzf, split_pileup


def _bgzf(data: bytes, block: int) -> bytes:
    out = b""
    for i in list(range(0, len(data), block)) + [None]:
        chunk = b"" if i is None else data[i:i + block]
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        body = c.compress(chunk) + c.flush()
        size = len(body) + 26
        out += (b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00" + (size - 1).to_bytes(2, "little") + body
                + zlib.crc32(chunk).to_bytes(4, "little") + len(chunk).to_bytes(4, "little"))
    return out


def test_bgzf_ranges(tmp_path):
    rng = random.Random(1)
    rows = []
    for c in range(60):  # contigs of 1 to 400 rows, so blocks start inside rows, at row starts and inside contigs
        for p in range(rng.choice([1, 1, 2, 30, 400])):
            rows.append(f"contig_{c}\t{p}\t{p + 1}\ta\t{rng.randrange(1000)}\t+\t{p}\t{p + 1}\t0,0,0\t20\t5.0\t1\n")
    data = "".join(rows).encode()
    for block in (97, 1000, 4096, 65280):
        f = tmp_path / f"p{block}.bed.gz"
        f.write_bytes(_bgzf(data, block))
        assert is_bgzf(f)
        for n_parts in (2, 7, 64):
            ranges = split_pileup(f, n_parts, min_size=0)
            chunks = [b"".join(_blocks(str(f), a, b, 1 << 12)) for a, b in ranges]
            assert b"".join(chunks) == data
            firsts = [c.split(b"\t", 1)[0] for c in chunks]
            assert len(set(firsts)) == len(firsts)  # every range starts a new contig
            if block <= 4096 and n_parts > 2:
                assert len(ranges) > 1


def test_plain_gzip_is_one_range(tmp_path):
    import gzip
    f = tmp_path / "p.bed.gz"
    with gzip.open(f, "wb") as fh:
        fh.write(b"c1\t0\t1\ta\t0\t+\n" * 100)
    assert not is_bgzf(f)
    assert split_pileup(f, 8, min_size=0) == [(0, -1)]
    assert b"".join(_blocks(str(f), 0, -1, 64)) == b"c1\t0\t1\ta\t0\t+\n" * 100


def _rows(n_contigs=40, seed=2):
    rng = random.Random(seed)
    return "".join(f"contig_{c}\t{p}\t{p + 1}\ta\t{rng.randrange(1000)}\t+\t{p}\t{p + 1}\t0,0,0\t20\t5.0\t1\n"
                   for c in range(n_contigs) for p in range(rng.choice([1, 2, 30, 400]))).encode()


def test_bgzf_chunks_are_those_of_a_gzip_stream(tmp_path):
    """Per-block inflate yields the chunks that reading `block` bytes at a time from gzip gave (they decide the parts)."""
    import gzip
    data = _rows()
    for bgzf_block in (97, 4096, 65280):
        f = tmp_path / f"p{bgzf_block}.bed.gz"
        f.write_bytes(_bgzf(data, bgzf_block))
        size = f.stat().st_size
        for block in (64, 1000, 1 << 12, 1 << 20):
            want, tail = [], b""
            with gzip.open(f, "rb") as gz:
                while True:
                    buf = gz.read(block)
                    if not buf:
                        break
                    buf = tail + buf
                    cut = buf.rfind(b"\n") + 1
                    tail = buf[cut:]
                    if cut:
                        want.append(buf[:cut])
            if tail:
                want.append(tail)
            assert list(_blocks(str(f), 0, size << 16, block)) == want


def test_damaged_bgzf_fails_loudly(tmp_path):
    import pytest
    data = _rows()
    good = _bgzf(data, 4096)
    f = tmp_path / "p.bed.gz"

    def read(raw, end=None):
        f.write_bytes(raw)
        return b"".join(_blocks(str(f), 0, (len(raw) if end is None else end) << 16, 1 << 12))
    assert read(good) == data
    first = int.from_bytes(good[16:18], "little") + 1          # size of the first block
    flipped = bytearray(good); flipped[first + 40] ^= 0x55      # a byte inside the second block's deflate data
    wrong_crc = bytearray(good); wrong_crc[first - 8] ^= 0x01   # the first block's CRC-32
    wrong_len = bytearray(good); wrong_len[first - 4] ^= 0x01   # the first block's length field
    no_header = bytearray(good); no_header[first] = 0x00        # the second block's magic
    for raw, end in ((bytes(flipped), None), (bytes(wrong_crc), None), (bytes(wrong_len), None), (bytes(no_header), None),
                     (good[:len(good) - 40], len(good)),        # file cut short, range still expects the rest
                     (good[:first + 30], first + 30)):          # cut inside a block
        with pytest.raises(ValueError, match="BGZF block"):
            read(raw, end)
