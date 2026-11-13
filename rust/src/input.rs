//! Text of a pileup range, in chunks: plain file, BGZF (virtual offsets) or a gzip stream.
//!
//! Follows EpiRefine's `sites._blocks`:
//!   * plain file, end >= 0: bytes [start, end);  end < 0: the whole file (start is ignored);
//!   * `*.gz`, end >= 0: BGZF virtual offsets (block offset << 16 | offset in the inflated block);
//!   * `*.gz`, end < 0: the whole file; BGZF is read block by block, any other gzip as one stream.
//! Chunks are arbitrary pieces of the text (not line aligned); the caller carries partial lines.
//! Every BGZF block is checked (CRC32 and ISIZE); a truncated or corrupt file is an error.
use crate::error::{io, value, Result};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

const CHUNK: usize = 4 << 20;
const READ: usize = 1 << 20;

pub enum Source {
    Plain(PlainSource),
    Bgzf(Box<BgzfSource>),
    Gzip(GzipSource),
}

impl Source {
    pub fn open(path: &str, start: i64, end: i64, check_eof: bool) -> Result<Source> {
        let gz = path.ends_with(".gz");
        let mut fh = File::open(path).map_err(|e| crate::error::Error::Io(format!("{path}: {e}")))?;
        let size = fh.metadata()?.len();
        if gz {
            if end >= 0 {
                if start < 0 {
                    return value("negative start offset");
                }
                return Ok(Source::Bgzf(Box::new(BgzfSource::new(fh, size, start as u64, Some(end as u64), check_eof)?)));
            }
            let mut head = [0u8; 18];
            let got = read_full(&mut fh, &mut head)?;
            fh.seek(SeekFrom::Start(0))?;
            if bsize(&head[..got]) > 0 {
                return Ok(Source::Bgzf(Box::new(BgzfSource::new(fh, size, 0, None, check_eof)?)));
            }
            return Ok(Source::Gzip(GzipSource {
                dec: flate2::read::MultiGzDecoder::new(BufReader::with_capacity(READ, fh)),
                buf: vec![0u8; CHUNK],
                text: 0,
            }));
        }
        let (a, b) = if end < 0 { (0u64, size) } else { (start.max(0) as u64, end as u64) };
        if end >= 0 && start < 0 {
            return value("negative start offset");
        }
        if b > size {
            return io(format!("{path}: range end {b} is beyond the end of the file ({size} bytes); was it truncated?"));
        }
        fh.seek(SeekFrom::Start(a))?;
        Ok(Source::Plain(PlainSource { fh, left: b.saturating_sub(a), buf: vec![0u8; CHUNK], text: 0 }))
    }

    /// Next piece of text; None at the end of the range.
    pub fn next_chunk(&mut self) -> Result<Option<&[u8]>> {
        match self {
            Source::Plain(s) => s.next_chunk(),
            Source::Bgzf(s) => s.next_chunk(),
            Source::Gzip(s) => s.next_chunk(),
        }
    }

    /// Bytes of text delivered so far.
    pub fn text_bytes(&self) -> u64 {
        match self {
            Source::Plain(s) => s.text,
            Source::Bgzf(s) => s.text_total,
            Source::Gzip(s) => s.text,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Source::Plain(_) => "plain",
            Source::Bgzf(_) => "bgzf",
            Source::Gzip(_) => "gzip",
        }
    }
}

fn read_full(fh: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        let n = fh.read(&mut buf[got..])?;
        if n == 0 {
            break;
        }
        got += n;
    }
    Ok(got)
}

pub struct PlainSource {
    fh: File,
    left: u64,
    buf: Vec<u8>,
    text: u64,
}

impl PlainSource {
    fn next_chunk(&mut self) -> Result<Option<&[u8]>> {
        if self.left == 0 {
            return Ok(None);
        }
        let want = (self.left.min(self.buf.len() as u64)) as usize;
        let got = read_full(&mut self.fh, &mut self.buf[..want])?;
        if got < want {
            return io("pileup ended before the end of the range; was it truncated?");
        }
        self.left -= got as u64;
        self.text += got as u64;
        Ok(Some(&self.buf[..got]))
    }
}

pub struct GzipSource {
    dec: flate2::read::MultiGzDecoder<BufReader<File>>,
    buf: Vec<u8>,
    text: u64,
}

impl GzipSource {
    fn next_chunk(&mut self) -> Result<Option<&[u8]>> {
        // flate2 checks each member's CRC32 and ISIZE and reports a stream that ends early
        let got = read_full(&mut self.dec, &mut self.buf)
            .map_err(|e| crate::error::Error::Io(format!("gzip stream: {e} (truncated or corrupt file?)")))?;
        if got == 0 {
            return Ok(None);
        }
        self.text += got as u64;
        Ok(Some(&self.buf[..got]))
    }
}

/// Total size of the BGZF block whose first 18 bytes are `h`; 0 if this is not a BGZF block header
/// (same test as `sites._bgzf_bsize`).
pub fn bsize(h: &[u8]) -> usize {
    if h.len() < 18 || h[..4] != [0x1f, 0x8b, 0x08, 0x04] || &h[10..16] != b"\x06\x00BC\x02\x00" {
        return 0;
    }
    u16::from_le_bytes([h[16], h[17]]) as usize + 1
}

struct Inflater {
    #[cfg(feature = "libdeflate")]
    dec: libdeflater::Decompressor,
}

impl Inflater {
    fn new() -> Self {
        Inflater {
            #[cfg(feature = "libdeflate")]
            dec: libdeflater::Decompressor::new(),
        }
    }

    /// Appends the inflated block to `out`; CRC32 and ISIZE of the block are verified.
    fn inflate(&mut self, block: &[u8], off: u64, out: &mut Vec<u8>) -> Result<usize> {
        let n = block.len();
        if n < 26 {
            return io(format!("BGZF block at offset {off} is too short ({n} bytes)"));
        }
        let isize = u32::from_le_bytes([block[n - 4], block[n - 3], block[n - 2], block[n - 1]]) as usize;
        if isize > 1 << 16 {
            return io(format!("BGZF block at offset {off} claims {isize} bytes of data (corrupt file?)"));
        }
        let old = out.len();
        out.resize(old + isize, 0);
        let ok = self.raw(block, &mut out[old..]);
        if !ok {
            out.truncate(old);
            return io(format!("BGZF block at offset {off} is corrupt (inflate, CRC32 or ISIZE check failed)"));
        }
        Ok(isize)
    }

    #[cfg(feature = "libdeflate")]
    fn raw(&mut self, block: &[u8], out: &mut [u8]) -> bool {
        // libdeflate checks the gzip CRC32 and ISIZE; the length is compared as well
        let want = out.len();
        matches!(self.dec.gzip_decompress(block, out), Ok(n) if n == want)
    }

    #[cfg(not(feature = "libdeflate"))]
    fn raw(&mut self, block: &[u8], out: &mut [u8]) -> bool {
        let n = block.len();
        let body = &block[18..n - 8];
        let mut d = flate2::Decompress::new(false);
        match d.decompress(body, out, flate2::FlushDecompress::Finish) {
            Ok(flate2::Status::StreamEnd) => {}
            _ => return false,
        }
        if d.total_out() as usize != out.len() || d.total_in() as usize != body.len() {
            return false;
        }
        let mut c = flate2::Crc::new();
        c.update(out);
        c.sum() == u32::from_le_bytes([block[n - 8], block[n - 7], block[n - 6], block[n - 5]])
    }
}

pub struct BgzfSource {
    fh: File,
    size: u64,
    /// file offset of the next block to inflate
    off: u64,
    cbuf: Vec<u8>,
    cpos: usize,
    file_eof: bool,
    /// block offset where the range ends (None: at the end of the file) and bytes taken from that block
    c1: Option<u64>,
    u1: usize,
    /// bytes of text still to drop at the start of the range
    skip: usize,
    text: Vec<u8>,
    text_total: u64,
    done: bool,
    last_isize: Option<usize>,
    check_eof: bool,
    inf: Inflater,
}

impl BgzfSource {
    fn new(mut fh: File, size: u64, start: u64, end: Option<u64>, check_eof: bool) -> Result<Self> {
        let (c0, u0) = (start >> 16, (start & 0xFFFF) as usize);
        let (mut c1, mut u1) = match end {
            Some(e) => (Some(e >> 16), (e & 0xFFFF) as usize),
            None => (None, 0),
        };
        if let Some(c) = c1 {
            if c < c0 {
                return value("range end is before its start");
            }
            if c > size || (c == size && u1 > 0) {
                return io(format!("range ends at block offset {c}, beyond the end of the file ({size} bytes); was it truncated?"));
            }
            if c == size {
                (c1, u1) = (None, 0);
            }
        }
        if c0 > size {
            return io(format!("range starts at block offset {c0}, beyond the end of the file ({size} bytes); was it truncated?"));
        }
        fh.seek(SeekFrom::Start(c0))?;
        Ok(BgzfSource {
            fh, size, off: c0, cbuf: Vec::with_capacity(2 * READ), cpos: 0, file_eof: false, c1, u1, skip: u0,
            text: Vec::with_capacity(CHUNK + (1 << 16)), text_total: 0, done: false, last_isize: None, check_eof,
            inf: Inflater::new(),
        })
    }

    /// Makes `n` bytes of compressed data available at cpos; false if the file ends first.
    fn ensure(&mut self, n: usize) -> Result<bool> {
        while self.cbuf.len() - self.cpos < n {
            if self.file_eof {
                return Ok(false);
            }
            if self.cpos > 0 {
                self.cbuf.drain(..self.cpos);
                self.cpos = 0;
            }
            let old = self.cbuf.len();
            self.cbuf.resize(old + READ, 0);
            let got = self.fh.read(&mut self.cbuf[old..])?;
            self.cbuf.truncate(old + got);
            if got == 0 {
                self.file_eof = true;
            }
        }
        Ok(true)
    }

    /// Inflates the block at self.off, keeping at most `keep` bytes of it, after the start skip.
    fn block(&mut self, keep: Option<usize>, limit: Option<u64>) -> Result<()> {
        let off = self.off;
        if !self.ensure(18)? {
            return io(format!("BGZF file ends inside a block header at offset {off}: truncated file"));
        }
        let n = bsize(&self.cbuf[self.cpos..self.cpos + 18]);
        if n == 0 {
            return io(format!("no BGZF block at offset {off} (corrupt file, or a range that does not start at a block)"));
        }
        if let Some(l) = limit {
            if off + n as u64 > l {
                return value(format!("range end {l} is not at a BGZF block boundary"));
            }
        }
        if !self.ensure(n)? {
            return io(format!("BGZF block at offset {off} is cut short by the end of the file: truncated file"));
        }
        let old = self.text.len();
        let block = &self.cbuf[self.cpos..self.cpos + n];
        let isize = self.inf.inflate(block, off, &mut self.text)?;
        self.cpos += n;
        self.off += n as u64;
        self.last_isize = Some(isize);
        if let Some(k) = keep {
            self.text.truncate(old + k.min(isize));
        }
        if self.skip > 0 {
            let have = self.text.len() - old;
            let d = self.skip.min(have);
            self.text.drain(old..old + d);
            // as in sites._bgzf_blocks: the skip applies to the start of the range only
            self.skip -= d;
        }
        Ok(())
    }

    fn next_chunk(&mut self) -> Result<Option<&[u8]>> {
        self.text.clear();
        while !self.done && self.text.len() < CHUNK {
            match self.c1 {
                Some(c1) if self.off == c1 => {
                    if self.u1 > 0 {
                        self.block(Some(self.u1), None)?;
                    }
                    self.done = true;
                }
                Some(c1) if self.off > c1 => return value(format!("range end {c1} is not at a BGZF block boundary")),
                Some(c1) => self.block(None, Some(c1))?,
                None if self.off >= self.size => {
                    if self.check_eof && self.last_isize.map_or(false, |n| n != 0) {
                        return io("the BGZF file does not end with the empty end-of-file block: it is probably truncated \
                                   (pass check_eof=False to read it anyway)");
                    }
                    self.done = true;
                }
                None => self.block(None, None)?,
            }
        }
        self.text_total += self.text.len() as u64;
        if self.text.is_empty() {
            return Ok(None);
        }
        Ok(Some(&self.text))
    }
}
