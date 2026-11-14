//! IUPAC motif matching on a 2-bit encoded sequence (A, C, G, T = 0..3; anything else = 4), both
//! strands. Reproduces `epirefine.motifs.MotifMatcher`: the same records in the same order.
//!
//! Each (motif, orientation) is a pattern: the motif read on + (orientation 0) or its reverse
//! complement read on + (orientation 1, the motif on -). Patterns with the same span and non-N
//! offsets share a shape; degenerate bases are expanded into concrete codes. One pass per shape.
use crate::error::{value, Result};
use std::collections::HashMap;

const PREFIX: usize = 10; // fixed bases resolved with a bit table; the rest are checked on candidates
const BLOCK: usize = 4096;

#[derive(Clone)]
pub struct MotifDef {
    pub seq: Vec<u8>,
    pub mod_pos: i64,
    pub mod_type: String,
}

fn iupac(c: u8) -> Option<&'static [u8]> {
    Some(match c {
        b'A' => b"A", b'C' => b"C", b'G' => b"G", b'T' => b"T",
        b'R' => b"AG", b'Y' => b"CT", b'S' => b"CG", b'W' => b"AT", b'K' => b"GT", b'M' => b"AC",
        b'B' => b"CGT", b'D' => b"AGT", b'H' => b"ACT", b'V' => b"ACG", b'N' => b"ACGT",
        _ => return None,
    })
}

fn comp(c: u8) -> u8 {
    match c {
        b'A' => b'T', b'C' => b'G', b'G' => b'C', b'T' => b'A', b'R' => b'Y', b'Y' => b'R', b'S' => b'S', b'W' => b'W',
        b'K' => b'M', b'M' => b'K', b'B' => b'V', b'D' => b'H', b'H' => b'D', b'V' => b'B', other => other,
    }
}

pub fn revcomp(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&c| comp(c)).collect()
}

fn base_index(c: u8) -> u64 {
    match c {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        _ => 3,
    }
}

pub struct Shape {
    pub span: usize,
    pub offs: Vec<usize>,
    k1: usize,
    /// bit table over the first k1 bases
    lut: Vec<u64>,
    ucodes: Vec<u64>,
    ptr: Vec<u32>,
    pub mi: Vec<i32>,
    pub orient: Vec<i8>,
    // window scan (span <= 32): the bases of the last 32 positions are kept as one 64-bit word
    /// the span does not fit the window: scanned offset by offset instead
    wide: bool,
    /// window bits of the non-N offsets (bit = span - 1 - offset)
    nmask: u32,
    /// runs of adjacent non-N offsets as (shift, mask, bits): the code is their concatenation
    runs: Vec<(u32, u64, u32)>,
    /// code >> pre_shift = index into the bit table
    pre_shift: u32,
}

pub struct Scanner {
    pub shapes: Vec<Shape>,
    pub canon: Vec<i32>,
    pub flip: Vec<bool>,
    pub mod_pos: Vec<i64>,
    pub length: Vec<i64>,
    pub palin: Vec<bool>,
}

/// Work buffers of a scan (one per thread of use).
pub struct Scratch {
    key: Vec<u32>,
    bad: Vec<u8>,
    win: Vec<u64>,
    amb: Vec<u32>,
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch { key: vec![0; BLOCK], bad: vec![0; BLOCK], win: vec![0; BLOCK], amb: vec![0; BLOCK] }
    }
}

/// Records of `MotifMatcher.find`, as columns.
#[derive(Default)]
pub struct Records {
    pub pos: Vec<i64>,
    pub strand: Vec<i8>,
    pub motif: Vec<i32>,
    pub occ_motif: Vec<i32>,
    pub occ: Vec<i64>,
}

impl Scanner {
    pub fn new(motifs: &[MotifDef]) -> Result<Scanner> {
        // canonical motif: a motif and its reverse complement (same modification type) share one
        let mut by_key: HashMap<(&[u8], &str), usize> = HashMap::new();
        for (i, m) in motifs.iter().enumerate() {
            by_key.insert((&m.seq[..], m.mod_type.as_str()), i); // a later duplicate wins, as in Python
        }
        let mut canon = Vec::with_capacity(motifs.len());
        for (i, m) in motifs.iter().enumerate() {
            let rc = revcomp(&m.seq);
            let j = by_key.get(&(&rc[..], m.mod_type.as_str())).copied().unwrap_or(i);
            canon.push(i.min(j) as i32);
        }
        let flip = canon.iter().enumerate().map(|(i, &c)| c as usize != i).collect();
        let palin = motifs.iter().map(|m| revcomp(&m.seq) == m.seq).collect();

        // shapes in order of first appearance; entries (code, motif, orientation)
        let mut index: HashMap<(usize, Vec<usize>), usize> = HashMap::new();
        let mut groups: Vec<((usize, Vec<usize>), Vec<(u64, i32, i8)>)> = Vec::new();
        for (mi, m) in motifs.iter().enumerate() {
            let name = String::from_utf8_lossy(&m.seq).into_owned();
            if m.seq.is_empty() {
                return value("empty motif sequence");
            }
            for (orient, seq) in [(0i8, m.seq.clone()), (1i8, revcomp(&m.seq))] {
                let mut offs = Vec::new();
                let mut choices: Vec<&'static [u8]> = Vec::new();
                let mut combos = 1f64;
                for (i, &c) in seq.iter().enumerate() {
                    let ch = match iupac(c) {
                        Some(ch) => ch,
                        None => return value(format!("motif {name}: '{}' is not an IUPAC base", c as char)),
                    };
                    if c != b'N' {
                        offs.push(i);
                        choices.push(ch);
                        combos *= ch.len() as f64;
                    }
                }
                if combos > 65536.0 {
                    return value(format!("motif {name} expands to too many concrete sequences"));
                }
                if offs.len() > 32 {
                    return value(format!("motif {name} has more than 32 non-N positions"));
                }
                let key = (seq.len(), offs);
                let gi = match index.get(&key) {
                    Some(&g) => g,
                    None => {
                        index.insert(key.clone(), groups.len());
                        groups.push((key, Vec::new()));
                        groups.len() - 1
                    }
                };
                // all concrete sequences (the entries are sorted afterwards, so the order is free)
                let mut codes = vec![0u64];
                for ch in &choices {
                    let mut next = Vec::with_capacity(codes.len() * ch.len());
                    for &c in &codes {
                        for &b in ch.iter() {
                            next.push(c * 4 + base_index(b));
                        }
                    }
                    codes = next;
                }
                for c in codes {
                    groups[gi].1.push((c, mi as i32, orient));
                }
            }
        }
        let mut shapes = Vec::with_capacity(groups.len());
        for ((span, offs), mut ents) in groups {
            ents.sort_unstable();
            let k1 = offs.len().min(PREFIX);
            let mut lut = vec![0u64; ((1usize << (2 * k1)) + 63) / 64];
            let mut ucodes: Vec<u64> = Vec::new();
            let mut ptr: Vec<u32> = Vec::new();
            for (i, e) in ents.iter().enumerate() {
                if ucodes.last() != Some(&e.0) {
                    ucodes.push(e.0);
                    ptr.push(i as u32);
                    let p = (e.0 >> (2 * (offs.len() - k1))) as usize;
                    lut[p >> 6] |= 1 << (p & 63);
                }
            }
            ptr.push(ents.len() as u32);
            let wide = span > 32;
            let (mut nmask, mut runs) = (0u32, Vec::new());
            if !wide {
                let mut i = 0;
                while i < offs.len() {
                    let mut j = i;
                    while j + 1 < offs.len() && offs[j + 1] == offs[j] + 1 {
                        j += 1;
                    }
                    let bits = 2 * (j - i + 1) as u32;
                    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
                    runs.push((2 * (span - 1 - offs[j]) as u32, mask, bits));
                    i = j + 1;
                }
                for &o in &offs {
                    nmask |= 1 << (span - 1 - o);
                }
            }
            let pre_shift = 2 * (offs.len() - k1) as u32;
            shapes.push(Shape {
                span, offs, k1, lut, ucodes, ptr, wide, nmask, runs, pre_shift,
                mi: ents.iter().map(|e| e.1).collect(),
                orient: ents.iter().map(|e| e.2).collect(),
            });
        }
        Ok(Scanner {
            shapes, canon, flip, palin,
            mod_pos: motifs.iter().map(|m| m.mod_pos).collect(),
            length: motifs.iter().map(|m| m.seq.len() as i64).collect(),
        })
    }

    /// Calls f(start, entry index) for every match of shape `si` in `codes`, by start, then by entry.
    #[inline]
    pub fn scan<F: FnMut(usize, usize)>(&self, si: usize, codes: &[u8], scratch: &mut Scratch, mut f: F) {
        let sh = &self.shapes[si];
        if codes.len() < sh.span {
            return;
        }
        let n = codes.len() - sh.span + 1;
        let (key, bad) = (&mut scratch.key, &mut scratch.bad);
        let (head, rest) = sh.offs.split_at(sh.k1);
        let mut b0 = 0;
        while b0 < n {
            let m = BLOCK.min(n - b0);
            let (key, bad) = (&mut key[..m], &mut bad[..m]);
            key.fill(0);
            bad.fill(0);
            for &o in head {
                let c = &codes[b0 + o..b0 + o + m];
                for i in 0..m {
                    key[i] = (key[i] << 2) | (c[i] & 3) as u32;
                    bad[i] |= (c[i] == 4) as u8;
                }
            }
            for i in 0..m {
                let k = key[i] as usize;
                if bad[i] != 0 || (sh.lut[k >> 6] >> (k & 63)) & 1 == 0 {
                    continue;
                }
                let p = b0 + i;
                let mut full = k as u64;
                if !rest.is_empty() {
                    let mut ok = true;
                    for &o in rest {
                        let c = codes[p + o];
                        full = (full << 2) | (c & 3) as u64;
                        ok &= c != 4;
                    }
                    if !ok {
                        continue;
                    }
                }
                if let Ok(j) = sh.ucodes.binary_search(&full) {
                    for e in sh.ptr[j]..sh.ptr[j + 1] {
                        f(p, e as usize);
                    }
                }
            }
            b0 += m;
        }
    }

    /// (pos, strand, motif, occ) of the match of entry `e` of shape `si` at `start`.
    #[inline]
    pub fn record(&self, si: usize, start: usize, e: usize) -> (i64, i8, i32, i64) {
        let sh = &self.shapes[si];
        let (mi, orient) = (sh.mi[e] as usize, sh.orient[e]);
        let start = start as i64;
        let fwd = orient == 0;
        let pos = if fwd { start + self.mod_pos[mi] } else { start + self.length[mi] - 1 - self.mod_pos[mi] };
        // occ = occurrence start * 2 + orientation of the canonical motif; palindromes share occ on
        // both strands, a motif and its reverse complement share occ at one occurrence
        let mut occ = start * 2 + if fwd || self.palin[mi] { 0 } else { 1 };
        if self.flip[mi] {
            occ ^= 1;
        }
        (pos, orient, mi as i32, occ)
    }

    /// Calls f(shape, start, entry index) for every match in `codes`; for one shape the calls come by
    /// start, then by entry (shapes are interleaved).
    pub fn scan_all<F: FnMut(usize, usize, usize)>(&self, codes: &[u8], scratch: &mut Scratch, mut f: F) {
        // window scan: win[i] holds the 32 bases ending at position q0 + i (2 bits each, the last base
        // lowest), amb[i] one bit per base that is not A, C, G or T
        let (mut w, mut a) = (0u64, 0u32);
        let mut q0 = 0;
        while q0 < codes.len() {
            let m = BLOCK.min(codes.len() - q0);
            let (win, amb) = (&mut scratch.win[..m], &mut scratch.amb[..m]);
            for (i, &c) in codes[q0..q0 + m].iter().enumerate() {
                w = (w << 2) | (c & 3) as u64;
                a = (a << 1) | (c == 4) as u32;
                win[i] = w;
                amb[i] = a;
            }
            for (si, sh) in self.shapes.iter().enumerate() {
                if sh.wide {
                    continue;
                }
                let last = sh.span - 1; // a match starting at p ends at q = p + last
                let from = last.saturating_sub(q0);
                for i in from..m {
                    if amb[i] & sh.nmask != 0 {
                        continue;
                    }
                    let v = win[i];
                    let mut code = 0u64;
                    for &(shift, mask, bits) in &sh.runs {
                        code = (if bits == 64 { 0 } else { code << bits }) | ((v >> shift) & mask);
                    }
                    let k = (code >> sh.pre_shift) as usize;
                    if (sh.lut[k >> 6] >> (k & 63)) & 1 == 0 {
                        continue;
                    }
                    if let Ok(j) = sh.ucodes.binary_search(&code) {
                        for e in sh.ptr[j]..sh.ptr[j + 1] {
                            f(si, q0 + i - last, e as usize);
                        }
                    }
                }
            }
            q0 += m;
        }
        for si in 0..self.shapes.len() {
            if self.shapes[si].wide {
                self.scan(si, codes, scratch, |p, e| f(si, p, e));
            }
        }
    }

    fn push(&self, si: usize, p: usize, e: usize, out: &mut Records) {
        let (pos, strand, mi, occ) = self.record(si, p, e);
        out.pos.push(pos);
        out.strand.push(strand);
        out.motif.push(mi);
        out.occ_motif.push(self.canon[mi as usize]);
        out.occ.push(occ);
    }

    /// `MotifMatcher.find(codes)`: records by shape, then start, then entry.
    pub fn find(&self, codes: &[u8], out: &mut Records) {
        let mut scratch = Scratch::default();
        let mut hits: Vec<Vec<(usize, u32)>> = vec![Vec::new(); self.shapes.len()];
        self.scan_all(codes, &mut scratch, |si, p, e| hits[si].push((p, e as u32)));
        for (si, h) in hits.iter().enumerate() {
            for &(p, e) in h {
                self.push(si, p, e as usize, out);
            }
        }
    }

    /// Records of several contigs in contig coordinates, in the order `MotifMatcher.find` gives on
    /// their concatenation (shape, then contig, then start); cidx = index into `seqs`.
    pub fn find_contigs(&self, seqs: &[&[u8]], cidx: &mut Vec<i64>, out: &mut Records) {
        let mut scratch = Scratch::default();
        let mut hits: Vec<Vec<(u32, usize, u32)>> = vec![Vec::new(); self.shapes.len()];
        for (ci, codes) in seqs.iter().enumerate() {
            self.scan_all(codes, &mut scratch, |si, p, e| hits[si].push((ci as u32, p, e as u32)));
        }
        for (si, h) in hits.iter().enumerate() {
            for &(ci, p, e) in h {
                cidx.push(ci as i64);
                self.push(si, p, e as usize, out);
            }
        }
    }
}
