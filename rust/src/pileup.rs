//! The pileup pass of one range: parse rows, look each row up in the motif-site keys of its contig,
//! sum counts per site and emit (contig, group, pos, n, k) when the contig ends.
//!
//! Row semantics follow `sites._parse` + `_Worker._sites`:
//!   * columns (tab or space separated): 0 contig, 1 position, 3 modification code, 5 strand,
//!     9 n_valid, 11 n_mod; lines starting with '#' are skipped;
//!   * rows with n_valid <= 0 or missing are dropped; codes other than a, m, 21839, h are dropped;
//!     strand '-' is 1, anything else 0; a missing n_mod counts as 0;
//!   * per site (motif occurrence): n = sum n_valid, k = sum n_mod over its rows; pos = lowest
//!     position among them; group = group of the site's keys.
//! Memory: the keys and accumulators of the current contig, the output rows not yet taken, and the
//! names of the contigs seen (for the contiguity check).
use crate::error::{value, Error, Result};
use crate::sites::{mod_code, ContigSites, SiteBuilder, POS_BITS};
use std::collections::{HashMap, HashSet};

const MAX_LINE: usize = 64 << 20;

/// Where the site keys of a contig come from.
pub enum Provider {
    /// motif scan + deduplication in Rust, on the encoded assembly passed with every call
    Scan { builder: Box<SiteBuilder>, offsets: Vec<i64>, lengths: Vec<i64> },
    /// keys computed by the caller (CSR over contigs; ascending within a contig)
    Keys { ptr: Vec<i64>, keys: Vec<u64>, group: Vec<i32>, site: Vec<u64> },
}

#[derive(Default)]
pub struct Out {
    pub contig: Vec<i32>,
    pub group: Vec<i32>,
    pub pos: Vec<i64>,
    pub n: Vec<i32>,
    pub k: Vec<i32>,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub lines: u64,
    pub rows_hit: u64,
    pub contigs_scanned: u64,
    pub keys: u64,
    pub scan_seconds: f64,
}

pub struct State {
    index: HashMap<Vec<u8>, u32>,
    wanted: Option<Vec<bool>>,
    provider: Provider,
    seen_known: Vec<bool>,
    seen_other: HashSet<Vec<u8>>,
    pub order: Vec<Vec<u8>>,
    carry: Vec<u8>,
    // current contig
    name: Vec<u8>,
    have_name: bool,
    active: bool,
    cid: i32,
    cs: ContigSites,
    cursor: usize,
    acc_n: Vec<i64>,
    acc_k: Vec<i64>,
    acc_pos: Vec<u64>,
    acc_group: Vec<i32>,
    pub out: Out,
    pub stats: Stats,
}

#[inline(always)]
fn is_sep(b: u8) -> bool {
    b == b'\t' || b == b' '
}

/// Next field of `line` from *p; moves *p past the separator. Empty when the line has ended.
#[inline(always)]
fn field<'a>(line: &'a [u8], p: &mut usize) -> &'a [u8] {
    let s = (*p).min(line.len());
    let mut i = s;
    while i < line.len() && !is_sep(line[i]) {
        i += 1;
    }
    *p = i + 1;
    &line[s..i]
}

/// Integer field: None when empty (a missing value); an error when it is not an integer.
#[inline(always)]
fn int(f: &[u8], what: &str, line: &[u8]) -> Result<Option<i64>> {
    let f = match f.last() {
        Some(b'\r') => &f[..f.len() - 1],
        _ => f,
    };
    if f.is_empty() {
        return Ok(None);
    }
    let (neg, digits) = if f[0] == b'-' { (true, &f[1..]) } else { (false, f) };
    if digits.is_empty() || digits.len() > 18 {
        return Err(bad_int(what, line));
    }
    let mut v = 0i64;
    for &b in digits {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return Err(bad_int(what, line));
        }
        v = v * 10 + d as i64;
    }
    Ok(Some(if neg { -v } else { v }))
}

#[cold]
fn bad_int(what: &str, line: &[u8]) -> Error {
    let shown = String::from_utf8_lossy(&line[..line.len().min(200)]).into_owned();
    Error::Value(format!("pileup: {what} is not an integer in line: {shown}"))
}

/// Lower bound of key in ks, starting from a cursor (rows of a contig are nearly sorted).
#[inline(always)]
fn seek(ks: &[u64], mut c: usize, key: u64) -> usize {
    let len = ks.len();
    if c < len && ks[c] < key {
        let mut step = 1usize;
        let mut lo = c + 1;
        loop {
            let hi = lo + step;
            if hi >= len || ks[hi] >= key {
                let hi = hi.min(len);
                return lo + ks[lo..hi].partition_point(|&x| x < key);
            }
            lo = hi;
            step <<= 1;
        }
    }
    let mut back = 0;
    while c > 0 && ks[c - 1] >= key {
        c -= 1;
        back += 1;
        if back > 16 {
            return ks[..c].partition_point(|&x| x < key);
        }
    }
    c
}

impl State {
    pub fn new(names: &[String], wanted: Option<&[i64]>, provider: Provider) -> Result<State> {
        let mut index = HashMap::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            index.insert(n.as_bytes().to_vec(), i as u32); // a later duplicate wins, as in Python
        }
        let wanted = match wanted {
            None => None,
            Some(w) => {
                let mut v = vec![false; names.len()];
                for &i in w {
                    if i >= 0 && (i as usize) < names.len() {
                        v[i as usize] = true;
                    }
                }
                Some(v)
            }
        };
        match &provider {
            Provider::Scan { offsets, lengths, .. } => {
                if offsets.len() != names.len() || lengths.len() != names.len() {
                    return value("offsets and lengths must have one entry per contig name");
                }
            }
            Provider::Keys { ptr, keys, group, site } => {
                if ptr.len() != names.len() + 1 || keys.len() != group.len() || keys.len() != site.len() {
                    return value("key_ptr must have len(names) + 1 entries and keys, groups, sites equal lengths");
                }
                let mut prev = 0i64;
                for &p in ptr {
                    if p < prev || p as usize > keys.len() {
                        return value("key_ptr must be non-decreasing and within the key array");
                    }
                    prev = p;
                }
                for c in 0..names.len() {
                    let ks = &keys[ptr[c] as usize..ptr[c + 1] as usize];
                    if ks.windows(2).any(|w| w[0] >= w[1]) {
                        return value(format!("keys of contig {} are not strictly ascending", names[c]));
                    }
                }
            }
        }
        Ok(State {
            index, wanted, provider, seen_known: vec![false; names.len()], seen_other: HashSet::new(), order: Vec::new(),
            carry: Vec::new(), name: Vec::new(), have_name: false, active: false, cid: -1, cs: ContigSites::default(),
            cursor: 0, acc_n: Vec::new(), acc_k: Vec::new(), acc_pos: Vec::new(), acc_group: Vec::new(),
            out: Out::default(), stats: Stats::default(),
        })
    }

    /// Emits the rows of the current contig.
    fn end_contig(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        for s in 0..self.acc_n.len() {
            let n = self.acc_n[s];
            if n == 0 {
                continue;
            }
            let k = self.acc_k[s];
            if n > i32::MAX as i64 || k > i32::MAX as i64 || k < i32::MIN as i64 {
                return Err(Error::Overflow(format!(
                    "summed counts of a site of contig {} do not fit 32 bits (n={n}, k={k})",
                    String::from_utf8_lossy(&self.name))));
            }
            self.out.contig.push(self.cid);
            self.out.group.push(self.acc_group[s]);
            self.out.pos.push(self.acc_pos[s] as i64);
            self.out.n.push(n as i32);
            self.out.k.push(k as i32);
        }
        Ok(())
    }

    #[cold]
    fn switch(&mut self, name: &[u8], codes: &[u8]) -> Result<()> {
        self.end_contig()?;
        if name.is_empty() {
            return value("pileup: a line starts with a separator (empty contig name)");
        }
        let id = self.index.get(name).copied();
        let fresh = match id {
            Some(i) => !std::mem::replace(&mut self.seen_known[i as usize], true),
            None => self.seen_other.insert(name.to_vec()),
        };
        if !fresh {
            return value(format!("pileup rows of contig {} are not contiguous; sort the pileup by contig",
                                 String::from_utf8_lossy(name)));
        }
        self.order.push(name.to_vec());
        self.name.clear();
        self.name.extend_from_slice(name);
        self.have_name = true;
        let i = match id {
            Some(i) if self.wanted.as_ref().map_or(true, |w| w[i as usize]) => i as usize,
            _ => return Ok(()),
        };
        match &mut self.provider {
            Provider::Scan { builder, offsets, lengths } => {
                let (o, l) = (offsets[i], lengths[i]);
                if o < 0 || l < 0 || (o + l) as usize > codes.len() {
                    return value(format!("contig {} lies outside the encoded assembly", String::from_utf8_lossy(name)));
                }
                let t = std::time::Instant::now();
                builder.build(&codes[o as usize..(o + l) as usize], &mut self.cs);
                self.stats.scan_seconds += t.elapsed().as_secs_f64();
            }
            Provider::Keys { ptr, keys, group, site } => {
                let (a, b) = (ptr[i] as usize, ptr[i + 1] as usize);
                self.cs.clear();
                self.cs.keys.extend_from_slice(&keys[a..b]);
                self.cs.group.extend_from_slice(&group[a..b]);
                self.cs.index_sites(&site[a..b]);
            }
        }
        self.stats.contigs_scanned += 1;
        self.stats.keys += self.cs.keys.len() as u64;
        if self.cs.keys.is_empty() {
            return Ok(());
        }
        let ns = self.cs.site_ids.len();
        self.acc_n.clear();
        self.acc_n.resize(ns, 0);
        self.acc_k.clear();
        self.acc_k.resize(ns, 0);
        self.acc_pos.clear();
        self.acc_pos.resize(ns, u64::MAX);
        self.acc_group.clear();
        self.acc_group.resize(ns, 0);
        self.cursor = 0;
        self.cid = i as i32;
        self.active = true;
        Ok(())
    }

    #[inline(always)]
    fn line(&mut self, line: &[u8], codes: &[u8]) -> Result<()> {
        if line.is_empty() || line[0] == b'#' || line == b"\r" {
            return Ok(());
        }
        self.stats.lines += 1;
        let nl = self.name.len();
        let mut q;
        if self.have_name && line.len() > nl && is_sep(line[nl]) && line[..nl] == self.name[..] {
            q = nl + 1;
        } else {
            q = 0;
            let name = field(line, &mut q);
            if !self.have_name || name != &self.name[..] {
                self.switch(name, codes)?;
            }
        }
        if !self.active {
            return Ok(());
        }
        let pos = match int(field(line, &mut q), "the position", line)? {
            Some(v) if v >= 0 && v < (1i64 << POS_BITS) => v as u64,
            _ => return Ok(()),
        };
        field(line, &mut q);
        let m = match mod_code(field(line, &mut q)) {
            Some(m) => m,
            None => return Ok(()),
        };
        field(line, &mut q);
        let strand: u64 = match field(line, &mut q) {
            b"" => return Ok(()),
            b"-" => 1,
            _ => 0,
        };
        let key = (pos << 3) | (strand << 2) | m;
        let c = seek(&self.cs.keys, self.cursor, key);
        self.cursor = c;
        if c >= self.cs.keys.len() || self.cs.keys[c] != key {
            return Ok(());
        }
        field(line, &mut q);
        field(line, &mut q);
        field(line, &mut q);
        let nv = match int(field(line, &mut q), "n_valid", line)? {
            Some(v) if v > 0 => v,
            _ => return Ok(()),
        };
        field(line, &mut q);
        let nm = int(field(line, &mut q), "n_mod", line)?.unwrap_or(0);
        if nv > i32::MAX as i64 || nm > i32::MAX as i64 || nm < i32::MIN as i64 {
            return Err(Error::Overflow("pileup: a count does not fit 32 bits".into()));
        }
        let s = self.cs.site[c] as usize;
        if self.acc_n[s] == 0 {
            self.acc_group[s] = self.cs.group[c];
        }
        self.acc_n[s] += nv;
        self.acc_k[s] += nm;
        if pos < self.acc_pos[s] {
            self.acc_pos[s] = pos;
        }
        self.stats.rows_hit += 1;
        Ok(())
    }

    /// buf: whole lines (the last one may lack its newline).
    fn process(&mut self, buf: &[u8], codes: &[u8]) -> Result<()> {
        let mut p = 0usize;
        while p < buf.len() {
            let e = match memchr::memchr(b'\n', &buf[p..]) {
                Some(i) => p + i,
                None => buf.len(),
            };
            self.line(&buf[p..e], codes)?;
            p = e + 1;
        }
        Ok(())
    }

    /// Next piece of the text, cut anywhere.
    pub fn feed(&mut self, data: &[u8], codes: &[u8]) -> Result<()> {
        let cut = match memchr::memrchr(b'\n', data) {
            Some(i) => i + 1,
            None => {
                if self.carry.len() + data.len() > MAX_LINE {
                    return value("pileup: a line is longer than 64 MB; this is not a text pileup");
                }
                self.carry.extend_from_slice(data);
                return Ok(());
            }
        };
        let mut from = 0;
        if !self.carry.is_empty() {
            from = memchr::memchr(b'\n', data).unwrap() + 1;
            let mut c = std::mem::take(&mut self.carry);
            c.extend_from_slice(&data[..from]);
            let r = self.process(&c, codes);
            c.clear();
            self.carry = c;
            r?;
        }
        self.process(&data[from..cut], codes)?;
        self.carry.extend_from_slice(&data[cut..]);
        Ok(())
    }

    /// End of the range: the last line (if it has no newline) and the last contig.
    pub fn finish(&mut self, codes: &[u8]) -> Result<()> {
        if !self.carry.is_empty() {
            let c = std::mem::take(&mut self.carry);
            self.process(&c, codes)?;
        }
        self.end_contig()
    }
}
