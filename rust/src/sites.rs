//! Motif-site keys of one contig: which (position, strand, modification) rows of the pileup are
//! motif sites, and the (group, site) each belongs to. Reproduces `sites._Worker.records`.
//!
//! key  = pos << 3 | strand << 2 | modification code        (within a contig)
//! site = canonical motif << 36 | occurrence                (within a contig)
//! A key matched by motifs of several groups goes to the most specific group (fewest records), and
//! within a group to its most general motif (most records): the first of the order
//! (group size, group, -motif rank, motif).
use crate::error::{value, Result};
use crate::motif::{MotifDef, Scanner, Scratch};

pub const POS_BITS: u32 = 33;

pub fn mod_code(mod_type: &[u8]) -> Option<u64> {
    match mod_type {
        b"a" => Some(0),
        b"m" => Some(1),
        b"21839" => Some(2),
        b"h" => Some(3),
        _ => None,
    }
}

/// Keys of the current contig, ascending, with the dense site index and the group of each.
#[derive(Default)]
pub struct ContigSites {
    pub keys: Vec<u64>,
    pub site: Vec<u32>,
    pub group: Vec<i32>,
    /// site ids (canonical motif << 36 | occurrence), ascending; index = dense site index
    pub site_ids: Vec<u64>,
}

impl ContigSites {
    pub fn clear(&mut self) {
        self.keys.clear();
        self.site.clear();
        self.group.clear();
        self.site_ids.clear();
    }

    /// Fills site and site_ids from one site id per key.
    pub fn index_sites(&mut self, ids: &[u64]) {
        self.site_ids.clear();
        self.site_ids.extend_from_slice(ids);
        self.site_ids.sort_unstable();
        self.site_ids.dedup();
        self.site.clear();
        self.site.reserve(ids.len());
        for id in ids {
            self.site.push(self.site_ids.partition_point(|x| x < id) as u32);
        }
    }
}

pub struct SiteBuilder {
    pub scanner: Scanner,
    scratch: Scratch,
    /// position of each motif in the order (group size, group, -rank, motif): lower wins a key
    prio: Vec<u32>,
    group_by_prio: Vec<i32>,
    modc: Vec<u64>,
    tmp: Vec<(u64, u32, u64)>,
    ids: Vec<u64>,
}

impl SiteBuilder {
    pub fn new(motifs: &[MotifDef], group_of: &[i64], motif_rank: &[i64]) -> Result<SiteBuilder> {
        if group_of.len() != motifs.len() || motif_rank.len() != motifs.len() {
            return value("group_of and motif_rank must have one entry per motif");
        }
        if motifs.is_empty() {
            return value("no motifs");
        }
        for (m, &g) in motifs.iter().zip(group_of) {
            if g < 0 || g > i32::MAX as i64 {
                return value("group ids must be non-negative 32-bit integers");
            }
            if m.mod_pos < 0 || m.mod_pos >= m.seq.len() as i64 {
                return value(format!("motif {}: modified position {} is outside the motif",
                                     String::from_utf8_lossy(&m.seq), m.mod_pos));
            }
        }
        let n_groups = *group_of.iter().max().unwrap() as usize + 1;
        let mut gsize = vec![0i64; n_groups];
        for (&g, &r) in group_of.iter().zip(motif_rank) {
            gsize[g as usize] += r;
        }
        let mut order: Vec<usize> = (0..motifs.len()).collect();
        order.sort_by_key(|&i| (gsize[group_of[i] as usize], group_of[i], std::cmp::Reverse(motif_rank[i]), i));
        let mut prio = vec![0u32; motifs.len()];
        for (p, &i) in order.iter().enumerate() {
            prio[i] = p as u32;
        }
        Ok(SiteBuilder {
            scanner: Scanner::new(motifs)?,
            scratch: Scratch::default(),
            prio,
            group_by_prio: order.iter().map(|&i| group_of[i] as i32).collect(),
            // as sites._Worker: a motif of an unknown modification type is looked up as code 3
            modc: motifs.iter().map(|m| mod_code(m.mod_type.as_bytes()).unwrap_or(3)).collect(),
            tmp: Vec::new(),
            ids: Vec::new(),
        })
    }

    /// Deduplicated site keys of one contig.
    pub fn build(&mut self, codes: &[u8], out: &mut ContigSites) {
        out.clear();
        self.tmp.clear();
        let (sc, tmp, prio, modc) = (&self.scanner, &mut self.tmp, &self.prio, &self.modc);
        sc.scan_all(codes, &mut self.scratch, |si, start, e| {
            let (pos, strand, mi, occ) = sc.record(si, start, e);
            let mi = mi as usize;
            let key = ((pos as u64) << 3) | ((strand as u64) << 2) | modc[mi];
            tmp.push((key, prio[mi], ((sc.canon[mi] as u64) << 36) | occ as u64));
        });
        // per key: the motif that comes first in the priority order
        tmp.sort_unstable();
        self.ids.clear();
        let mut last = u64::MAX;
        for &(key, p, site) in tmp.iter() {
            if key != last {
                out.keys.push(key);
                out.group.push(self.group_by_prio[p as usize]);
                self.ids.push(site);
                last = key;
            }
        }
        out.index_sites(&self.ids);
    }
}
