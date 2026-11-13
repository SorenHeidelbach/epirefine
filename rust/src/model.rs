//! Model kernels: passes over the [units, groups, grid] likelihood tables of EpiRefine's model that numpy
//! makes as a chain of whole-array operations (one temporary per operation). Here each unit is read once
//! and nothing larger than one grid row is written. Tables are float32 as EpiRefine stores them; all
//! arithmetic is float64, in the order of EpiRefine's expressions, so results differ from the numpy path
//! only by the last bits of exp/log and of sums (relative 1e-15).
use rayon::prelude::*;
use std::sync::OnceLock;

static AVX2_FMA: OnceLock<bool> = OnceLock::new();

/// Whether the AVX2+FMA builds of the bodies run: the CPU has both, and EPIREFINE_RS_NO_SIMD is unset (or 0,
/// false, no, off). Read once per process. Both builds do the same arithmetic, so the switch is for tests and
/// for ruling the CPU path out when results are compared across machines.
pub fn avx2_fma() -> bool {
    *AVX2_FMA.get_or_init(|| {
        let off = std::env::var("EPIREFINE_RS_NO_SIMD")
            .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "0" | "false" | "no" | "off"))
            .unwrap_or(false);
        if off {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// The CPU path of the model kernels in this process: "avx2_fma", or "generic" (the bodies built for the target's
/// baseline instruction set: x86-64 SSE2, or NEON on aarch64).
pub fn cpu_path() -> &'static str {
    if avx2_fma() { "avx2_fma" } else { "generic" }
}

/// numpy's pairwise summation (the order of `ndarray.sum` over a contiguous axis).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut s = 0.0;
        for &x in a {
            s += x;
        }
        s
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// exp(x) to about one ulp, branch-free so that loops over slices vectorise: Cody-Waite reduction by ln 2,
/// a degree-13 Taylor polynomial in Horner form with fused multiply-adds (exactly rounded, so the same bits with
/// or without FMA hardware), scaling by 2^k through the exponent bits. Results below 2^-1022 (x < -708) are 0
/// instead of subnormal; +inf above 709; NaN stays NaN. libm's exp was most of the time of the kernels that
/// take a log sum exp per (unit, group).
#[inline(always)]
pub fn exp_fast(x: f64) -> f64 {
    const LOG2E: f64 = std::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.931_471_803_691_238_164_90e-01;
    const LN2_LO: f64 = 1.908_214_929_270_587_700_02e-10;
    const MAGIC: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52: adding it rounds to an integer
    const C: [f64; 14] = [
        1.0 / 6_227_020_800.0, 1.0 / 479_001_600.0, 1.0 / 39_916_800.0, 1.0 / 3_628_800.0, 1.0 / 362_880.0,
        1.0 / 40_320.0, 1.0 / 5_040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0,
    ];
    let xc = if x > 709.0 { 709.0 } else if x < -708.0 { -708.0 } else { x };
    let t = xc.mul_add(LOG2E, MAGIC);
    let k = t - MAGIC;
    let r = k.mul_add(-LN2_LO, k.mul_add(-LN2_HI, xc));
    let mut p = C[0];
    for &c in &C[1..] {
        p = p.mul_add(r, c);
    }
    let ki = (t.to_bits() as i64).wrapping_sub(MAGIC.to_bits() as i64);
    let y = p * f64::from_bits(((ki + 1023) as u64) << 52);
    if x < -708.0 {
        0.0
    } else if x > 709.0 {
        f64::INFINITY
    } else if x.is_nan() {
        x
    } else {
        y
    }
}

/// log sum exp with the arithmetic of EpiRefine's `model.logsumexp` (scipy's): the largest elements (m of
/// them when tied) leave the sum, result = log1p(sum(exp(a - max)) / m) + log(m) + max. `e` is scratch.
/// The exponentials are exp_fast's (within about one ulp of numpy's).
#[inline(always)]
pub fn logsumexp(a: &[f64], e: &mut [f64]) -> f64 {
    let mut mx = f64::NEG_INFINITY;
    let mut nan = false;
    for &x in a {
        if x > mx {
            mx = x;
        }
        nan |= x.is_nan();
    }
    if nan || !mx.is_finite() {
        let mut s = 0.0;
        for &x in a {
            s += x.exp();
        }
        return s.ln();
    }
    let mut m = 0.0;
    for &x in a {
        m += if x == mx { 1.0 } else { 0.0 };
    }
    for (o, &x) in e.iter_mut().zip(a) {
        let v = exp_fast(x - mx);
        *o = if x == mx { 0.0 } else { v };
    }
    let mut s = pairwise_sum(&e[..a.len()]);
    if s != 0.0 {
        s /= m;
    }
    s.ln_1p() + m.ln() + mx
}

/// `$name` calls `$body` (an #[inline(always)] fn) compiled for AVX2 and FMA when avx2_fma() says so, otherwise
/// as it is: the same arithmetic either way (no contraction; explicit fused multiply-adds are exactly rounded).
macro_rules! simd_dispatch {
    ($name:ident, $avx:ident, $body:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx($($arg: $ty),*) -> $ret {
            $body($($arg),*)
        }
        fn $name($($arg: $ty),*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            {
                if avx2_fma() {
                    return unsafe { $avx($($arg),*) };
                }
            }
            $body($($arg),*)
        }
    };
}

/// Dot product with eight running sums (vectorises; the order is fixed, so results do not depend on threads).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut r = [0.0f64; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ta, tb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for j in 0..8 {
            r[j] += x[j] * y[j];
        }
    }
    let mut s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (x, y) in ta.iter().zip(tb) {
        s += x * y;
    }
    s
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().expect("thread pool")
}

/// ell[h, m, g] = sum over the chunks k of bin h (ascending k) of w[k] * lp[k, m, g].
pub fn bin_loglik(lp: &[f32], mg: usize, chunk_bin: &[i64], w: &[f64], n_bins: usize, threads: usize) -> Vec<f64> {
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_bins];
    for (k, &b) in chunk_bin.iter().enumerate() {
        members[b as usize].push(k);
    }
    let mut ell = vec![0.0f64; n_bins * mg];
    pool(threads).install(|| {
        ell.par_chunks_mut(mg).zip(members.par_iter()).for_each(|(out, ks)| {
            for &k in ks {
                let wk = w[k];
                let row = &lp[k * mg..(k + 1) * mg];
                for (o, &x) in out.iter_mut().zip(row) {
                    *o += wk * x as f64;
                }
            }
        })
    });
    ell
}

/// s[k] = sum over groups of logsumexp_g(lp[k, m, g] + log_pi[m, g]): a unit under a community draw.
#[inline(always)]
fn community_row_body(row: &[f32], m: usize, g: usize, log_pi: &[f64], a: &mut [f64], e: &mut [f64], t: &mut [f64]) -> f64 {
    for j in 0..m {
        for i in 0..g {
            a[i] = row[j * g + i] as f64 + log_pi[j * g + i];
        }
        t[j] = logsumexp(a, e);
    }
    pairwise_sum(t)
}
simd_dispatch!(community_row, community_row_avx2, community_row_body,
               (row: &[f32], m: usize, g: usize, log_pi: &[f64], a: &mut [f64], e: &mut [f64], t: &mut [f64]) -> f64);

pub fn community_terms(lp: &[f32], m: usize, g: usize, log_pi: &[f64], threads: usize) -> Vec<f64> {
    let mg = m * g;
    let mut out = vec![0.0f64; lp.len() / mg];
    pool(threads).install(|| {
        out.par_iter_mut().zip(lp.par_chunks(mg)).for_each_init(
            || (vec![0.0f64; g], vec![0.0f64; g], vec![0.0f64; m]),
            |(a, e, t), (o, row)| *o = community_row(row, m, g, log_pi, a, e, t),
        )
    });
    out
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn lco_row_body(row: &[f32], el: &[f64], wk: f64, m: usize, g: usize, log_pi: &[f64], a: &mut [f64], e: &mut [f64],
                t: &mut [f64]) -> f64 {
    for j in 0..m {
        for i in 0..g {
            a[i] = (el[j * g + i] - wk * row[j * g + i] as f64) + log_pi[j * g + i];
        }
        let l = logsumexp(a, e);
        for i in 0..g {
            a[i] = row[j * g + i] as f64 + (a[i] - l);
        }
        t[j] = logsumexp(a, e);
    }
    pairwise_sum(t)
}
simd_dispatch!(lco_row, lco_row_avx2, lco_row_body,
               (row: &[f32], el: &[f64], wk: f64, m: usize, g: usize, log_pi: &[f64], a: &mut [f64], e: &mut [f64],
                t: &mut [f64]) -> f64);

/// s[k] = sum over groups of logsumexp_g(lp[k] + prof), prof = the profile of the chunk's bin without the
/// chunk: host_profiles(ell[bin] - w[k] * lp[k], log_pi) = lp_ + log_pi - logsumexp_g(lp_ + log_pi).
pub fn leave_chunk_out_terms(lp: &[f32], m: usize, g: usize, chunk_bin: &[i64], w: &[f64], ell: &[f64], log_pi: &[f64],
                             threads: usize) -> Vec<f64> {
    let mg = m * g;
    let mut out = vec![0.0f64; lp.len() / mg];
    pool(threads).install(|| {
        out.par_iter_mut().enumerate().zip(lp.par_chunks(mg)).for_each_init(
            || (vec![0.0f64; g], vec![0.0f64; g], vec![0.0f64; m]),
            |(a, e, t), ((k, o), row)| {
                *o = lco_row(row, &ell[chunk_bin[k] as usize * mg..][..mg], w[k], m, g, log_pi, a, e, t)
            },
        )
    });
    out
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn npmle_rounds_body(lk: &[f64], g: usize, b: usize, n: usize, p: &mut [f64], acc: &mut [f64], inv: &mut [f64],
                     snap: &mut [f64], delta: &mut [f64]) {
    for it in 0..n {
        for (row, o) in lk.chunks_exact(g).zip(inv.iter_mut()) {
            *o = 1.0 / dot(row, p).max(1e-300);
        }
        acc.iter_mut().for_each(|x| *x = 0.0);
        for (row, &w) in lk.chunks_exact(g).zip(inv.iter()) {
            for i in 0..g {
                acc[i] += w * row[i];
            }
        }
        let mut dmax = 0.0f64;
        let out = &mut snap[it * g..(it + 1) * g];
        for i in 0..g {
            let new = (p[i] * acc[i] / b as f64).max(1e-300);
            dmax = dmax.max((new - p[i]).abs());
            out[i] = new;
        }
        p.copy_from_slice(out);
        delta[it] = dmax;
    }
}
simd_dispatch!(npmle_rounds, npmle_rounds_avx2, npmle_rounds_body,
               (lk: &[f64], g: usize, b: usize, n: usize, p: &mut [f64], acc: &mut [f64], inv: &mut [f64],
                snap: &mut [f64], delta: &mut [f64]) -> ());

/// Grid NPMLE by EM for every group (EpiRefine's `model.npmle`): lik[m, b, g], pi[m, g] the start, returns pi.
/// Per iteration and group: denom[b] = max(lik[b] . pi, 1e-300), new[g] = max(pi[g] * sum_b lik[b, g] / denom[b] / B,
/// 1e-300). The fit stops at the first iteration in which no cell of any group moves by `tol` or more, as the
/// all-groups loop does: groups advance `ROUND` iterations at a time on threads, keeping each iteration's
/// state, and the stop is looked up over all groups after each round.
pub fn npmle(lik: &[f64], m: usize, b: usize, g: usize, pi0: &[f64], iters: usize, tol: f64, threads: usize) -> Vec<f64> {
    const ROUND: usize = 16;
    let mut pi = pi0.to_vec();
    if m == 0 || b == 0 || g == 0 {
        return pi;
    }
    let mut snaps = vec![0.0f64; m * ROUND * g];
    let mut deltas = vec![0.0f64; m * ROUND];
    let pool = pool(threads);
    let mut done = 0;
    while done < iters {
        let n = ROUND.min(iters - done);
        pool.install(|| {
            snaps.par_chunks_mut(ROUND * g).zip(deltas.par_chunks_mut(ROUND)).enumerate().for_each_init(
                || (vec![0.0f64; g], vec![0.0f64; g], vec![0.0f64; b]),
                |(p, acc, inv), (j, (snap, delta))| {
                    p.copy_from_slice(&pi[j * g..(j + 1) * g]);
                    npmle_rounds(&lik[j * b * g..(j + 1) * b * g], g, b, n, p, acc, inv, snap, delta)
                },
            )
        });
        let mut stop = None;
        for it in 0..n {
            let moved = (0..m).map(|j| deltas[j * ROUND + it]).fold(0.0f64, f64::max);
            if moved < tol {
                stop = Some(it);
                break;
            }
        }
        let last = stop.unwrap_or(n - 1);
        for j in 0..m {
            pi[j * g..(j + 1) * g].copy_from_slice(&snaps[(j * ROUND + last) * g..(j * ROUND + last + 1) * g]);
        }
        if stop.is_some() {
            break;
        }
        done += n;
    }
    pi
}

/// numpy's pairwise summation over rows lo..hi of the products w[i] * table[r[i], :], all G columns at once
/// (acc and tmp are scratch of 8 * G and G).
fn pairwise_rows(table: &[f64], g: usize, r: &[i64], w: &[f64], lo: usize, hi: usize, out: &mut [f64], acc: &mut [f64]) {
    let n = hi - lo;
    let row = |i: usize| (&table[r[i] as usize * g..][..g], w[i]);
    if n < 8 {
        out.iter_mut().for_each(|x| *x = 0.0);
        for i in lo..hi {
            let (t, wi) = row(i);
            for j in 0..g {
                out[j] += t[j] * wi;
            }
        }
    } else if n <= 128 {
        for q in 0..8 {
            let (t, wi) = row(lo + q);
            for j in 0..g {
                acc[q * g + j] = t[j] * wi;
            }
        }
        let mut i = lo + 8;
        while i + 8 <= hi {
            for q in 0..8 {
                let (t, wi) = row(i + q);
                for j in 0..g {
                    acc[q * g + j] += t[j] * wi;
                }
            }
            i += 8;
        }
        for j in 0..g {
            out[j] = ((acc[j] + acc[g + j]) + (acc[2 * g + j] + acc[3 * g + j]))
                + ((acc[4 * g + j] + acc[5 * g + j]) + (acc[6 * g + j] + acc[7 * g + j]));
        }
        while i < hi {
            let (t, wi) = row(i);
            for j in 0..g {
                out[j] += t[j] * wi;
            }
            i += 1;
        }
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        let mut right = vec![0.0f64; g];
        pairwise_rows(table, g, r, w, lo, lo + n2, out, acc);
        pairwise_rows(table, g, r, w, lo + n2, hi, &mut right, acc);
        for j in 0..g {
            out[j] += right[j];
        }
    }
}

/// out[c, g] = sum over the rows i of cell c (starts[c] <= i < starts[c + 1]) of w[i] * table[r[i], g], added
/// as `np.add.reduceat(table[r] * w[:, None], starts, axis=0)` adds them: the first row plus numpy's pairwise
/// sum of the others. The gather, product and sum of EpiRefine's unit tables in one pass, with the same bits.
pub fn cell_sums(table: &[f64], g: usize, r: &[i64], w: &[f64], starts: &[i64], threads: usize) -> Vec<f64> {
    let n_cells = starts.len();
    let mut out = vec![0.0f64; n_cells * g];
    let work = |(rest, acc): &mut (Vec<f64>, Vec<f64>), (c, o): (usize, &mut [f64])| {
        let lo = starts[c] as usize;
        let hi = if c + 1 < n_cells { starts[c + 1] as usize } else { r.len() };
        let t0 = &table[r[lo] as usize * g..][..g];
        for j in 0..g {
            o[j] = t0[j] * w[lo];
        }
        if hi > lo + 1 {
            pairwise_rows(table, g, r, w, lo + 1, hi, rest, acc);
            for j in 0..g {
                o[j] += rest[j];
            }
        }
    };
    let scratch = || (vec![0.0f64; g], vec![0.0f64; 8 * g]);
    if threads <= 1 {
        let mut s = scratch();
        out.chunks_mut(g).enumerate().for_each(|x| work(&mut s, x));
    } else {
        pool(threads).install(|| out.par_chunks_mut(g).enumerate().for_each_init(scratch, work));
    }
    out
}

/// numpy's logaddexp (npy_logaddexp): the larger argument plus log1p(exp(-|difference|)).
#[inline]
fn logaddexp(x: f64, y: f64) -> f64 {
    if x == y {
        return x + std::f64::consts::LN_2;
    }
    let t = x - y;
    if t > 0.0 {
        x + (-t).exp().ln_1p()
    } else if t <= 0.0 {
        y + t.exp().ln_1p()
    } else {
        t
    }
}

/// q[t] = sum over k of row[k] * ebm[k * h + t] (k ascending; zero entries of row skipped).
#[inline(always)]
fn products_body(row: &[f64], ebm: &[f64], h: usize, q: &mut [f64]) {
    q.iter_mut().for_each(|x| *x = 0.0);
    for (k, &a) in row.iter().enumerate() {
        if a != 0.0 {
            let col = &ebm[k * h..(k + 1) * h];
            for (o, &c) in q.iter_mut().zip(col) {
                *o += a * c;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn products_avx2(row: &[f64], ebm: &[f64], h: usize, q: &mut [f64]) {
    products_body(row, ebm, h, q)
}

/// The same additions in the same order with or without AVX2 (no fused multiply-add), so the same bits.
#[inline]
fn products(row: &[f64], ebm: &[f64], h: usize, q: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 only is needed here, but one switch for all bodies keeps the CPU path a single fact per process
        if avx2_fma() {
            return unsafe { products_avx2(row, ebm, h, q) };
        }
    }
    products_body(row, ebm, h, q)
}

/// S[u, h] = sum over groups m of A, A = ln(max(sum_g e[u, m, g] * eb[m, g, h], 1e-300)) + am[u, m] + bm[m, h], and
/// with rel = (le1, le0[u, m]) also R[u, h] = sum_m logaddexp(A + le1, le0[u, m]): EpiRefine's summed_scores with
/// the exponentials e = exp(Lp - am) and eb = exp(log_prof - bm) supplied by numpy. Units run in blocks on threads,
/// groups in the outer loop of a block so that a group's [G, H] profile table stays in cache; groups are added in
/// ascending order, so the result does not depend on the thread count.
#[allow(clippy::too_many_arguments)]
pub fn score_sums(e: &[f64], am: &[f64], u: usize, m: usize, g: usize, eb: &[f64], bm: &[f64], h: usize,
                  rel: Option<(f64, &[f64])>, threads: usize) -> (Vec<f64>, Vec<f64>) {
    const BLOCK: usize = 32;
    let mut s = vec![0.0f64; u * h];
    let mut r = vec![0.0f64; if rel.is_some() { u * h } else { 0 }];
    let work = |q: &mut Vec<f64>, (b, (sb, rb)): (usize, (&mut [f64], &mut [f64]))| {
        let u0 = b * BLOCK;
        let nu = sb.len() / h;
        for j in 0..m {
            let ebm = &eb[j * g * h..(j + 1) * g * h];
            let bmj = &bm[j * h..(j + 1) * h];
            for i in 0..nu {
                let row = &e[((u0 + i) * m + j) * g..][..g];
                products(row, ebm, h, q);
                let a_um = am[(u0 + i) * m + j];
                let so = &mut sb[i * h..(i + 1) * h];
                for t in 0..h {
                    q[t] = (q[t].max(1e-300).ln() + a_um) + bmj[t];
                    so[t] += q[t];
                }
                if let Some((le1, le0)) = rel {
                    let y = le0[(u0 + i) * m + j];
                    let ro = &mut rb[i * h..(i + 1) * h];
                    for t in 0..h {
                        ro[t] += logaddexp(q[t] + le1, y);
                    }
                }
            }
        }
    };
    let rchunk = if rel.is_some() { BLOCK * h } else { 1 };
    pool(threads).install(|| {
        if rel.is_some() {
            s.par_chunks_mut(BLOCK * h).zip(r.par_chunks_mut(rchunk)).enumerate().for_each_init(|| vec![0.0f64; h], work);
        } else {
            s.par_chunks_mut(BLOCK * h).enumerate().for_each_init(|| vec![0.0f64; h], |q, (b, sb)| work(q, (b, (sb, &mut []))));
        }
    });
    (s, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exp_fast_ulps() {
        let mut worst = 0i64;
        let mut x = -708.0f64;
        while x < 709.0 {
            let (a, b) = (exp_fast(x), x.exp());
            worst = worst.max((a.to_bits() as i64 - b.to_bits() as i64).abs());
            x += 0.000_731;
        }
        assert!(worst <= 2, "{worst} ulps");
        assert_eq!(exp_fast(0.0), 1.0);
        assert_eq!(exp_fast(-800.0), 0.0);
        assert_eq!(exp_fast(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp_fast(800.0), f64::INFINITY);
        assert!(exp_fast(f64::NAN).is_nan());
    }

    #[test]
    fn lse() {
        let a = [0.0, 0.0, -1.0, f64::NEG_INFINITY];
        let mut e = [0.0; 4];
        let r = logsumexp(&a, &mut e);
        assert!((r - (2.0f64 + (-1.0f64).exp()).ln()).abs() < 1e-15);
        assert_eq!(logsumexp(&[f64::NEG_INFINITY; 3], &mut e), f64::NEG_INFINITY);
        let v: Vec<f64> = (0..300).map(|i| i as f64 * 0.1).collect();
        assert!((pairwise_sum(&v) - v.iter().sum::<f64>()).abs() < 1e-9);
    }
}
