//! The compiled kernels of EpiRefine: the pileup reader, the motif scan and the model kernels, built by maturin into
//! the extension `epirefine._kernels._core` of the platform wheels (D73). The Python package `epirefine._kernels`
//! wraps this module; see its docstrings for the interface. EpiRefine's numpy code is the reference and fallback.
mod error;
mod input;
mod model;
mod motif;
mod pileup;
mod sites;

use error::Error;
use motif::{MotifDef, Records, Scanner};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pileup::{Provider, State};
use pyo3::exceptions::{PyOSError, PyOverflowError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

impl From<Error> for PyErr {
    fn from(e: Error) -> PyErr {
        match e {
            Error::Io(s) => PyOSError::new_err(s),
            Error::Value(s) => PyValueError::new_err(s),
            Error::Overflow(s) => PyOverflowError::new_err(s),
        }
    }
}

fn motif_defs(motifs: Vec<(String, i64, String)>) -> Vec<MotifDef> {
    motifs.into_iter().map(|(seq, mod_pos, mod_type)| MotifDef { seq: seq.into_bytes(), mod_pos, mod_type }).collect()
}

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{what} must be a contiguous array")))
}

type RecordArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
);

fn record_arrays(py: Python<'_>, r: Records) -> RecordArrays<'_> {
    (r.pos.into_pyarray_bound(py), r.strand.into_pyarray_bound(py), r.motif.into_pyarray_bound(py),
     r.occ_motif.into_pyarray_bound(py), r.occ.into_pyarray_bound(py))
}

/// Motif matcher: the records of `epirefine.motifs.MotifMatcher`, in the same order.
#[pyclass(module = "epirefine._kernels._core")]
struct MotifScanner {
    scanner: Scanner,
}

#[pymethods]
impl MotifScanner {
    #[new]
    fn new(motifs: Vec<(String, i64, String)>) -> PyResult<Self> {
        Ok(MotifScanner { scanner: Scanner::new(&motif_defs(motifs))? })
    }

    #[getter]
    fn n_shapes(&self) -> usize {
        self.scanner.shapes.len()
    }

    /// (pos, strand, motif, occ_motif, occ) of every record in one encoded sequence.
    fn find<'py>(&self, py: Python<'py>, codes: PyReadonlyArray1<'py, u8>) -> PyResult<RecordArrays<'py>> {
        let codes = slice(&codes, "codes")?;
        let sc = &self.scanner;
        let rec = py.allow_threads(|| {
            let mut r = Records::default();
            sc.find(codes, &mut r);
            r
        });
        Ok(record_arrays(py, rec))
    }

    /// (index into contig_ids, (pos, strand, motif, occ_motif, occ)) of the records of the given
    /// contigs, in contig coordinates.
    fn find_contigs<'py>(&self, py: Python<'py>, codes: PyReadonlyArray1<'py, u8>, offsets: PyReadonlyArray1<'py, i64>,
                         lengths: PyReadonlyArray1<'py, i64>, contig_ids: PyReadonlyArray1<'py, i64>)
                         -> PyResult<(Bound<'py, PyArray1<i64>>, RecordArrays<'py>)> {
        let codes = slice(&codes, "codes")?;
        let seqs = contig_slices(codes, slice(&offsets, "offsets")?, slice(&lengths, "lengths")?,
                                 slice(&contig_ids, "contig_ids")?)?;
        let sc = &self.scanner;
        let (cidx, rec) = py.allow_threads(|| {
            let (mut c, mut r) = (Vec::new(), Records::default());
            sc.find_contigs(&seqs, &mut c, &mut r);
            (c, r)
        });
        Ok((cidx.into_pyarray_bound(py), record_arrays(py, rec)))
    }
}

fn contig_slices<'a>(codes: &'a [u8], offsets: &[i64], lengths: &[i64], ids: &[i64]) -> PyResult<Vec<&'a [u8]>> {
    if offsets.len() != lengths.len() {
        return Err(PyValueError::new_err("offsets and lengths must have equal lengths"));
    }
    let mut out = Vec::with_capacity(ids.len());
    for &i in ids {
        if i < 0 || i as usize >= offsets.len() {
            return Err(PyValueError::new_err(format!("contig id {i} is out of range")));
        }
        let (o, l) = (offsets[i as usize], lengths[i as usize]);
        if o < 0 || l < 0 || (o + l) as usize > codes.len() {
            return Err(PyValueError::new_err(format!("contig {i} lies outside the encoded assembly")));
        }
        out.push(&codes[o as usize..(o + l) as usize]);
    }
    Ok(out)
}

/// Deduplicated site records of the given contigs, as `sites._Worker.records` builds them:
/// key = index into contig_ids << 36 | pos << 3 | strand << 2 | modification code, the group and the
/// site (index << 50 | canonical motif << 36 | occurrence); ascending by key.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn site_records<'py>(py: Python<'py>, codes: PyReadonlyArray1<'py, u8>, offsets: PyReadonlyArray1<'py, i64>,
                     lengths: PyReadonlyArray1<'py, i64>, contig_ids: PyReadonlyArray1<'py, i64>,
                     motifs: Vec<(String, i64, String)>, group_of: Vec<i64>, motif_rank: Vec<i64>)
                     -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>)> {
    let codes = slice(&codes, "codes")?;
    let seqs = contig_slices(codes, slice(&offsets, "offsets")?, slice(&lengths, "lengths")?,
                             slice(&contig_ids, "contig_ids")?)?;
    let mut builder = sites::SiteBuilder::new(&motif_defs(motifs), &group_of, &motif_rank)?;
    let (key, group, site) = py.allow_threads(|| {
        let (mut key, mut group, mut site) = (Vec::new(), Vec::new(), Vec::new());
        let mut cs = sites::ContigSites::default();
        for (ci, seq) in seqs.iter().enumerate() {
            builder.build(seq, &mut cs);
            let ci = ci as i64;
            for j in 0..cs.keys.len() {
                key.push((ci << 36) | cs.keys[j] as i64);
                group.push(cs.group[j]);
                site.push((ci << 50) | cs.site_ids[cs.site[j] as usize] as i64);
            }
        }
        (key, group, site)
    });
    Ok((key.into_pyarray_bound(py), group.into_pyarray_bound(py), site.into_pyarray_bound(py)))
}

type SiteArrays<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
);

struct Inner {
    src: input::Source,
    st: State,
    finished: bool,
    failed: bool,
    read_seconds: f64,
}

impl Inner {
    /// Reads on until at least max_rows site rows are buffered at a contig boundary, or the range ends.
    fn advance(&mut self, codes: &[u8], max_rows: usize) -> error::Result<()> {
        while !self.finished && self.st.out.contig.len() < max_rows {
            let t = std::time::Instant::now();
            let chunk = self.src.next_chunk()?;
            self.read_seconds += t.elapsed().as_secs_f64();
            match chunk {
                Some(data) => self.st.feed(data, codes)?,
                None => {
                    self.st.finish(codes)?;
                    self.finished = true;
                }
            }
        }
        Ok(())
    }
}

/// One range of a pileup -> site rows (contig, group, pos, n, k), in batches of whole contigs.
#[pyclass(module = "epirefine._kernels._core")]
struct PileupReader {
    inner: Inner,
    codes: Option<Py<PyArray1<u8>>>,
}

fn names_to_py(names: &[Vec<u8>]) -> PyResult<Vec<String>> {
    names.iter()
        .map(|n| String::from_utf8(n.clone()).map_err(|_| PyValueError::new_err("pileup: a contig name is not valid UTF-8")))
        .collect()
}

#[pymethods]
impl PileupReader {
    /// Motif matching in Rust: the contigs' site keys are computed from the encoded assembly when
    /// the contig's first row is read.
    #[new]
    #[pyo3(signature = (path, start, end, names, offsets, lengths, codes, motifs, group_of, motif_rank, wanted=None, check_eof=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(path: &str, start: i64, end: i64, names: Vec<String>, offsets: PyReadonlyArray1<'_, i64>,
           lengths: PyReadonlyArray1<'_, i64>, codes: Bound<'_, PyArray1<u8>>, motifs: Vec<(String, i64, String)>,
           group_of: Vec<i64>, motif_rank: Vec<i64>, wanted: Option<Vec<i64>>, check_eof: bool) -> PyResult<Self> {
        let builder = sites::SiteBuilder::new(&motif_defs(motifs), &group_of, &motif_rank)?;
        if codes.readonly().as_slice().is_err() {
            return Err(PyValueError::new_err("codes must be a contiguous array"));
        }
        let provider = Provider::Scan {
            builder: Box::new(builder),
            offsets: slice(&offsets, "offsets")?.to_vec(),
            lengths: slice(&lengths, "lengths")?.to_vec(),
        };
        let st = State::new(&names, wanted.as_deref(), provider)?;
        let src = input::Source::open(path, start, end, check_eof)?;
        Ok(PileupReader { inner: Inner { src, st, finished: false, failed: false, read_seconds: 0.0 }, codes: Some(codes.unbind()) })
    }

    /// Motif matching by the caller: keys (pos << 3 | strand << 2 | modification code, ascending
    /// within a contig) with their group and site id, as CSR over the contigs of `names`.
    #[staticmethod]
    #[pyo3(signature = (path, start, end, names, key_ptr, keys, key_group, key_site, check_eof=true))]
    #[allow(clippy::too_many_arguments)]
    fn with_keys(path: &str, start: i64, end: i64, names: Vec<String>, key_ptr: PyReadonlyArray1<'_, i64>,
                 keys: PyReadonlyArray1<'_, i64>, key_group: PyReadonlyArray1<'_, i32>,
                 key_site: PyReadonlyArray1<'_, i64>, check_eof: bool) -> PyResult<Self> {
        let keys = slice(&keys, "keys")?;
        if keys.iter().any(|&k| k < 0 || k >= 1 << 36) {
            return Err(PyValueError::new_err("keys must be within a contig: 0 <= key < 2**36"));
        }
        let provider = Provider::Keys {
            ptr: slice(&key_ptr, "key_ptr")?.to_vec(),
            keys: keys.iter().map(|&k| k as u64).collect(),
            group: slice(&key_group, "key_group")?.to_vec(),
            site: slice(&key_site, "key_site")?.iter().map(|&s| s as u64).collect(),
        };
        let st = State::new(&names, None, provider)?;
        let src = input::Source::open(path, start, end, check_eof)?;
        Ok(PileupReader { inner: Inner { src, st, finished: false, failed: false, read_seconds: 0.0 }, codes: None })
    }

    /// The next site rows (contig, group, pos, n, k): whole contigs, at least `max_rows` rows unless
    /// the range ends first. None when the range is exhausted. The GIL is released while reading.
    fn next_batch<'py>(&mut self, py: Python<'py>, max_rows: usize) -> PyResult<Option<SiteArrays<'py>>> {
        if self.inner.failed {
            return Err(PyRuntimeError::new_err("the reader failed earlier; its rows are incomplete"));
        }
        let inner = &mut self.inner;
        let res = match &self.codes {
            Some(c) => {
                let ro = c.bind(py).readonly();
                let codes = ro.as_slice().map_err(|_| PyValueError::new_err("codes must be a contiguous array"))?;
                py.allow_threads(|| inner.advance(codes, max_rows.max(1)))
            }
            None => py.allow_threads(|| inner.advance(&[], max_rows.max(1))),
        };
        if let Err(e) = res {
            inner.failed = true;
            inner.st.out = pileup::Out::default();
            return Err(e.into());
        }
        if inner.st.out.contig.is_empty() {
            return Ok(None);
        }
        let o = std::mem::take(&mut inner.st.out);
        Ok(Some((o.contig.into_pyarray_bound(py), o.group.into_pyarray_bound(py), o.pos.into_pyarray_bound(py),
                 o.n.into_pyarray_bound(py), o.k.into_pyarray_bound(py))))
    }

    /// True once the whole range has been read.
    #[getter]
    fn finished(&self) -> bool {
        self.inner.finished
    }

    /// Names of the contigs with rows read so far, in file order.
    fn contigs(&self) -> PyResult<Vec<String>> {
        names_to_py(&self.inner.st.order)
    }

    /// Counters: text bytes, lines, rows that hit a site, contigs with keys built, keys, seconds
    /// reading/inflating and building keys.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        let s = &self.inner.st.stats;
        d.set_item("input", self.inner.src.kind())?;
        d.set_item("text_bytes", self.inner.src.text_bytes())?;
        d.set_item("lines", s.lines)?;
        d.set_item("rows_hit", s.rows_hit)?;
        d.set_item("contigs_keyed", s.contigs_scanned)?;
        d.set_item("keys", s.keys)?;
        d.set_item("read_seconds", self.inner.read_seconds)?;
        d.set_item("key_seconds", s.scan_seconds)?;
        Ok(d)
    }
}

fn table<'a>(lp: &'a PyReadonlyArray3<'_, f32>) -> PyResult<(&'a [f32], usize, usize, usize)> {
    let sh = lp.shape();
    let (u, m, g) = (sh[0], sh[1], sh[2]);
    let s = lp.as_slice().map_err(|_| PyValueError::new_err("the table must be a C-contiguous float32 array"))?;
    Ok((s, u, m, g))
}

fn check(ok: bool, what: &str) -> PyResult<()> {
    if ok { Ok(()) } else { Err(PyValueError::new_err(format!("shape mismatch: {what}"))) }
}

/// A grid of no points has no meaning (and would divide by zero below): a ValueError, not a panic. pyo3 turns a
/// panic into PanicException, a BaseException that `except Exception` does not catch.
fn no_empty_grid(g: usize) -> PyResult<()> {
    if g == 0 { Err(PyValueError::new_err("the methylation grid has no points (G = 0)")) } else { Ok(()) }
}

/// ell[h, m, g] = sum over the chunks of bin h of w * lp (float64, flat [H * M * G]).
#[pyfunction]
fn bin_loglik<'py>(py: Python<'py>, lp: PyReadonlyArray3<'py, f32>, chunk_bin: PyReadonlyArray1<'py, i64>,
                   w: PyReadonlyArray1<'py, f64>, n_bins: usize, threads: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (t, u, m, g) = table(&lp)?;
    let (cb, w) = (slice(&chunk_bin, "chunk_bin")?, slice(&w, "w")?);
    check(cb.len() == u && w.len() == u, "chunk_bin, w")?;
    if cb.iter().any(|&b| b < 0 || b as usize >= n_bins) {
        return Err(PyValueError::new_err("chunk_bin out of range"));
    }
    if m * g == 0 {  // no groups or no grid: nothing to add (as numpy's zeros)
        return Ok(Vec::<f64>::new().into_pyarray_bound(py));
    }
    Ok(py.allow_threads(|| model::bin_loglik(t, m * g, cb, w, n_bins, threads)).into_pyarray_bound(py))
}

/// Per unit: sum over groups of logsumexp(lp + log_pi).
#[pyfunction]
fn community_terms<'py>(py: Python<'py>, lp: PyReadonlyArray3<'py, f32>, log_pi: PyReadonlyArray2<'py, f64>,
                        threads: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (t, u, m, g) = table(&lp)?;
    let pi = log_pi.as_slice().map_err(|_| PyValueError::new_err("log_pi must be contiguous"))?;
    check(pi.len() == m * g, "log_pi")?;
    if m == 0 {  // a sum over no groups
        return Ok(vec![0.0f64; u].into_pyarray_bound(py));
    }
    no_empty_grid(g)?;
    Ok(py.allow_threads(|| model::community_terms(t, m, g, pi, threads)).into_pyarray_bound(py))
}

/// Per chunk: sum over groups of logsumexp(lp + profile of its bin without the chunk).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn leave_chunk_out_terms<'py>(py: Python<'py>, lp: PyReadonlyArray3<'py, f32>, chunk_bin: PyReadonlyArray1<'py, i64>,
                              w: PyReadonlyArray1<'py, f64>, ell: PyReadonlyArray3<'py, f64>,
                              log_pi: PyReadonlyArray2<'py, f64>, threads: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (t, u, m, g) = table(&lp)?;
    let (cb, w) = (slice(&chunk_bin, "chunk_bin")?, slice(&w, "w")?);
    let el = ell.as_slice().map_err(|_| PyValueError::new_err("ell must be contiguous"))?;
    let pi = log_pi.as_slice().map_err(|_| PyValueError::new_err("log_pi must be contiguous"))?;
    check(cb.len() == u && w.len() == u, "chunk_bin, w")?;
    if m == 0 {  // a sum over no groups
        check(pi.is_empty() && el.is_empty(), "ell, log_pi")?;
        return Ok(vec![0.0f64; u].into_pyarray_bound(py));
    }
    no_empty_grid(g)?;
    check(pi.len() == m * g && el.len() % (m * g).max(1) == 0, "ell, log_pi")?;
    let h = el.len() / (m * g).max(1);
    if cb.iter().any(|&b| b < 0 || b as usize >= h) {
        return Err(PyValueError::new_err("chunk_bin out of range"));
    }
    Ok(py.allow_threads(|| model::leave_chunk_out_terms(t, m, g, cb, w, el, pi, threads)).into_pyarray_bound(py))
}

/// Grid NPMLE (EM) of pi[m, g] from lik[m, b, g], started at pi; flat [M * G].
#[pyfunction]
fn npmle<'py>(py: Python<'py>, lik: PyReadonlyArray3<'py, f64>, pi: PyReadonlyArray2<'py, f64>, iters: usize, tol: f64,
              threads: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let sh = lik.shape();
    let (m, b, g) = (sh[0], sh[1], sh[2]);
    let lk = lik.as_slice().map_err(|_| PyValueError::new_err("lik must be contiguous"))?;
    let p0 = pi.as_slice().map_err(|_| PyValueError::new_err("pi must be contiguous"))?;
    check(p0.len() == m * g, "pi")?;
    Ok(py.allow_threads(|| model::npmle(lk, m, b, g, p0, iters, tol, threads)).into_pyarray_bound(py))
}

/// Per cell: the sum over its rows of w * table[r] (flat [cells * G]).
#[pyfunction]
fn cell_sums<'py>(py: Python<'py>, table: PyReadonlyArray2<'py, f64>, r: PyReadonlyArray1<'py, i64>,
                  w: PyReadonlyArray1<'py, f64>, starts: PyReadonlyArray1<'py, i64>, threads: usize)
                  -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (p, g) = (table.shape()[0], table.shape()[1]);
    let t = table.as_slice().map_err(|_| PyValueError::new_err("table must be contiguous"))?;
    let (r, w, st) = (slice(&r, "r")?, slice(&w, "w")?, slice(&starts, "starts")?);
    check(r.len() == w.len(), "r, w")?;
    if g == 0 {  // cells of no columns
        return Ok(Vec::<f64>::new().into_pyarray_bound(py));
    }
    if r.iter().any(|&x| x < 0 || x as usize >= p) {
        return Err(PyValueError::new_err("row index out of range"));
    }
    let n = r.len() as i64;
    if st.iter().any(|&x| x < 0 || x >= n) || st.windows(2).any(|v| v[1] <= v[0]) {
        return Err(PyValueError::new_err("starts must be ascending row numbers of non-empty cells"));
    }
    Ok(py.allow_threads(|| model::cell_sums(t, g, r, w, st, threads)).into_pyarray_bound(py))
}

/// Sums over groups of the per-group scores of units against bins (S) and, with le0, against relatives (R).
#[pyfunction]
#[pyo3(signature = (e, am, eb, bm, threads, le1=0.0, le0=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn score_sums<'py>(py: Python<'py>, e: PyReadonlyArray3<'py, f64>, am: PyReadonlyArray2<'py, f64>,
                   eb: PyReadonlyArray3<'py, f64>, bm: PyReadonlyArray2<'py, f64>, threads: usize, le1: f64,
                   le0: Option<PyReadonlyArray2<'py, f64>>)
                   -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (u, m, g) = (e.shape()[0], e.shape()[1], e.shape()[2]);
    let h = eb.shape()[2];
    let err = |_| PyValueError::new_err("arrays must be C-contiguous float64");
    let (es, ams, ebs, bms) = (e.as_slice().map_err(err)?, am.as_slice().map_err(err)?, eb.as_slice().map_err(err)?,
                               bm.as_slice().map_err(err)?);
    check(ams.len() == u * m && eb.shape()[0] == m && eb.shape()[1] == g && bms.len() == m * h, "am, eb, bm")?;
    let l0 = match &le0 {
        Some(a) => {
            let sl = a.as_slice().map_err(err)?;
            check(sl.len() == u * m, "le0")?;
            Some(sl)
        }
        None => None,
    };
    if h == 0 {  // scores against no bins
        return Ok((Vec::<f64>::new().into_pyarray_bound(py), Vec::<f64>::new().into_pyarray_bound(py)));
    }
    let (s, r) = py.allow_threads(|| model::score_sums(es, ams, u, m, g, ebs, bms, h, l0.map(|x| (le1, x)), threads));
    Ok((s.into_pyarray_bound(py), r.into_pyarray_bound(py)))
}

/// Names of the features this build provides.
#[pyfunction]
fn features() -> Vec<&'static str> {
    let mut f = vec!["pileup_reader", "pileup_reader_keys", "motif_scan", "site_records", "plain", "bgzf", "gzip", "model_kernels", "cell_sums", "score_sums"];
    if cfg!(feature = "libdeflate") {
        f.push("libdeflate");
    }
    // the CPU path of the model kernels in this process ("cpu:avx2_fma" or "cpu:generic"), so that run records show it
    f.push(if model::avx2_fma() { "cpu:avx2_fma" } else { "cpu:generic" });
    f
}

/// The CPU path of the model kernels in this process: "avx2_fma" or "generic" (EPIREFINE_RS_NO_SIMD=1 forces it).
#[pyfunction]
fn cpu_path() -> &'static str {
    model::cpu_path()
}

/// Interface version of this extension (see the module init).
const API: u32 = 1;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // the commit the build came from (build.rs); None for a build from an sdist without its commit
    m.add("__commit__", option_env!("EPIREFINE_RS_COMMIT"))?;
    // the interface version of the functions below, checked by epirefine._rs against the version its Python code
    // was written for: raised when a function changes meaning or signature, so that a stale build (an editable
    // install that was not rebuilt) is refused instead of run
    m.add("API", API)?;
    m.add_class::<MotifScanner>()?;
    m.add_class::<PileupReader>()?;
    m.add_function(wrap_pyfunction!(site_records, m)?)?;
    m.add_function(wrap_pyfunction!(features, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_path, m)?)?;
    m.add_function(wrap_pyfunction!(bin_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(score_sums, m)?)?;
    m.add_function(wrap_pyfunction!(cell_sums, m)?)?;
    m.add_function(wrap_pyfunction!(npmle, m)?)?;
    m.add_function(wrap_pyfunction!(community_terms, m)?)?;
    m.add_function(wrap_pyfunction!(leave_chunk_out_terms, m)?)?;
    Ok(())
}
