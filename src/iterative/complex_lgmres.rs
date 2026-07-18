//! ComplexLgmres — LGMRES (augmented GMRES) for complex linear systems.
//!
//! At each restart, augments the Krylov basis with approximate-error vectors
//! saved from prior cycles, enabling information to persist across restarts.
//!
//! **Algorithm** (Baker, Jessup & Manteuffel 2005):
//! aug = []  outer: r = b−Ax, v[0]=r/β
//!   Phase A: for a_i in aug: w=A·M⁻¹·a_i, GS(w)→Arnoldi
//!   Phase B: for j=0..m: z=M⁻¹·v[j], w=A·z, GS(w)→Arnoldi
//!   y=H⁻¹g; dx=Σy[j]·z[j]; x+=dx; aug.push(dx/‖dx‖)

use crate::core::{
    error::SolverError,
    operator::LinearOperator,
    preconditioner::Preconditioner,
    scalar::Scalar,
    vector::{DenseVec, Vector},
};
use num_complex::Complex;

#[inline]
fn to_f64<T: Scalar>(v: T) -> f64 {
    <f64 as num_traits::NumCast>::from(v).unwrap_or(0.0)
}

/// Reusable scratch workspace for [`ComplexLgmres`].
pub struct ComplexLgmresWorkspace<T: Scalar> {
    restart: usize,
    aug_dim: usize,
    r:      DenseVec<Complex<T>>,
    v:      Vec<DenseVec<Complex<T>>>,
    z:      DenseVec<Complex<T>>,
    w:      DenseVec<Complex<T>>,
    h:      Vec<Vec<Complex<T>>>,
    cs:     Vec<T>,
    sn:     Vec<Complex<T>>,
    g:      Vec<Complex<T>>,
}

impl<T: Scalar> ComplexLgmresWorkspace<T> {
    fn new(n: usize, restart: usize, aug_dim: usize) -> Self {
        let m = restart.max(1);
        let k = aug_dim.min(20);
        let max_basis = m + k;
        let zc = Complex::new(T::zero(), T::zero());
        ComplexLgmresWorkspace {
            restart: m, aug_dim: k,
            r: vec![zc; n].into(),
            v: (0..=max_basis).map(|_| vec![zc; n].into()).collect(),
            z: vec![zc; n].into(), w: vec![zc; n].into(),
            h: (0..max_basis).map(|_| vec![zc; max_basis + 1]).collect(),
            cs: vec![T::zero(); max_basis],
            sn: vec![zc; max_basis],
            g: vec![zc; max_basis + 1],
        }
    }
    fn ensure(&mut self, n: usize, restart: usize, aug_dim: usize) {
        let m = restart.max(1); let k = aug_dim.min(20);
        if self.restart != m || self.aug_dim != k || self.r.len() != n {
            *self = Self::new(n, m, k);
        }
    }
}

/// Complex LGMRES(m, k) solver with restart and augmentation.
pub struct ComplexLgmres<T: Scalar> {
    pub restart: usize,
    pub aug_dim: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Scalar> ComplexLgmres<T> {
    pub fn new(restart: usize, aug_dim: usize) -> Self {
        ComplexLgmres {
            restart: restart.max(1),
            aug_dim: aug_dim.min(20),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Solve `A x = b` using complex LGMRES(m, k).
    pub fn solve(
        &self,
        op: &dyn LinearOperator<Vector = DenseVec<Complex<T>>>,
        precond: Option<&dyn Preconditioner<Vector = DenseVec<Complex<T>>>>,
        b: &DenseVec<Complex<T>>,
        x: &mut DenseVec<Complex<T>>,
        rtol: f64,
        atol: f64,
        max_iter: usize,
    ) -> Result<super::complex_gmres::ComplexGmresResult, SolverError> {
        let n = b.len();
        if op.nrows() != n || op.ncols() != n || x.len() != n {
            return Err(SolverError::DimensionMismatch {
                op_rows: op.nrows(), op_cols: op.ncols(), rhs_len: n,
            });
        }
        let m = self.restart;
        let k = self.aug_dim;
        let max_basis = m + k;
        let mut ws = ComplexLgmresWorkspace::new(n, m, k);
        let zc = Complex::new(T::zero(), T::zero());

        let norm_b = b.norm2();
        if norm_b < T::machine_epsilon() {
            x.fill(zc);
            return Ok(super::complex_gmres::ComplexGmresResult {
                iters: 0, residual_norm: 0.0, converged: true, residual_history: vec![0.0],
            });
        }

        let tol = T::from_f64(f64::max(rtol * to_f64(norm_b), atol));
        let mut residual_history: Vec<f64> = Vec::new();
        let mut total_iters = 0usize;
        let mut aug_vecs: Vec<DenseVec<Complex<T>>> = Vec::with_capacity(k);

        'outer: loop {
            // r = b - A x
            op.apply(x, &mut ws.r);
            let r_sl = ws.r.as_mut_slice();
            let b_sl = b.as_slice();
            for i in 0..n { r_sl[i] = b_sl[i] - r_sl[i]; }
            let beta = ws.r.norm2();
            if beta <= tol || total_iters >= max_iter {
                return Ok(super::complex_gmres::ComplexGmresResult {
                    iters: total_iters, residual_norm: to_f64(beta),
                    converged: beta <= tol, residual_history,
                });
            }

            let inv_beta = Complex::new(T::one() / beta, T::zero());
            ws.v[0].copy_from(&ws.r); ws.v[0].scale(inv_beta);
            ws.g.fill(zc); ws.g[0] = Complex::new(beta, T::zero());
            for c in ws.cs.iter_mut() { *c = T::zero(); }
            for s in ws.sn.iter_mut() { *s = zc; }

            let aug_count = aug_vecs.len().min(k);
            let total_steps = (m + aug_count).min(max_basis);
            let mut j_final = 0usize;
            let mut inner_done = false;

            // Phase A: augmentation vectors
            for ai in 0..aug_count {
                if total_iters >= max_iter { break; }
                let j = ai;
                apply_pc(precond, &aug_vecs[ai], &mut ws.z);
                op.apply(&ws.z, &mut ws.w);
                for i in 0..=j {
                    let h_ij = ws.v[i].dot(&ws.w);
                    ws.h[j][i] = h_ij;
                    let vi = ws.v[i].clone();
                    ws.w.axpy(-h_ij, &vi);
                }
                let h_next = ws.w.norm2();
                apply_givens_seq(&mut ws.h, &ws.cs, &ws.sn, j);
                let (c_j, s_j) = crate::iterative::complex_gmres::complex_givens(
                    ws.h[j][j], Complex::new(h_next, T::zero()));
                ws.cs[j] = c_j; ws.sn[j] = s_j;
                ws.h[j][j] = Complex::new(c_j, T::zero()) * ws.h[j][j]
                    + s_j * Complex::new(h_next, T::zero());
                let gj = ws.g[j];
                ws.g[j] = Complex::new(c_j, T::zero()) * gj;
                ws.g[j + 1] = -s_j.conj() * gj;
                total_iters += 1; j_final = j + 1;
                if ws.g[j + 1].norm() <= tol { inner_done = true; break; }
                if h_next > T::machine_epsilon() && j_final < total_steps {
                    let inv_h = Complex::new(T::one() / h_next, T::zero());
                    let wc = ws.w.clone(); ws.v[j + 1].copy_from(&wc); ws.v[j + 1].scale(inv_h);
                } else { break; }
            }

            // Phase B: standard Arnoldi
            if !inner_done {
                for j in aug_count..total_steps {
                    if total_iters >= max_iter { break; }
                    let vj = ws.v[j].clone();
                    apply_pc(precond, &vj, &mut ws.z);
                    op.apply(&ws.z, &mut ws.w);
                    for i in 0..=j {
                        let h_ij = ws.v[i].dot(&ws.w);
                        ws.h[j][i] = h_ij;
                        let vi = ws.v[i].clone();
                        ws.w.axpy(-h_ij, &vi);
                    }
                    let h_next = ws.w.norm2();
                    apply_givens_seq(&mut ws.h, &ws.cs, &ws.sn, j);
                    let (c_j, s_j) = crate::iterative::complex_gmres::complex_givens(
                        ws.h[j][j], Complex::new(h_next, T::zero()));
                    ws.cs[j] = c_j; ws.sn[j] = s_j;
                    ws.h[j][j] = Complex::new(c_j, T::zero()) * ws.h[j][j]
                        + s_j * Complex::new(h_next, T::zero());
                    let gj = ws.g[j];
                    ws.g[j] = Complex::new(c_j, T::zero()) * gj;
                    ws.g[j + 1] = -s_j.conj() * gj;
                    total_iters += 1; j_final = j + 1;
                    if ws.g[j + 1].norm() <= tol { inner_done = true; break; }
                    if h_next > T::machine_epsilon() && j + 1 < total_steps {
                        let inv_h = Complex::new(T::one() / h_next, T::zero());
                        let wc = ws.w.clone(); ws.v[j + 1].copy_from(&wc); ws.v[j + 1].scale(inv_h);
                    } else { break; }
                }
            }

            // Back-substitute & update
            if j_final > 0 {
                do_update(x, &ws.v, &mut ws.z, &ws.h, &ws.g, &mut ws.r,
                    &mut aug_vecs, k, j_final, n, precond,
                    &mut residual_history);
            }
            if total_iters >= max_iter { break; }
        }

        Err(SolverError::ConvergenceFailed {
            max_iter,
            residual: residual_history.last().copied().unwrap_or(f64::INFINITY),
        })
    }
}

fn do_update<T: Scalar>(
    x: &mut DenseVec<Complex<T>>,
    v: &[DenseVec<Complex<T>>],
    z: &mut DenseVec<Complex<T>>,
    h: &[Vec<Complex<T>>],
    g: &[Complex<T>],
    _r: &mut DenseVec<Complex<T>>,
    aug_vecs: &mut Vec<DenseVec<Complex<T>>>,
    k: usize,
    jf: usize,
    n: usize,
    precond: Option<&dyn Preconditioner<Vector = DenseVec<Complex<T>>>>,
    residual_history: &mut Vec<f64>,
) {
    // Back-substitution
    let mut y = vec![Complex::new(T::zero(), T::zero()); jf];
    for i in (0..jf).rev() {
        let mut s = g[i];
        for kk in (i + 1)..jf { s -= h[kk][i] * y[kk]; }
        y[i] = if h[i][i].norm() > T::machine_epsilon() { s / h[i][i] }
               else { Complex::new(T::zero(), T::zero()) };
    }

    // dx = Σ y[j] · z[j]
    let mut dx = DenseVec::zeros(n);
    for j in 0..jf {
        apply_pc(precond, &v[j], z);
        dx.axpy(y[j], z);
    }

    // x += dx
    let xs = x.as_mut_slice();
    let dxs = dx.as_slice();
    for i in 0..n { xs[i] += dxs[i]; }

    // Residual estimate
    residual_history.push(to_f64(g[jf - 1].norm()));

    // Save augmentation vector
    let dx_norm = dx.norm2();
    if dx_norm > T::machine_epsilon() {
        dx.scale(Complex::new(T::one() / dx_norm, T::zero()));
        if aug_vecs.len() >= k && k > 0 { aug_vecs.remove(0); }
        if k > 0 { aug_vecs.push(dx); }
    }
}

fn apply_givens_seq<T: Scalar>(h: &mut [Vec<Complex<T>>], cs: &[T], sn: &[Complex<T>], j: usize) {
    let hj = &mut h[j];
    for i in 0..j {
        let tmp = Complex::new(cs[i], T::zero()) * hj[i] + sn[i] * hj[i + 1];
        hj[i + 1] = -sn[i].conj() * hj[i] + Complex::new(cs[i], T::zero()) * hj[i + 1];
        hj[i] = tmp;
    }
}

fn apply_pc<T: Scalar>(
    precond: Option<&dyn Preconditioner<Vector = DenseVec<Complex<T>>>>,
    src: &DenseVec<Complex<T>>,
    dst: &mut DenseVec<Complex<T>>,
) {
    match precond { Some(m) => m.apply_precond(src, dst), None => dst.copy_from(src), }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dense::DenseMatrix;

    #[test]
    fn lgmres_small_diag_dominant() {
        let n = 8;
        let a = DenseMatrix::from_fn(n, n, |i, j| {
            if i == j { Complex::new(10.0 + i as f64, 1.0) }
            else if (i as i32 - j as i32).abs() == 1 { Complex::new(-1.0, 0.1) }
            else { Complex::new(0.0, 0.0) }
        });
        let b: DenseVec<Complex<f64>> = (0..n).map(|i| Complex::new((i+1) as f64, 0.0)).collect::<Vec<_>>().into();
        let mut x: DenseVec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n].into();
        let solver = ComplexLgmres::<f64>::new(8, 3);
        let r = solver.solve(&a, None, &b, &mut x, 1e-10, 0.0, 500).unwrap();
        assert!(r.converged, "ComplexLgmres should converge on diag-dominant system");
        // Verify: A·x ≈ b
        let mut ax = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            for j in 0..n {
                ax[i] += a[(i, j)] * x[j];
            }
        }
        for i in 0..n {
            let err = (ax[i] - b[i]).norm();
            assert!(err < 1e-8, "row {i}: |Ax-b| = {err:.2e}");
        }
    }
}
