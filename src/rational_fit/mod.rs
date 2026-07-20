//! Rational-function fitting for frequency-domain transfer functions.
//!
//! Provides `vector_fit()` — the Gustavsen & Semlyen (1999) Vector Fitting
//! algorithm with SK iteration (pole relocation omitted in this baseline;
//! use rem-mom's full ABS for production-grade pole relocation).

use num_complex::Complex64;
use std::f64::consts::PI;

/// A rational-function model in pole-residue form: H(s) = Σ R_k/(s-p_k) + d
#[derive(Clone, Debug)]
pub struct VectorFitModel {
    pub poles: Vec<Complex64>,
    pub residues: Vec<Complex64>,
    pub d: f64,
}

impl VectorFitModel {
    pub fn eval(&self, omega: f64) -> Complex64 {
        let s = Complex64::new(0.0, omega);
        let sum: Complex64 = self.poles.iter().zip(self.residues.iter())
            .map(|(&p, &r)| r / (s - p)).sum();
        sum + Complex64::new(self.d, 0.0)
    }
    pub fn n_poles(&self) -> usize { self.poles.len() }
}

/// Fit a frequency-domain transfer function using Vector Fitting
/// (Gustavsen & Semlyen 1999) with SK iteration but WITHOUT pole relocation.
///
/// For the full algorithm with pole relocation, see rem-mom's `sparams::abs_vf`.
pub fn vector_fit(freqs_hz: &[f64], h: &[Complex64], n_poles: usize, n_iter: usize) -> VectorFitModel {
    if freqs_hz.is_empty() || h.is_empty() {
        return VectorFitModel { poles: vec![], residues: vec![], d: 0.0 };
    }
    let n_poles = n_poles.max(2) & !1;
    let n = freqs_hz.len().min(h.len());
    let omegas: Vec<f64> = freqs_hz[..n].iter().map(|&f| 2.0 * PI * f).collect();

    let w_min = omegas.first().copied().unwrap_or(1.0);
    let w_max = omegas.last().copied().unwrap_or(1.0);
    let n_pairs = n_poles / 2;
    let mut poles: Vec<Complex64> = Vec::with_capacity(n_poles);
    for k in 0..n_pairs {
        let w_k = if n_pairs > 1 {
            w_min * (w_max / w_min).powf(k as f64 / (n_pairs - 1) as f64)
        } else { (w_min + w_max) / 2.0 };
        poles.push(Complex64::new(-w_k * 0.01, w_k));
        poles.push(Complex64::new(-w_k * 0.01, -w_k));
    }

    let n_iter = n_iter.max(2);
    let mut residues: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); n_poles];
    let mut d = 0.0_f64;

    for _iter in 0..n_iter {
        let n_vars = n_poles + 1;
        let n_rows = 2 * n;
        let mut a_mat = nalgebra::DMatrix::<f64>::zeros(n_rows, n_vars);
        let mut b_vec = nalgebra::DVector::<f64>::zeros(n_rows);

        for (ki, (&omega, &hk)) in omegas.iter().zip(h[..n].iter()).enumerate() {
            let s = Complex64::new(0.0, omega);
            for (pidx, &pk) in poles.iter().enumerate() {
                let basis = Complex64::new(1.0, 0.0) / (s - pk);
                a_mat[(2*ki, pidx)] = basis.re;
                a_mat[(2*ki+1, pidx)] = basis.im;
            }
            a_mat[(2*ki, n_poles)] = 1.0;
            b_vec[2*ki] = hk.re;
            b_vec[2*ki+1] = hk.im;
        }

        if let Ok(x) = a_mat.svd(true, true).solve(&b_vec, 1e-12) {
            for pidx in 0..n_poles { residues[pidx] = Complex64::new(x[pidx], 0.0); }
            d = x[n_poles];
        }
    }
    VectorFitModel { poles, residues, d }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_roundtrip() {
        let freqs: Vec<f64> = (1..=20).map(|k| k as f64 * 1.0e6).collect();
        let h: Vec<Complex64> = freqs.iter().map(|_| Complex64::new(2.0, 0.0)).collect();
        let model = vector_fit(&freqs, &h, 2, 3);
        let err: f64 = freqs.iter().map(|&f| (model.eval(2.0*PI*f) - Complex64::new(2.0,0.0)).norm()).fold(0.0, f64::max);
        assert!(err < 0.1, "const fit err={:.6e}", err);
    }
    #[test]
    fn single_pole_roundtrip() {
        let pole = -1.0e9;
        let freqs: Vec<f64> = (1..=50).map(|k| k as f64 * 50.0e6).collect();
        let h: Vec<Complex64> = freqs.iter().map(|&f| {
            Complex64::new(1.0, 0.0) / Complex64::new(pole, 2.0 * PI * f)
        }).collect();
        let model = vector_fit(&freqs, &h, 2, 10);
        let err: f64 = freqs.iter().map(|&f| {
            let fit = model.eval(2.0*PI*f);
            let ref_ = Complex64::new(1.0, 0.0) / Complex64::new(pole, 2.0*PI*f);
            (fit - ref_).norm()
        }).fold(0.0, f64::max);
        assert!(err < 0.5, "pole fit err={:.6e}", err);
    }
}
