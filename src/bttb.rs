use num_complex::Complex64;

/// BTTB matrix-vector product via 2D FFT with 4-quadrant embedding.
pub fn bttb_matvec(kernel: &[Complex64], x: &[Complex64], nx: usize, ny: usize) -> Vec<Complex64> {
    let szx = 2 * nx;
    let szy = 2 * ny;
    let sz = szy * szx;
    let n = nx * ny;
    let mut ke = vec![Complex64::ZERO; sz];
    for iy in 0..ny { for ix in 0..nx {
        let v = kernel[iy * nx + ix];
        ke[iy * szx + ix] = v;
        if iy > 0 { ke[(szy - iy) * szx + ix] = v; }
        if ix > 0 { ke[iy * szx + (szx - ix)] = v; }
        if iy > 0 && ix > 0 { ke[(szy - iy) * szx + (szx - ix)] = v; }
    }}
    let mut xe = vec![Complex64::ZERO; sz];
    for iy in 0..ny { for ix in 0..nx {
        let v = x[iy * nx + ix];
        xe[iy * szx + ix] = v;
        if iy > 0 { xe[(szy - iy) * szx + ix] = v; }
        if ix > 0 { xe[iy * szx + (szx - ix)] = v; }
        if iy > 0 && ix > 0 { xe[(szy - iy) * szx + (szx - ix)] = v; }
    }}
    fft2d(&mut ke, szx, szy, false);
    fft2d(&mut xe, szx, szy, false);
    for i in 0..sz { xe[i] = ke[i] * xe[i]; }
    fft2d(&mut xe, szx, szy, true);
    let s = 1.0 / sz as f64;
    let mut y = vec![Complex64::ZERO; n];
    for iy in 0..ny { for ix in 0..nx { y[iy * nx + ix] = xe[iy * szx + ix] * s; }}
    y
}

fn fft(buf: &mut [Complex64], inv: bool) {
    let n = buf.len();
    if n <= 1 { return; }
    assert!(n.is_power_of_two());
    let b = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - b);
        if i < j { buf.swap(i, j); }
    }
    let mut l = 1;
    while l < n {
        let s = l * 2;
        let a = std::f64::consts::PI / l as f64;
        for i in (0..n).step_by(s) {
            for k in 0..l {
                let sn = (a * k as f64).sin();
            let c = (a * k as f64).cos();
                let w = if inv { Complex64::new(c, sn) } else { Complex64::new(c, -sn) };
                let u = buf[i + k];
                let v = w * buf[i + k + l];
                buf[i + k] = u + v;
                buf[i + k + l] = u - v;
            }
        }
        l = s;
    }
}

fn fft2d(buf: &mut [Complex64], nx: usize, ny: usize, inv: bool) {
    for iy in 0..ny { fft(&mut buf[iy * nx .. (iy + 1) * nx], inv); }
    let mut c = vec![Complex64::ZERO; ny];
    for ix in 0..nx {
        for iy in 0..ny { c[iy] = buf[iy * nx + ix]; }
        fft(&mut c, inv);
        for iy in 0..ny { buf[iy * nx + ix] = c[iy]; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fft_roundtrip() {
        let n = 8;
        let mut b: Vec<Complex64> = (0..n).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let o = b.clone();
        fft(&mut b, false);
        fft(&mut b, true);
        for v in b.iter_mut() { *v *= Complex64::new(1.0 / n as f64, 0.0); }
        let e: f64 = o.iter().zip(b.iter()).map(|(a,b)| (a-b).norm()).sum();
        assert!(e < 1e-10, "fft roundtrip err={:.6e}", e);
    }
    #[test]
    fn identity_kernel() {
        let nx = 4; let ny = 4;
        let mut k = vec![Complex64::ZERO; nx * ny];
        k[0] = Complex64::new(1.0, 0.0);
        let x: Vec<Complex64> = (0..nx*ny).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let y = bttb_matvec(&k, &x, nx, ny);
        let e: f64 = x.iter().zip(y.iter()).map(|(a,b)| (a-b).norm()).sum();
        assert!(e < 1e-10, "id err={:.6e}", e);
    }
}
