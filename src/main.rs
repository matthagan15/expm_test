use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::expm::expm;
use ndarray_linalg::{Eig, OperationNorm};
use num_complex::{Complex64 as c64, ComplexFloat};
use rand::*;
use std::process::Command;

fn random_sparse_matrix() {
    let mut rng = rand::thread_rng();
    let num_samples = 100;
    let num_qubits: u32 = 10; // controls dimension of resulting sparse matrix
    let dim = 2_usize.pow(num_qubits);
    let i = c64::new(0., 1.);
    let zero = c64::new(0., 0.);
    let pauli_x = array![[zero, c64::new(1., 0.)], [c64::new(1., 0.), zero]];
    let pauli_y = array![[zero, c64::new(0., -1.)], [c64::new(0., 1.), zero]];
    let pauli_z = array![[c64::new(1., 0.), zero], [zero, c64::new(-1., 0.)]];
    let mut results_difference_norms = Vec::with_capacity(num_samples);
    for ix in 0..num_samples {
        println!("{:2}% complete", ix as f64 / num_samples as f64);
        let mut matrix = Array2::<c64>::eye(2);
        for n in 0..num_qubits {
            let pauli_matrix = match rng.gen_range::<i32,_>(0..=3) {
                0 => {
                    Array2::<c64>::eye(2)
                },
                1 => {
                    pauli_x.clone()
                },
                2 => {
                    pauli_y.clone()
                },
                3 => {
                    pauli_z.clone()
                },
                _ => unreachable!(),
            };
            if n == 0 {
                matrix = matrix.dot(&pauli_matrix);
            } else {
                matrix = kron(&matrix, &pauli_matrix);
            }
        }
        // now check that this matrix squares to the identity
        let matrix_squared = matrix.dot(&matrix);
        let diff = &matrix_squared - Array2::<c64>::eye(dim);
        assert!(diff.opnorm_one().unwrap() < 10. * (dim as f64) * f64::EPSILON);
        let theta = 1. * std::f64::consts::PI * rng.gen::<f64>();
        let scaled_matrix = matrix.clone() * c64::new(0., theta);
        let expm_computed = expm(&scaled_matrix).unwrap();
        let expm_expected = Array2::<c64>::eye(dim) * theta.cos() + c64::new(0., theta.sin()) * matrix;
        let comp_diff = &expm_expected - &expm_computed;
        results_difference_norms.push(comp_diff.opnorm_one().unwrap()); 
    }

    let avg: f64 = results_difference_norms.iter().sum::<f64>() / results_difference_norms.len() as f64;
    println!("dimensions: {:}", dim);
    println!("average diff norm per epsilon: {:}", avg / f64::EPSILON);
}

fn random_dense_matrix() {
    let mut rng = rand::thread_rng();
    let n = 200;
    let samps = 100;
    let scale = 1.;
    let mut results = Vec::new();
    let mut avg_entry_error = Vec::new();
    // Used to control what pade approximation is most likely to be used.
    // the smaller the norm the lower the degree used.
    for _ in 0..samps {
        // Sample a completely random matrix.
        let mut matrix: Array2<c64> = Array2::<c64>::ones((n, n).f());
        matrix.mapv_inplace(|_| c64::new(rng.gen::<f64>() * 1., rng.gen::<f64>() * 1.));

        // Make m positive semidefinite so it has orthonormal eigenvecs.
        matrix = matrix.dot(&matrix.t().map(|x| x.conj()));
        let (mut eigs, vecs) = matrix.eig().unwrap();
        let adjoint_vecs = vecs.t().clone().mapv(|x| x.conj());

        // Generate new random eigenvalues (complex, previously m had real eigenvals)
        // and a new matrix m
        eigs.mapv_inplace(|_| scale * c64::new(rng.gen::<f64>(), rng.gen::<f64>()));
        let new_matrix = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);

        // compute the exponentiated matrix by exponentiating the eigenvalues
        // and doing V e^Lambda V^\dagger
        eigs.mapv_inplace(|x| x.exp());
        let eigen_expm = vecs.dot(&Array2::from_diag(&eigs)).dot(&adjoint_vecs);

        // Compute the expm routine, compute error metrics for this sample
        let expm_comp = expm(&new_matrix).unwrap();
        // println!("deg: {:}", deg);
        let diff = &expm_comp - &eigen_expm;
        avg_entry_error.push({
            let tot = diff.map(|x| x.abs()).into_iter().sum::<f64>();
            tot / (n * n) as f64
        });
        results.push(diff.opnorm_one().unwrap());
    }

    // compute averages
    let avg: f64 = results.iter().sum::<f64>() / results.len() as f64;
    let avg_entry_diff = avg_entry_error.iter().sum::<f64>() / avg_entry_error.len() as f64;
    let std: f64 = f64::powf(
        results.iter().map(|x| f64::powi(x - avg, 2)).sum::<f64>() / (results.len() - 1) as f64,
        0.5,
    );
    println!("collected {:} samples.", results.len());
    println!("scaling factor: {:}", scale);
    println!("dimensions: {:}", n);
    println!("diff norm per epsilon: {:} +- ({:})", avg / f64::EPSILON, std / f64::EPSILON);
    println!(
        "average entry error over epsilon: {:}",
        avg_entry_diff / f64::EPSILON
    );
}

fn main() {
    println!("Hello, world!");
    println!("###########################################################################################################################");
    // println!("python results below.");
    // let out = Command::new("/Users/matt/repos/expm_test/scipy_expm_random_matrix_tester.py")
    //     .output()
    //     .expect("failed to execute process");
    // let formatted_py_out = String::from_utf8(out.stdout).unwrap();
    // println!("{:}", formatted_py_out);
    // println!("###########################################################################################################################");
    println!("Random pauli test:");
    random_sparse_matrix();
    println!("###########################################################################################################################");
    println!("Random dense matrix test");
    random_dense_matrix();
}
