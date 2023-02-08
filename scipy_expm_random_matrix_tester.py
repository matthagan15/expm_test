#!/usr/bin/env python
import numpy as np
import scipy.linalg

n = 200
samps = 100
scale = 1.
diff_norms = []
avg_entry_diff = []
for _ in range(samps):
    m = np.random.normal(scale = np.sqrt(2)/2, size=(n,n * 2)).view(np.complex128)
    m = m.dot(m.conj().T)
    new_vals = []
    for _ in range(n):
        new_vals.append(scale * np.random.normal() + scale * 1j * np.random.normal())
    # new_vals = np.ndarray([np.random.normal() + 1.j * np.random.normal() for _ in range(n)], dtype=np.complex128)
    new_exp_vals = np.diag(np.exp(new_vals))
    new_normal_vals = np.diag(new_vals)
    (vals, vecs) = np.linalg.eig(m)
    expected_expm = np.linalg.multi_dot([vecs, new_exp_vals, vecs.conj().T])
    reconstructed = np.linalg.multi_dot([vecs, new_normal_vals, vecs.conj().T])
    computed_expm = scipy.linalg.expm(reconstructed)
    diff_norms.append(np.linalg.norm(computed_expm - expected_expm, ord=1))
    avg_entry_diff.append(np.mean(np.abs(computed_expm - expected_expm)))
print("dimension of matrix: ", n, "x", n) 
print("scaling factor for random eigenvalues: ", scale)
print("number samples: ", samps)
print("one norm of matrix expected - computed (over epsilon):", np.mean(diff_norms) / np.finfo(float).eps, " +- ", np.std(diff_norms) / np.finfo(float).eps)
# print("std one norm of diff:", np.std(diff_norms) / np.finfo(float).eps)
print("average magnitude of each entry of expected - computed (over epsilon):", np.mean(avg_entry_diff) / np.finfo(float).eps)
