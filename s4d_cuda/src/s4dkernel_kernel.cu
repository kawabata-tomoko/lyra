// === s4d_kernel.cu ===
#include <cuda_runtime.h>
#include <thrust/complex.h> // Use thrust complex for device code
#include <c10/cuda/CUDAStream.h> // For getting current stream
#include <torch/extension.h>
#include <cufft.h>
#include <cmath> // For std::exp, std::log etc. on host if needed, device math functions are usually built-in

#include "s4dkernel_kernel.h" // Include the header with declarations and autograd classes

// Constants for CUDA kernels
constexpr int THREADS_PER_BLOCK = 256;

// Helper to calculate grid size
inline int GET_BLOCKS(const int N) {
    return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

// -------- compute_K forward/backward kernels --------

// Forward kernel: Computes K based on S4D formula
__global__ void compute_K_forward_kernel(
    const float* __restrict__ dt,
    const float* __restrict__ Creal, const float* __restrict__ Cim,
    const float* __restrict__ logA, const float* __restrict__ Aimag,
    int H, int N2, int L,
    float* __restrict__ K_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * L) return;

    int h = idx / L;
    int l = idx % L;

    thrust::complex<float> K_val(0.0f, 0.0f);
    float dt_h = dt[h];
    const float* Creal_h = Creal + h * N2;
    const float* Cim_h   = Cim + h * N2;
    const float* logA_h  = logA + h * N2;
    const float* Aimag_h = Aimag + h * N2;

    for (int n = 0; n < N2; ++n) {
        // Recompute A = -exp(logA) + j * Aimag
        // IMPORTANT FIX: Use '+' not '▪'
        thrust::complex<float> A = thrust::complex<float>(-expf(logA_h[n]), Aimag_h[n]);
        thrust::complex<float> C(Creal_h[n], Cim_h[n]);

        thrust::complex<float> dtA = A * dt_h;
        thrust::complex<float> exp_dtA = thrust::exp(dtA);

        // Compute C' = C * (exp(dtA) - 1) / A safely for A close to 0
        thrust::complex<float> C_prime;
        // Avoid division by zero or precision issues when A is small
        // Use Taylor expansion for (exp(x) - 1) / x ≈ 1 + x/2 + x^2/6 + ... around x=0
        // Here x = dtA. If |dtA| is small, use approximation.
        // A simpler check: if |A| is small. If A = 0, dtA=0, exp(dtA)=1, numerator is 0.
        // Let's use a tolerance check.
        const float abs_A_sq = A.real() * A.real() + A.imag() * A.imag();
        const float tol = 1e-12f; // Tolerance for A being close to zero

        if (abs_A_sq < tol) {
             // If A is very close to 0, dtA is also very close to 0.
             // (exp(dtA) - 1)/A approaches dt.
             C_prime = C * dt_h;
        } else {
            C_prime = C * (exp_dtA - 1.0f) / A;
        }


        // K_l = Sum_n C'_n * exp(dtA * l)
        thrust::complex<float> exp_dtA_l;
        if (l == 0) {
            exp_dtA_l = thrust::complex<float>(1.0f, 0.0f);
        } else {
            // Efficiently compute exp(dtA * l) = (exp(dtA))^l
            // For stability/performance, maybe not power, just recalculate exp
             exp_dtA_l = thrust::exp(dtA * static_cast<float>(l));
            // Or potentially use exp_dtA_l = thrust::pow(exp_dtA, l); // Might be less stable?
        }

        K_val += C_prime * exp_dtA_l;
    }
    // K = 2 * Real part
    K_out[idx] = 2.0f * K_val.real();
}


// Backward kernel for C_real
__global__ void compute_K_backward_Creal_kernel(
    const float* __restrict__ grad_K, const float* __restrict__ dt,
    const float* __restrict__ Creal, const float* __restrict__ Cim, // C needed for A calculation? No.
    const float* __restrict__ logA, const float* __restrict__ Aimag,
    int H, int N2, int L,
    float* __restrict__ grad_Creal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * N2) return;

    int h = idx / N2;
    int n = idx % N2;

    float grad_val = 0.0f;
    float dt_h = dt[h];
    const float* grad_K_h = grad_K + h * L;

    // Recompute A
    thrust::complex<float> A = thrust::complex<float>(-expf(logA[idx]), Aimag[idx]);
    thrust::complex<float> dtA = A * dt_h;
    thrust::complex<float> exp_dtA = thrust::exp(dtA);

    // Recompute base = (exp(dtA) - 1) / A safely
    thrust::complex<float> base;
    const float abs_A_sq = A.real() * A.real() + A.imag() * A.imag();
    const float tol = 1e-12f;
    if (abs_A_sq < tol) {
        base = dt_h;
    } else {
        base = (exp_dtA - 1.0f) / A;
    }


    thrust::complex<float> exp_dtA_l_term = 1.0f; // For l=0
    for (int l = 0; l < L; ++l) {
        // dK_l / dCreal_n = 2 * Re( (d C'_n / dCreal_n) * exp(dtA*l) )
        // d C'_n / dCreal_n = base (since C' = (Creal + j*Cim) * base)
        // dK_l / dCreal_n = 2 * Re( base * exp(dtA*l) )
        // grad_Creal_n = Sum_l grad_K_l * (dK_l / dCreal_n)

        if (l > 0) {
            // exp_dtA_l_term *= exp_dtA; // Accumulate power (potential precision issue for large l)
             exp_dtA_l_term = thrust::exp(dtA * static_cast<float>(l)); // Recalculate
        }

        grad_val += grad_K_h[l] * 2.0f * (base * exp_dtA_l_term).real();
    }

    grad_Creal[idx] = grad_val;
}

// Backward kernel for C_imag
__global__ void compute_K_backward_Cimag_kernel(
    const float* __restrict__ grad_K, const float* __restrict__ dt,
    const float* __restrict__ Creal, const float* __restrict__ Cim, // Not needed
    const float* __restrict__ logA, const float* __restrict__ Aimag,
    int H, int N2, int L,
    float* __restrict__ grad_Cimag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * N2) return;

    int h = idx / N2;
    int n = idx % N2;

    float grad_val = 0.0f;
    float dt_h = dt[h];
    const float* grad_K_h = grad_K + h * L;

    // Recompute A, dtA, base
    thrust::complex<float> A = thrust::complex<float>(-expf(logA[idx]), Aimag[idx]);
    thrust::complex<float> dtA = A * dt_h;
    thrust::complex<float> exp_dtA = thrust::exp(dtA);

    thrust::complex<float> base;
    const float abs_A_sq = A.real() * A.real() + A.imag() * A.imag();
    const float tol = 1e-12f;
    if (abs_A_sq < tol) {
        base = dt_h;
    } else {
        base = (exp_dtA - 1.0f) / A;
    }

    thrust::complex<float> exp_dtA_l_term = 1.0f;
    for (int l = 0; l < L; ++l) {
        // dK_l / dCimag_n = 2 * Re( (d C'_n / dCimag_n) * exp(dtA*l) )
        // d C'_n / dCimag_n = j * base
        // dK_l / dCimag_n = 2 * Re( j * base * exp(dtA*l) )
        // grad_Cimag_n = Sum_l grad_K_l * (dK_l / dCimag_n)

         if (l > 0) {
             exp_dtA_l_term = thrust::exp(dtA * static_cast<float>(l)); // Recalculate
         }
         thrust::complex<float> j_base_exp = thrust::complex<float>(0.0f, 1.0f) * base * exp_dtA_l_term;

        grad_val += grad_K_h[l] * 2.0f * j_base_exp.real();
    }

    grad_Cimag[idx] = grad_val;
}


// Backward kernel for log_dt
// This requires calculating dK/d(log_dt) = dK/d(dt) * d(dt)/d(log_dt) = dK/d(dt) * dt
__global__ void compute_K_backward_logdt_kernel(
    const float* __restrict__ grad_K, const float* __restrict__ dt,
    const float* __restrict__ Creal, const float* __restrict__ Cim,
    const float* __restrict__ logA, const float* __restrict__ Aimag,
    int H, int N2, int L,
    float* __restrict__ grad_log_dt)
{
    // Each thread calculates the gradient for one 'h'
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) return;

    float grad_val = 0.0f;
    float dt_h = dt[h];
    const float* grad_K_h = grad_K + h * L;
    const float* Creal_h = Creal + h * N2;
    const float* Cim_h   = Cim + h * N2;
    const float* logA_h  = logA + h * N2;
    const float* Aimag_h = Aimag + h * N2;

    for (int l = 0; l < L; ++l) {
        if (grad_K_h[l] == 0.0f) continue; // Skip if downstream grad is zero

        thrust::complex<float> dK_l_ddt_h(0.0f, 0.0f);
        for (int n = 0; n < N2; ++n) {
             thrust::complex<float> A = thrust::complex<float>(-expf(logA_h[n]), Aimag_h[n]);
             thrust::complex<float> C(Creal_h[n], Cim_h[n]);

             // Need d/d(dt) [ C' * exp(dt*A*l) ]
             // C' = C * (exp(dt*A) - 1) / A
             // Let f(dt) = C * [(exp(dt*A)-1)/A] * exp(dt*A*l)
             //           = C/A * [exp(dt*A*(l+1)) - exp(dt*A*l)]
             // df/d(dt)  = C/A * [ A*(l+1)*exp(dt*A*(l+1)) - A*l*exp(dt*A*l) ]
             //           = C * [ (l+1)*exp(dt*A*(l+1)) - l*exp(dt*A*l) ]

             thrust::complex<float> dtA = A * dt_h;
             thrust::complex<float> term1 = (float)(l + 1) * thrust::exp(dtA * (float)(l + 1));
             thrust::complex<float> term2 = (float)l * thrust::exp(dtA * (float)l); // Term is 0 if l=0
             dK_l_ddt_h += C * (term1 - term2);
        }
        // Accumulate: grad_log_dt = Sum_l grad_K_l * dK_l/d(log_dt)
        // dK_l/d(log_dt) = 2 * Re(dK_l_ddt_h) * dt_h
        grad_val += grad_K_h[l] * 2.0f * dK_l_ddt_h.real() * dt_h;
    }
    grad_log_dt[h] = grad_val;
}

// Backward kernel for log_A_real
// dK/d(logA) = dK/dA_real * dA_real/d(logA) = dK/dA_real * (-exp(logA)) = dK/dA_real * A_real
__global__ void compute_K_backward_logAreal_kernel(
    const float* __restrict__ grad_K, const float* __restrict__ dt,
    const float* __restrict__ Creal, const float* __restrict__ Cim,
    const float* __restrict__ logA, const float* __restrict__ Aimag,
    int H, int N2, int L,
    float* __restrict__ grad_logA)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * N2) return;

    int h = idx / N2;
    int n = idx % N2;

    float grad_val = 0.0f;
    float dt_h = dt[h];
    const float* grad_K_h = grad_K + h * L;
    thrust::complex<float> C(Creal[idx], Cim[idx]);
    float logA_hn = logA[idx];
    float Aimag_hn = Aimag[idx];
    float Areal_hn = -expf(logA_hn); // A_real
    thrust::complex<float> A = thrust::complex<float>(Areal_hn, Aimag_hn);

    for (int l = 0; l < L; ++l) {
        if (grad_K_h[l] == 0.0f) continue;

        // Need d/d(A_real) [ C' * exp(dt*A*l) ] where A = A_real + j*A_imag
        // This derivative is complex. Let g(A) = C' * exp(dt*A*l)
        // dg/dA_real = ?
        // Using chain rule might be easier: dg/dA * dA/dA_real = dg/dA * 1
        // Let f(A) = (exp(dt*A)-1)/A
        // Let h(A) = exp(dt*A*l)
        // g(A) = C * f(A) * h(A)
        // dg/dA = C * [ f'(A)h(A) + f(A)h'(A) ]
        // f'(A) = d/dA [ (e^(dtA)-1)/A ] = (A*dt*e^(dtA) - (e^(dtA)-1)) / A^2
        // h'(A) = d/dA [ e^(dtAl) ] = dt*l*e^(dtAl)

        thrust::complex<float> dtA = A * dt_h;
        thrust::complex<float> exp_dtA = thrust::exp(dtA);
        thrust::complex<float> exp_dtA_l = thrust::exp(dtA * (float)l);

        thrust::complex<float> fA, f_prime_A;
        const float abs_A_sq = A.real() * A.real() + A.imag() * A.imag();
        const float tol = 1e-12f;
        if (abs_A_sq < tol) {
            // Limit A->0: f(A) -> dt, f'(A) -> dt^2/2
             fA = dt_h;
             f_prime_A = 0.5f * dt_h * dt_h; // Check Taylor expansion
        } else {
            fA = (exp_dtA - 1.0f) / A;
            f_prime_A = (A * dt_h * exp_dtA - fA * A) / (A*A); // Simplified (A*f')
        }

        thrust::complex<float> h_prime_A = dt_h * (float)l * exp_dtA_l;
        thrust::complex<float> dg_dA = C * (f_prime_A * exp_dtA_l + fA * h_prime_A);

        // dg/dA_real = dg/dA * dA/dA_real = dg/dA * 1
        thrust::complex<float> dg_dAreal = dg_dA;

        // Accumulate: grad_logA = Sum_l grad_K_l * dK_l/d(logA)
        // dK_l/d(logA) = 2 * Re(dg/dA_real) * dA_real/d(logA)
        // dA_real/d(logA) = -exp(logA) = A_real
        grad_val += grad_K_h[l] * 2.0f * dg_dAreal.real() * Areal_hn;
    }
    grad_logA[idx] = grad_val;
}


// Backward kernel for A_imag
// dK/d(A_imag) = dK/dA_imag
__global__ void compute_K_backward_Aimag_kernel(
    const float* __restrict__ grad_K, const float* __restrict__ dt,
    const float* __restrict__ Creal, const float* __restrict__ Cim,
    const float* __restrict__ logA, const float* __restrict__ Aimag,
    int H, int N2, int L,
    float* __restrict__ grad_Aimag)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * N2) return;

    int h = idx / N2;
    int n = idx % N2;

    float grad_val = 0.0f;
    float dt_h = dt[h];
    const float* grad_K_h = grad_K + h * L;
    thrust::complex<float> C(Creal[idx], Cim[idx]);
    float logA_hn = logA[idx];
    float Aimag_hn = Aimag[idx];
    float Areal_hn = -expf(logA_hn);
    thrust::complex<float> A = thrust::complex<float>(Areal_hn, Aimag_hn);

    for (int l = 0; l < L; ++l) {
        if (grad_K_h[l] == 0.0f) continue;

        // Need d/d(A_imag) [ C' * exp(dt*A*l) ]
        // Chain rule: dg/dA * dA/dA_imag = dg/dA * j
        // dg/dA was computed for logAreal gradient.

        thrust::complex<float> dtA = A * dt_h;
        thrust::complex<float> exp_dtA = thrust::exp(dtA);
        thrust::complex<float> exp_dtA_l = thrust::exp(dtA * (float)l);

        thrust::complex<float> fA, f_prime_A;
        const float abs_A_sq = A.real() * A.real() + A.imag() * A.imag();
        const float tol = 1e-12f;
        if (abs_A_sq < tol) {
             fA = dt_h;
             f_prime_A = 0.5f * dt_h * dt_h;
        } else {
            fA = (exp_dtA - 1.0f) / A;
            f_prime_A = (A * dt_h * exp_dtA - fA * A) / (A*A);
        }
        thrust::complex<float> h_prime_A = dt_h * (float)l * exp_dtA_l;
        thrust::complex<float> dg_dA = C * (f_prime_A * exp_dtA_l + fA * h_prime_A);

        // dg/dA_imag = dg/dA * dA/dA_imag = dg/dA * j
        thrust::complex<float> dg_dAimag = dg_dA * thrust::complex<float>(0.0f, 1.0f);

        // Accumulate: grad_Aimag = Sum_l grad_K_l * dK_l/d(A_imag)
        // dK_l/d(A_imag) = 2 * Re(dg/dA_imag)
        grad_val += grad_K_h[l] * 2.0f * dg_dAimag.real();
    }
    grad_Aimag[idx] = grad_val;
}


// -------- Host functions launching the kernels --------

// Compute K host function
void compute_K_forward_kernel_launcher(
    const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* K_out)
{
    const int total_threads = H * L;
    const int blocks = GET_BLOCKS(total_threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compute_K_forward_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        dt, Creal, Cim, logA, Aimag, H, N2, L, K_out);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Compute K backward host functions
void compute_K_backward_Creal_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_Creal)
{
    const int total_threads = H * N2;
    const int blocks = GET_BLOCKS(total_threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compute_K_backward_Creal_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        grad_K, dt, Creal, Cim, logA, Aimag, H, N2, L, grad_Creal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_K_backward_Cimag_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_Cimag)
{
    const int total_threads = H * N2;
    const int blocks = GET_BLOCKS(total_threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compute_K_backward_Cimag_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        grad_K, dt, Creal, Cim, logA, Aimag, H, N2, L, grad_Cimag);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_K_backward_logdt_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_log_dt)
{
    const int total_threads = H; // One thread per H dimension
    const int blocks = GET_BLOCKS(total_threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // Note: Kernel calculates grad for log_dt, input dt is exp(log_dt)
    compute_K_backward_logdt_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        grad_K, dt, Creal, Cim, logA, Aimag, H, N2, L, grad_log_dt);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_K_backward_logAreal_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_logA)
{
    const int total_threads = H * N2;
    const int blocks = GET_BLOCKS(total_threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compute_K_backward_logAreal_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        grad_K, dt, Creal, Cim, logA, Aimag, H, N2, L, grad_logA);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void compute_K_backward_Aimag_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_Aimag)
{
    const int total_threads = H * N2;
    const int blocks = GET_BLOCKS(total_threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    compute_K_backward_Aimag_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        grad_K, dt, Creal, Cim, logA, Aimag, H, N2, L, grad_Aimag);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// -------- FFT conv1d with cuFFT --------
// Static plans are problematic if L changes. Re-creating each time is safer but slower.
// A better approach might involve caching plans based on (batch_size, n).
// For simplicity, we stick to re-creation for now.
static cufftHandle plan_r2c = 0;
static cufftHandle plan_c2r = 0;
static int current_batch = -1;
static int current_n = -1;

void init_fft_plan(int batch, int n) {
    // Only recreate plans if size or batch changes
    if (batch != current_batch || n != current_n) {
       if (plan_r2c) cufftDestroy(plan_r2c);
       if (plan_c2r) cufftDestroy(plan_c2r);

       cufftResult res_r2c = cufftPlanMany(&plan_r2c, 1, &n,
                                            nullptr, 1, n, // input stride/dist
                                            nullptr, 1, (n/2+1), // output stride/dist
                                            CUFFT_R2C, batch);
       TORCH_CHECK(res_r2c == CUFFT_SUCCESS, "cufftPlanMany R2C failed");

       cufftResult res_c2r = cufftPlanMany(&plan_c2r, 1, &n,
                                            nullptr, 1, (n/2+1), // input stride/dist
                                            nullptr, 1, n, // output stride/dist
                                            CUFFT_C2R, batch);
        TORCH_CHECK(res_c2r == CUFFT_SUCCESS, "cufftPlanMany C2R failed");

        current_batch = batch;
        current_n = n;
    }
    // Associate plan with the current stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cufftSetStream(plan_r2c, stream);
    cufftSetStream(plan_c2r, stream);
}

// Host wrapper for FFT convolution kernel
void fft_conv1d_kernel_launcher(
    const float* u, const float* K,
    int B, int H, int L, float* y_out)
{
  TORCH_CHECK(B > 0 && H > 0 && L > 0, "Input dimensions must be positive");
  int n = 2 * L;       // Padded length for FFT
  int nFreq = n / 2 + 1; // Number of complex frequencies
  int batch_fft = B * H;   // Total number of 1D FFTs

  // Allocate padded buffers on GPU
  // Use torch::empty for better integration with PyTorch's memory management
  auto options_real = torch::dtype(torch::kFloat32).device(torch::kCUDA);
  auto options_complex = torch::dtype(torch::kComplexFloat).device(torch::kCUDA);

  auto u_pad = torch::zeros({batch_fft, n}, options_real);
  // Use advanced indexing to copy u into the first L elements
  u_pad.slice(1, 0, L).copy_(torch::from_blob((void*)u, {batch_fft, L}, options_real).contiguous(), /*non_blocking=*/true);


  auto K_pad = torch::zeros({H, n}, options_real);
  // Copy K into the first L elements
   K_pad.slice(1, 0, L).copy_(torch::from_blob((void*)K, {H, L}, options_real).contiguous(), /*non_blocking=*/true);


  // Allocate frequency-domain buffers
  auto Uf = torch::empty({batch_fft, nFreq}, options_complex);
  auto Kf = torch::empty({H, nFreq}, options_complex);

  // Initialize FFT plans (safe re-creation)
  init_fft_plan(batch_fft, n); // Plan for u (batch_fft)
    // We need a separate plan for K if batch size is different (H vs batch_fft)
    // Or we can execute K transform H times with batch=1 plan?
    // Let's use a plan for H transforms
    cufftHandle plan_k_r2c = 0;
    cufftPlanMany(&plan_k_r2c, 1, &n, nullptr, 1, n, nullptr, 1, nFreq, CUFFT_R2C, H);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cufftSetStream(plan_k_r2c, stream);


  // Execute R2C FFTs
  cufftResult res_u = cufftExecR2C(plan_r2c,
                                   const_cast<float*>(u_pad.data_ptr<float>()), // API expects non-const but doesn't modify
                                   reinterpret_cast<cufftComplex*>(Uf.data_ptr<c10::complex<float>>()));
  TORCH_CHECK(res_u == CUFFT_SUCCESS, "cufftExecR2C for u failed");

  cufftResult res_k = cufftExecR2C(plan_k_r2c, // Use K's plan
                                   const_cast<float*>(K_pad.data_ptr<float>()),
                                   reinterpret_cast<cufftComplex*>(Kf.data_ptr<c10::complex<float>>()));
  TORCH_CHECK(res_k == CUFFT_SUCCESS, "cufftExecR2C for K failed");
  cufftDestroy(plan_k_r2c); // Destroy temporary plan


  // Perform element-wise multiplication in frequency domain: Yf = Uf * Kf (broadcasted)
  // Kf needs to be repeated B times: (H, nFreq) -> (B*H, nFreq)
  auto Kf_rep = Kf.repeat({B, 1}); // Repeat B times along the first dimension
  auto Yf = Uf * Kf_rep;


  // Execute C2R inverse FFT
  auto y_pad = torch::empty({batch_fft, n}, options_real);
  cufftResult res_y = cufftExecC2R(plan_c2r,
                                   reinterpret_cast<cufftComplex*>(Yf.data_ptr<c10::complex<float>>()),
                                   y_pad.data_ptr<float>());
   TORCH_CHECK(res_y == CUFFT_SUCCESS, "cufftExecC2R for y failed");


  // Normalize the result
  y_pad.div_(static_cast<float>(n));

  // Extract the first L elements and copy to output
  auto y_flat = y_pad.slice(1, 0, L).contiguous(); // Get the relevant part and make contiguous

  // Copy result to the output pointer y_out, reshaping implicitly
   cudaError_t cuda_err = cudaMemcpyAsync(y_out, y_flat.data_ptr<float>(), y_flat.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
   TORCH_CHECK(cuda_err == cudaSuccess, "cudaMemcpyAsync failed in fft_conv1d");
}


// ======== C++ Interfaces for Python Binding ========

// Interface for Compute K (calls the forward kernel launcher)
// This version is WITHOUT autograd support.
torch::Tensor compute_K_cuda(
    torch::Tensor log_dt,
    torch::Tensor C_real,
    torch::Tensor C_imag,
    torch::Tensor log_A_real,
    torch::Tensor A_imag,
    int L) {

    TORCH_CHECK(log_dt.is_cuda() && C_real.is_cuda() && C_imag.is_cuda() && log_A_real.is_cuda() && A_imag.is_cuda(), "Inputs must be CUDA tensors");
    // Add more checks for contiguity, dtype, shapes if needed

    int H = log_dt.size(0);
    int N2 = C_real.size(1);
    auto K = torch::empty({H, L}, log_dt.options());
    auto dt = torch::exp(log_dt); // Kernel uses dt

    compute_K_forward_kernel_launcher(
      dt.data_ptr<float>(), C_real.data_ptr<float>(), C_imag.data_ptr<float>(),
      log_A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
      H, N2, L, K.data_ptr<float>());

    return K;
}

// Interface for FFT Convolution (calls the forward kernel launcher)
// This version is WITHOUT autograd support.
torch::Tensor fft_conv1d_cuda(
    torch::Tensor u,
    torch::Tensor K) {

    TORCH_CHECK(u.is_cuda() && K.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(u.dim() == 3 && K.dim() == 2, "Input u must be 3D (B, H, L) and K must be 2D (H, L)");
    TORCH_CHECK(u.size(1) == K.size(0), "Dimension H mismatch between u and K");
    TORCH_CHECK(u.size(2) == K.size(1), "Dimension L mismatch between u and K");
    TORCH_CHECK(u.is_contiguous() && K.is_contiguous(), "Inputs must be contiguous"); // Or make contiguous inside

    int B = u.size(0);
    int H = u.size(1);
    int L = u.size(2);

    auto y = torch::empty_like(u);

    fft_conv1d_kernel_launcher(
        u.data_ptr<float>(), K.data_ptr<float>(),
        B, H, L, y.data_ptr<float>());

    return y;
}