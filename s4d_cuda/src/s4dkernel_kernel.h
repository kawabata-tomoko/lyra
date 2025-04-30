// === s4dkernel_kernel.h ===
#pragma once
#include <torch/extension.h>
#include <vector> // Required for tensor_list

// Forward declarations for CUDA kernels (implemented in .cu)
// Note: These are the raw kernel launchers, not typically called directly from Python

// Compute K forward kernel
void compute_K_forward_kernel_launcher(
    const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* K_out);

// Compute K backward kernels
void compute_K_backward_Creal_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_Creal);

void compute_K_backward_Cimag_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_Cimag);

void compute_K_backward_logdt_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_log_dt);

void compute_K_backward_logAreal_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_logA);

void compute_K_backward_Aimag_kernel_launcher(
    const float* grad_K, const float* dt, const float* Creal, const float* Cim,
    const float* logA, const float* Aimag,
    int H, int N2, int L, float* grad_Aimag);

// FFT Convolution kernels (forward only needed if using custom autograd)
void fft_conv1d_kernel_launcher(
    const float* u, const float* K,
    int B, int H, int L, float* y);

// Host functions (callable from C++ wrapper) that launch kernels
torch::Tensor compute_K_cuda(
    torch::Tensor log_dt,
    torch::Tensor C_real,
    torch::Tensor C_imag,
    torch::Tensor log_A_real,
    torch::Tensor A_imag,
    int L
);

torch::Tensor fft_conv1d_cuda(
    torch::Tensor u,
    torch::Tensor K
);


// Autograd Function definitions (must be visible to all translation units)

// Note: We focus on ComputeKFunction as the primary custom autograd needed.
// FFT convolution autograd can often rely on PyTorch's built-in torch.fft autograd.
// If custom FFT autograd is strictly needed, FFTConv1dFunction would be defined here
// similar to ComputeKFunction, calling fft_conv1d_cuda for forward and implementing
// its specific backward logic (potentially using fft_conv1d_cuda again as shown
// in the original attempt).

class ComputeKFunction : public torch::autograd::Function<ComputeKFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
    torch::Tensor log_dt, torch::Tensor C_real, torch::Tensor C_imag,
    torch::Tensor log_A_real, torch::Tensor A_imag, int64_t L_int) {

    // Ensure inputs are contiguous and on CUDA
    log_dt = log_dt.contiguous();
    C_real = C_real.contiguous();
    C_imag = C_imag.contiguous();
    log_A_real = log_A_real.contiguous();
    A_imag = A_imag.contiguous();

    TORCH_CHECK(log_dt.is_cuda() && C_real.is_cuda() && C_imag.is_cuda() && log_A_real.is_cuda() && A_imag.is_cuda(), "All input tensors must be on CUDA");
    TORCH_CHECK(log_dt.dtype() == torch::kFloat32 && C_real.dtype() == torch::kFloat32 && C_imag.dtype() == torch::kFloat32 && log_A_real.dtype() == torch::kFloat32 && A_imag.dtype() == torch::kFloat32, "All input tensors must be float32");

    int H = log_dt.size(0);
    int N2 = C_real.size(1);
    int L = static_cast<int>(L_int); // Cast L to int for kernel calls

    auto K = torch::empty({H, L}, log_dt.options());
    auto dt = torch::exp(log_dt); // Need dt for backward

    // Launch the forward CUDA kernel
    compute_K_forward_kernel_launcher(
      dt.data_ptr<float>(), C_real.data_ptr<float>(), C_imag.data_ptr<float>(),
      log_A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
      H, N2, L, K.data_ptr<float>());

    // Save tensors needed for backward pass
    // Save dt instead of log_dt because kernel uses dt directly
    ctx->save_for_backward({dt, C_real, C_imag, log_A_real, A_imag});
    // Save non-tensor variables
    ctx->saved_data["L"] = L_int; // Store the original int64_t

    return K;
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    // Retrieve saved tensors and variables
    auto saved = ctx->get_saved_variables();
    auto dt = saved[0];
    auto C_real = saved[1];
    auto C_imag = saved[2];
    auto log_A_real = saved[3];
    auto A_imag = saved[4];

    int64_t L_int = ctx->saved_data["L"].toInt();
    int L = static_cast<int>(L_int); // Cast L to int for kernel calls

    auto grad_K = grad_outputs[0].contiguous();
    TORCH_CHECK(grad_K.is_cuda(), "Gradient of K must be on CUDA");
    TORCH_CHECK(grad_K.dtype() == torch::kFloat32, "Gradient of K must be float32");


    int H = dt.size(0);
    int N2 = C_real.size(1);

    // Allocate tensors for gradients of inputs
    auto grad_log_dt = torch::empty_like(dt); // Gradient is w.r.t log_dt
    auto grad_C_real = torch::empty_like(C_real);
    auto grad_C_imag = torch::empty_like(C_imag);
    auto grad_log_A_real = torch::empty_like(log_A_real); // Gradient is w.r.t log_A_real
    auto grad_A_imag = torch::empty_like(A_imag);

    // Launch backward CUDA kernels
    compute_K_backward_Creal_kernel_launcher(
        grad_K.data_ptr<float>(), dt.data_ptr<float>(), C_real.data_ptr<float>(), C_imag.data_ptr<float>(),
        log_A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
        H, N2, L, grad_C_real.data_ptr<float>());

    compute_K_backward_Cimag_kernel_launcher(
        grad_K.data_ptr<float>(), dt.data_ptr<float>(), C_real.data_ptr<float>(), C_imag.data_ptr<float>(),
        log_A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
        H, N2, L, grad_C_imag.data_ptr<float>());

    compute_K_backward_logdt_kernel_launcher(
        grad_K.data_ptr<float>(), dt.data_ptr<float>(), C_real.data_ptr<float>(), C_imag.data_ptr<float>(),
        log_A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
        H, N2, L, grad_log_dt.data_ptr<float>());

    compute_K_backward_logAreal_kernel_launcher(
        grad_K.data_ptr<float>(), dt.data_ptr<float>(), C_real.data_ptr<float>(), C_imag.data_ptr<float>(),
        log_A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
        H, N2, L, grad_log_A_real.data_ptr<float>());

    compute_K_backward_Aimag_kernel_launcher(
        grad_K.data_ptr<float>(), dt.data_ptr<float>(), C_real.data_ptr<float>(), C_imag.data_ptr<float>(),
        log_A_real.data_ptr<float>(), A_imag.data_ptr<float>(),
        H, N2, L, grad_A_imag.data_ptr<float>());

    // Return gradients corresponding to the inputs of the forward function
    // The last one is for L_int, which does not require grad (return undefined tensor)
    return {grad_log_dt, grad_C_real, grad_C_imag, grad_log_A_real, grad_A_imag, torch::Tensor()};
  }
};


// Optional: Define FFTConv1dFunction if custom FFT autograd is needed
/*
class FFTConv1dFunction : public torch::autograd::Function<FFTConv1dFunction> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor u, torch::Tensor K) {
    // Call fft_conv1d_cuda
    // Save u, K
  }

  static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                               torch::autograd::tensor_list grad_outputs) {
    // Retrieve u, K, grad_y
    // Calculate grad_u = fft_conv1d_cuda(grad_y, K.flip(-1))
    // Calculate grad_K = fft_conv1d_cuda(u.transpose(0,1), grad_y.transpose(0,1)).sum(1) // Adjust dims carefully
    // Return {grad_u, grad_K}
  }
};
*/