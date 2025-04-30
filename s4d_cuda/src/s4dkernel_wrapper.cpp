// === s4dkernel_wrapper.cpp ===
#include <torch/extension.h>
#include "s4dkernel_kernel.h" // Include header with autograd class definitions and kernel declarations

// Helper function to apply ComputeKFunction (with autograd support)
// It handles splitting the complex C tensor from Python
torch::Tensor compute_K_autograd_wrapper(
    torch::Tensor log_dt,
    torch::Tensor C,           // Input C is the real tensor (H, N2, 2)
    torch::Tensor log_A_real,
    torch::Tensor A_imag,
    int64_t L) {               // Use int64_t consistent with PyTorch sizes

    // C comes in as (H, N/2, 2) view from Python
    TORCH_CHECK(C.dim() == 3 && C.size(2) == 2, "C must have shape (H, N/2, 2)");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous"); // Necessary for select().contiguous()

    // Split C into real and imaginary parts
    auto C_real = C.select(2, 0).contiguous();
    auto C_imag = C.select(2, 1).contiguous();

    // Call the autograd Function's apply method
    return ComputeKFunction::apply(log_dt, C_real, C_imag, log_A_real, A_imag, L);
}


// Helper function to apply FFTConv1dFunction (with autograd support)
// If you decide to implement and use the custom FFT autograd
/*
torch::Tensor fft_conv1d_autograd_wrapper(torch::Tensor u, torch::Tensor K) {
    return FFTConv1dFunction::apply(u, K);
}
*/

// Pybind11 Module Definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_K_autograd", &compute_K_autograd_wrapper,
          "Compute S4D kernel K with Autograd support (CUDA)",
          py::arg("log_dt"),
          py::arg("C"),           // Pass the combined real tensor
          py::arg("log_A_real"),
          py::arg("A_imag"),
          py::arg("L"));

    // Expose the non-autograd forward FFT convolution if needed for debugging/specific cases
    m.def("fft_conv1d_forward", &fft_conv1d_cuda,
          "FFT-based 1D convolution forward pass (CUDA, no autograd)",
          py::arg("u"),
          py::arg("K"));

    // If FFTConv1dFunction is implemented, bind its wrapper:
    /*
    m.def("fft_conv1d_autograd", &fft_conv1d_autograd_wrapper,
          "FFT-based 1D convolution with Autograd support (CUDA)",
          py::arg("u"),
          py::arg("K"));
    */

    // It's generally not necessary to expose the non-autograd compute_K
    // unless specifically required for non-training purposes.
    /*
    m.def("compute_K_forward", &compute_K_cuda,
          "Compute S4D kernel K forward pass (CUDA, no autograd)",
           py::arg("log_dt"), py::arg("C_real"), py::arg("C_imag"),
           py::arg("log_A_real"), py::arg("A_imag"), py::arg("L"));
    */
}