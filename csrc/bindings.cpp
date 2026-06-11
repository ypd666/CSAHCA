#include <torch/extension.h>

torch::Tensor csa_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor chunk_indices,
    int64_t chunk_size);

torch::Tensor csa_decode_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor chunk_indices,
    int64_t chunk_size) {
  if (!q.is_cuda() || !k_cache.is_cuda() || !v_cache.is_cuda() || !chunk_indices.is_cuda()) {
    TORCH_CHECK(false, "all tensors must be CUDA tensors");
  }
  return csa_decode_forward_cuda(q, k_cache, v_cache, chunk_indices, chunk_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csa_decode_forward", &csa_decode_forward, "Naive CSA decode forward (CUDA)");
}

