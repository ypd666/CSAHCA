#include <torch/extension.h>

torch::Tensor csa_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor chunk_indices,
    int64_t chunk_size);

torch::Tensor csa_decode_forward_tiled_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor chunk_indices,
    int64_t chunk_size,
    int64_t tile_size);

torch::Tensor dsv4_swa_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor paged_k_cache_u8,
    torch::Tensor token_indices,
    torch::Tensor topk_lengths,
    torch::Tensor attn_sink,
    int64_t page_size,
    double softmax_scale);

torch::Tensor dsv4_sparse_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor paged_k_cache_u8,
    torch::Tensor token_indices,
    torch::Tensor topk_lengths,
    torch::Tensor extra_paged_k_cache_u8,
    torch::Tensor extra_token_indices,
    torch::Tensor extra_topk_lengths,
    torch::Tensor attn_sink,
    int64_t page_size,
    int64_t extra_page_size,
    double softmax_scale);

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

torch::Tensor csa_decode_forward_tiled(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor chunk_indices,
    int64_t chunk_size,
    int64_t tile_size) {
  if (!q.is_cuda() || !k_cache.is_cuda() || !v_cache.is_cuda() || !chunk_indices.is_cuda()) {
    TORCH_CHECK(false, "all tensors must be CUDA tensors");
  }
  return csa_decode_forward_tiled_cuda(q, k_cache, v_cache, chunk_indices, chunk_size, tile_size);
}

torch::Tensor dsv4_swa_decode_forward(
    torch::Tensor q,
    torch::Tensor paged_k_cache_u8,
    torch::Tensor token_indices,
    torch::Tensor topk_lengths,
    torch::Tensor attn_sink,
    int64_t page_size,
    double softmax_scale) {
  if (!q.is_cuda() || !paged_k_cache_u8.is_cuda() || !token_indices.is_cuda()) {
    TORCH_CHECK(false, "q, paged_k_cache_u8, and token_indices must be CUDA tensors");
  }
  return dsv4_swa_decode_forward_cuda(
      q, paged_k_cache_u8, token_indices, topk_lengths, attn_sink, page_size, softmax_scale);
}

torch::Tensor dsv4_sparse_decode_forward(
    torch::Tensor q,
    torch::Tensor paged_k_cache_u8,
    torch::Tensor token_indices,
    torch::Tensor topk_lengths,
    torch::Tensor extra_paged_k_cache_u8,
    torch::Tensor extra_token_indices,
    torch::Tensor extra_topk_lengths,
    torch::Tensor attn_sink,
    int64_t page_size,
    int64_t extra_page_size,
    double softmax_scale) {
  if (!q.is_cuda() || !paged_k_cache_u8.is_cuda() || !token_indices.is_cuda() ||
      !extra_paged_k_cache_u8.is_cuda() || !extra_token_indices.is_cuda()) {
    TORCH_CHECK(
        false,
        "q, paged_k_cache_u8, token_indices, extra_paged_k_cache_u8, and extra_token_indices "
        "must be CUDA tensors");
  }
  return dsv4_sparse_decode_forward_cuda(
      q,
      paged_k_cache_u8,
      token_indices,
      topk_lengths,
      extra_paged_k_cache_u8,
      extra_token_indices,
      extra_topk_lengths,
      attn_sink,
      page_size,
      extra_page_size,
      softmax_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csa_decode_forward", &csa_decode_forward, "Naive CSA decode forward (CUDA)");
  m.def("csa_decode_forward_tiled", &csa_decode_forward_tiled, "Tiled CSA decode forward (CUDA)");
  m.def("dsv4_swa_decode_forward", &dsv4_swa_decode_forward, "DSV4 SWA paged FP8 decode forward (CUDA)");
  m.def("dsv4_sparse_decode_forward", &dsv4_sparse_decode_forward, "DSV4 SWA + C4/C128 paged FP8 decode forward (CUDA)");
}
