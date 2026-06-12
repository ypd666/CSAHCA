#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDsv4NopeDim = 448;
constexpr int kDsv4RopeDim = 64;
constexpr int kDsv4HeadDim = kDsv4NopeDim + kDsv4RopeDim;
constexpr int kDsv4TileSize = 64;
constexpr int kDsv4ScaleTiles = kDsv4NopeDim / kDsv4TileSize;
constexpr int kDsv4NopeRopeBytes = kDsv4NopeDim + kDsv4RopeDim * 2;
constexpr int kDsv4ScaleBytesPerToken = kDsv4ScaleTiles + 1;

__device__ __forceinline__ float dsv4_load_fp8_e4m3(const uint8_t byte) {
  __nv_fp8_e4m3 value;
  *reinterpret_cast<uint8_t*>(&value) = byte;
  return static_cast<float>(value);
}

__device__ __forceinline__ float dsv4_load_bf16(const uint8_t* base) {
  const __nv_bfloat16 raw = *reinterpret_cast<const __nv_bfloat16*>(base);
  return __bfloat162float(raw);
}

__device__ __forceinline__ float dsv4_load_k_dim(
    const uint8_t* __restrict__ cache,
    int64_t loc,
    int dim,
    int page_size,
    int bytes_per_page) {
  const int64_t page = loc / page_size;
  const int in_page = static_cast<int>(loc - page * page_size);
  const int64_t page_base = page * static_cast<int64_t>(bytes_per_page);
  const int64_t token_data_base = page_base + static_cast<int64_t>(in_page) * kDsv4NopeRopeBytes;

  if (dim < kDsv4NopeDim) {
    const int tile = dim / kDsv4TileSize;
    const int64_t scale_base =
        page_base + static_cast<int64_t>(page_size) * kDsv4NopeRopeBytes +
        static_cast<int64_t>(in_page) * kDsv4ScaleBytesPerToken;
    const uint8_t scale_u8 = cache[scale_base + tile];
    const float scale = exp2f(static_cast<float>(static_cast<int>(scale_u8) - 127));
    return dsv4_load_fp8_e4m3(cache[token_data_base + dim]) * scale;
  }

  const int rope_dim = dim - kDsv4NopeDim;
  const int64_t byte_offset = token_data_base + kDsv4NopeDim + static_cast<int64_t>(rope_dim) * 2;
  return dsv4_load_bf16(cache + byte_offset);
}

__device__ __forceinline__ void dsv4_accumulate_token(
    const uint8_t* __restrict__ cache,
    int64_t loc,
    int page_size,
    int bytes_per_page,
    const float* __restrict__ q_s,
    float* __restrict__ acc_s,
    float* __restrict__ red_s,
    int tid,
    float softmax_scale,
    float& max_score,
    float& norm) {
  float partial = 0.0f;
  for (int d = tid; d < kDsv4HeadDim; d += blockDim.x) {
    partial += q_s[d] * dsv4_load_k_dim(cache, loc, d, page_size, bytes_per_page);
  }

  red_s[tid] = partial;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_s[tid] += red_s[tid + stride];
    }
    __syncthreads();
  }

  const float score = red_s[0] * softmax_scale;
  const float next_max = max_score > score ? max_score : score;
  const float old_scale = expf(max_score - next_max);
  const float score_scale = expf(score - next_max);

  for (int d = tid; d < kDsv4HeadDim; d += blockDim.x) {
    const float value = dsv4_load_k_dim(cache, loc, d, page_size, bytes_per_page);
    acc_s[d] = acc_s[d] * old_scale + score_scale * value;
  }
  norm = norm * old_scale + score_scale;
  max_score = next_max;
  __syncthreads();
}

template <typename scalar_t>
__global__ void dsv4_swa_decode_forward_kernel(
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ paged_k_cache,
    const int* __restrict__ token_indices,
    const int* __restrict__ topk_lengths,
    const uint8_t* __restrict__ extra_paged_k_cache,
    const int* __restrict__ extra_token_indices,
    const int* __restrict__ extra_topk_lengths,
    const float* __restrict__ attn_sink,
    scalar_t* __restrict__ out,
    int num_queries,
    int heads,
    int topk_slots,
    int extra_topk_slots,
    int page_size,
    int bytes_per_page,
    int extra_page_size,
    int extra_bytes_per_page,
    float softmax_scale) {
  const int qh = blockIdx.x;
  const int query = qh / heads;
  const int head = qh - query * heads;
  const int tid = threadIdx.x;

  extern __shared__ float smem[];
  float* q_s = smem;
  float* acc_s = q_s + kDsv4HeadDim;
  float* red_s = acc_s + kDsv4HeadDim;

  const int q_base = (query * heads + head) * kDsv4HeadDim;
  for (int d = tid; d < kDsv4HeadDim; d += blockDim.x) {
    q_s[d] = static_cast<float>(q[q_base + d]);
    acc_s[d] = 0.0f;
  }
  __syncthreads();

  float max_score = -INFINITY;
  float norm = 0.0f;
  if (attn_sink != nullptr) {
    max_score = attn_sink[head];
    norm = 1.0f;
  }

  const int raw_topk = topk_lengths == nullptr ? topk_slots : topk_lengths[query];
  const int actual_topk = max(0, min(raw_topk, topk_slots));
  const int idx_base = query * topk_slots;

  for (int i = 0; i < actual_topk; ++i) {
    const int loc_i32 = token_indices[idx_base + i];
    if (loc_i32 < 0) {
      continue;
    }
    const int64_t loc = static_cast<int64_t>(loc_i32);
    dsv4_accumulate_token(
        paged_k_cache,
        loc,
        page_size,
        bytes_per_page,
        q_s,
        acc_s,
        red_s,
        tid,
        softmax_scale,
        max_score,
        norm);
  }

  if (extra_paged_k_cache != nullptr && extra_token_indices != nullptr && extra_topk_slots > 0) {
    const int raw_extra_topk = extra_topk_lengths == nullptr
        ? extra_topk_slots
        : extra_topk_lengths[query];
    const int actual_extra_topk = max(0, min(raw_extra_topk, extra_topk_slots));
    const int extra_idx_base = query * extra_topk_slots;

    for (int i = 0; i < actual_extra_topk; ++i) {
      const int loc_i32 = extra_token_indices[extra_idx_base + i];
      if (loc_i32 < 0) {
        continue;
      }
      const int64_t loc = static_cast<int64_t>(loc_i32);
      dsv4_accumulate_token(
          extra_paged_k_cache,
          loc,
          extra_page_size,
          extra_bytes_per_page,
          q_s,
          acc_s,
          red_s,
          tid,
          softmax_scale,
          max_score,
          norm);
    }
  }

  const float inv_norm = norm > 0.0f ? 1.0f / norm : 0.0f;
  for (int d = tid; d < kDsv4HeadDim; d += blockDim.x) {
    out[q_base + d] = static_cast<scalar_t>(acc_s[d] * inv_norm);
  }
}

template <typename scalar_t>
void launch_dsv4_swa_decode_forward(
    const torch::Tensor& q,
    const torch::Tensor& paged_k_cache_u8,
    const torch::Tensor& token_indices,
    const torch::Tensor& topk_lengths,
    const torch::Tensor& extra_paged_k_cache_u8,
    const torch::Tensor& extra_token_indices,
    const torch::Tensor& extra_topk_lengths,
    const torch::Tensor& attn_sink,
    torch::Tensor& out,
    int num_queries,
    int heads,
    int topk_slots,
    int extra_topk_slots,
    int page_size,
    int bytes_per_page,
    int extra_page_size,
    int extra_bytes_per_page,
    float softmax_scale) {
  const int block = 256;
  const dim3 grid(num_queries * heads);
  const size_t shared_bytes = static_cast<size_t>(2 * kDsv4HeadDim + block) * sizeof(float);
  const float* attn_sink_ptr = attn_sink.defined() && attn_sink.numel() > 0
      ? attn_sink.data_ptr<float>()
      : nullptr;
  const int* topk_lengths_ptr = topk_lengths.defined() && topk_lengths.numel() > 0
      ? topk_lengths.data_ptr<int>()
      : nullptr;
  const bool has_extra = extra_paged_k_cache_u8.defined() &&
      extra_paged_k_cache_u8.numel() > 0 &&
      extra_token_indices.defined() &&
      extra_token_indices.numel() > 0;
  const uint8_t* extra_cache_ptr = has_extra ? extra_paged_k_cache_u8.data_ptr<uint8_t>() : nullptr;
  const int* extra_indices_ptr = has_extra ? extra_token_indices.data_ptr<int>() : nullptr;
  const int* extra_topk_lengths_ptr =
      has_extra && extra_topk_lengths.defined() && extra_topk_lengths.numel() > 0
      ? extra_topk_lengths.data_ptr<int>()
      : nullptr;

  dsv4_swa_decode_forward_kernel<scalar_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<scalar_t>(),
      paged_k_cache_u8.data_ptr<uint8_t>(),
      token_indices.data_ptr<int>(),
      topk_lengths_ptr,
      extra_cache_ptr,
      extra_indices_ptr,
      extra_topk_lengths_ptr,
      attn_sink_ptr,
      out.data_ptr<scalar_t>(),
      num_queries,
      heads,
      topk_slots,
      extra_topk_slots,
      page_size,
      bytes_per_page,
      extra_page_size,
      extra_bytes_per_page,
      softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor dsv4_swa_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor paged_k_cache_u8,
    torch::Tensor token_indices,
    torch::Tensor topk_lengths,
    torch::Tensor attn_sink,
    int64_t page_size,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(paged_k_cache_u8.is_cuda(), "paged_k_cache_u8 must be CUDA");
  TORCH_CHECK(token_indices.is_cuda(), "token_indices must be CUDA");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.is_cuda(), "topk_lengths must be CUDA when defined");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.is_cuda(), "attn_sink must be CUDA when defined");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(paged_k_cache_u8.is_contiguous(), "paged_k_cache_u8 must be contiguous");
  TORCH_CHECK(token_indices.is_contiguous(), "token_indices must be contiguous");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.is_contiguous(), "topk_lengths must be contiguous");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.is_contiguous(), "attn_sink must be contiguous");
  TORCH_CHECK(q.dim() == 3, "q must be [num_queries, heads, 512]");
  TORCH_CHECK(q.size(2) == kDsv4HeadDim, "q head_dim must be 512 for DSV4");
  TORCH_CHECK(paged_k_cache_u8.dim() == 2, "paged_k_cache_u8 must be [num_pages, bytes_per_page]");
  TORCH_CHECK(paged_k_cache_u8.scalar_type() == at::kByte, "paged_k_cache_u8 must be uint8");
  TORCH_CHECK(token_indices.dim() == 2, "token_indices must be [num_queries, topk_slots]");
  TORCH_CHECK(token_indices.scalar_type() == at::kInt, "token_indices must be int32");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.scalar_type() == at::kInt,
              "topk_lengths must be int32 when defined");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.scalar_type() == at::kFloat,
              "attn_sink must be float32 when defined");
  TORCH_CHECK(page_size > 0, "page_size must be positive");

  const int num_queries = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int topk_slots = static_cast<int>(token_indices.size(1));
  const int bytes_per_page = static_cast<int>(paged_k_cache_u8.size(1));
  TORCH_CHECK(token_indices.size(0) == num_queries, "token_indices query dimension must match q");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.numel() == num_queries,
              "topk_lengths must have one value per query");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.numel() == heads,
              "attn_sink must have one value per head");
  TORCH_CHECK(bytes_per_page >= page_size * (kDsv4NopeRopeBytes + kDsv4ScaleBytesPerToken),
              "bytes_per_page is too small for DSV4 page layout");

  c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_like(q);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "dsv4_swa_decode_forward_cuda",
      [&] {
        launch_dsv4_swa_decode_forward<scalar_t>(
            q,
            paged_k_cache_u8,
            token_indices,
            topk_lengths,
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            attn_sink,
            out,
            num_queries,
            heads,
            topk_slots,
            0,
            static_cast<int>(page_size),
            bytes_per_page,
            0,
            0,
            static_cast<float>(softmax_scale));
      });
  return out;
}

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
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(paged_k_cache_u8.is_cuda(), "paged_k_cache_u8 must be CUDA");
  TORCH_CHECK(token_indices.is_cuda(), "token_indices must be CUDA");
  TORCH_CHECK(extra_paged_k_cache_u8.is_cuda(), "extra_paged_k_cache_u8 must be CUDA");
  TORCH_CHECK(extra_token_indices.is_cuda(), "extra_token_indices must be CUDA");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.is_cuda(), "topk_lengths must be CUDA when defined");
  TORCH_CHECK(!extra_topk_lengths.defined() || extra_topk_lengths.is_cuda(),
              "extra_topk_lengths must be CUDA when defined");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.is_cuda(), "attn_sink must be CUDA when defined");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(paged_k_cache_u8.is_contiguous(), "paged_k_cache_u8 must be contiguous");
  TORCH_CHECK(token_indices.is_contiguous(), "token_indices must be contiguous");
  TORCH_CHECK(extra_paged_k_cache_u8.is_contiguous(), "extra_paged_k_cache_u8 must be contiguous");
  TORCH_CHECK(extra_token_indices.is_contiguous(), "extra_token_indices must be contiguous");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.is_contiguous(), "topk_lengths must be contiguous");
  TORCH_CHECK(!extra_topk_lengths.defined() || extra_topk_lengths.is_contiguous(),
              "extra_topk_lengths must be contiguous");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.is_contiguous(), "attn_sink must be contiguous");
  TORCH_CHECK(q.dim() == 3, "q must be [num_queries, heads, 512]");
  TORCH_CHECK(q.size(2) == kDsv4HeadDim, "q head_dim must be 512 for DSV4");
  TORCH_CHECK(paged_k_cache_u8.dim() == 2, "paged_k_cache_u8 must be [num_pages, bytes_per_page]");
  TORCH_CHECK(extra_paged_k_cache_u8.dim() == 2,
              "extra_paged_k_cache_u8 must be [num_pages, bytes_per_page]");
  TORCH_CHECK(paged_k_cache_u8.scalar_type() == at::kByte, "paged_k_cache_u8 must be uint8");
  TORCH_CHECK(extra_paged_k_cache_u8.scalar_type() == at::kByte, "extra_paged_k_cache_u8 must be uint8");
  TORCH_CHECK(token_indices.dim() == 2, "token_indices must be [num_queries, topk_slots]");
  TORCH_CHECK(extra_token_indices.dim() == 2,
              "extra_token_indices must be [num_queries, extra_topk_slots]");
  TORCH_CHECK(token_indices.scalar_type() == at::kInt, "token_indices must be int32");
  TORCH_CHECK(extra_token_indices.scalar_type() == at::kInt, "extra_token_indices must be int32");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.scalar_type() == at::kInt,
              "topk_lengths must be int32 when defined");
  TORCH_CHECK(!extra_topk_lengths.defined() || extra_topk_lengths.scalar_type() == at::kInt,
              "extra_topk_lengths must be int32 when defined");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.scalar_type() == at::kFloat,
              "attn_sink must be float32 when defined");
  TORCH_CHECK(page_size > 0, "page_size must be positive");
  TORCH_CHECK(extra_page_size > 0, "extra_page_size must be positive");

  const int num_queries = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int topk_slots = static_cast<int>(token_indices.size(1));
  const int extra_topk_slots = static_cast<int>(extra_token_indices.size(1));
  const int bytes_per_page = static_cast<int>(paged_k_cache_u8.size(1));
  const int extra_bytes_per_page = static_cast<int>(extra_paged_k_cache_u8.size(1));
  TORCH_CHECK(token_indices.size(0) == num_queries, "token_indices query dimension must match q");
  TORCH_CHECK(extra_token_indices.size(0) == num_queries,
              "extra_token_indices query dimension must match q");
  TORCH_CHECK(!topk_lengths.defined() || topk_lengths.numel() == num_queries,
              "topk_lengths must have one value per query");
  TORCH_CHECK(!extra_topk_lengths.defined() || extra_topk_lengths.numel() == num_queries,
              "extra_topk_lengths must have one value per query");
  TORCH_CHECK(!attn_sink.defined() || attn_sink.numel() == heads,
              "attn_sink must have one value per head");
  TORCH_CHECK(bytes_per_page >= page_size * (kDsv4NopeRopeBytes + kDsv4ScaleBytesPerToken),
              "bytes_per_page is too small for DSV4 page layout");
  TORCH_CHECK(extra_bytes_per_page >= extra_page_size * (kDsv4NopeRopeBytes + kDsv4ScaleBytesPerToken),
              "extra_bytes_per_page is too small for DSV4 page layout");

  c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_like(q);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "dsv4_sparse_decode_forward_cuda",
      [&] {
        launch_dsv4_swa_decode_forward<scalar_t>(
            q,
            paged_k_cache_u8,
            token_indices,
            topk_lengths,
            extra_paged_k_cache_u8,
            extra_token_indices,
            extra_topk_lengths,
            attn_sink,
            out,
            num_queries,
            heads,
            topk_slots,
            extra_topk_slots,
            static_cast<int>(page_size),
            bytes_per_page,
            static_cast<int>(extra_page_size),
            extra_bytes_per_page,
            static_cast<float>(softmax_scale));
      });
  return out;
}
