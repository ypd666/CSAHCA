#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, typename index_t>
__global__ void csa_decode_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const index_t* __restrict__ chunk_indices,
    scalar_t* __restrict__ out,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    int top_k,
    int chunk_size) {
  const int bh = blockIdx.x;
  const int b = bh / heads;
  const int h = bh - b * heads;
  const int tid = threadIdx.x;

  extern __shared__ float smem[];
  float* q_s = smem;
  float* acc_s = q_s + head_dim;
  float* red_s = acc_s + head_dim;

  const int q_base = (b * heads + h) * head_dim;
  const int kv_base = ((b * heads + h) * seq_len) * head_dim;
  const int idx_base = (b * heads + h) * top_k;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_s[d] = static_cast<float>(q[q_base + d]);
    acc_s[d] = 0.0f;
  }
  __syncthreads();

  float max_score = -INFINITY;
  float norm = 0.0f;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  for (int i = 0; i < top_k; ++i) {
    const index_t chunk = chunk_indices[idx_base + i];
    if (chunk < 0) {
      continue;
    }
    const int token_start = static_cast<int>(chunk) * chunk_size;
    for (int offset = 0; offset < chunk_size; ++offset) {
      const int token = token_start + offset;
      if (token >= seq_len) {
        break;
      }

      const int token_base = kv_base + token * head_dim;
      float partial = 0.0f;
      for (int d = tid; d < head_dim; d += blockDim.x) {
        partial += q_s[d] * static_cast<float>(k_cache[token_base + d]);
      }

      red_s[tid] = partial;
      __syncthreads();
      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
          red_s[tid] += red_s[tid + stride];
        }
        __syncthreads();
      }

      const float score = red_s[0] * scale;
      const float next_max = fmaxf(max_score, score);
      const float old_scale = __expf(max_score - next_max);
      const float score_scale = __expf(score - next_max);

      for (int d = tid; d < head_dim; d += blockDim.x) {
        acc_s[d] = acc_s[d] * old_scale + score_scale * static_cast<float>(v_cache[token_base + d]);
      }
      norm = norm * old_scale + score_scale;
      max_score = next_max;
      __syncthreads();
    }
  }

  const float inv_norm = norm > 0.0f ? 1.0f / norm : 0.0f;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[q_base + d] = static_cast<scalar_t>(acc_s[d] * inv_norm);
  }
}

template <typename scalar_t, typename index_t>
void launch_csa_decode_forward(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& chunk_indices,
    torch::Tensor& out,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    int top_k,
    int chunk_size) {
  const int block = 256;
  const dim3 grid(batch * heads);
  const size_t shared_bytes = static_cast<size_t>(2 * head_dim + block) * sizeof(float);
  csa_decode_forward_kernel<scalar_t, index_t><<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<scalar_t>(),
      k_cache.data_ptr<scalar_t>(),
      v_cache.data_ptr<scalar_t>(),
      chunk_indices.data_ptr<index_t>(),
      out.data_ptr<scalar_t>(),
      batch,
      heads,
      seq_len,
      head_dim,
      top_k,
      chunk_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor csa_decode_forward_cuda(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor chunk_indices,
    int64_t chunk_size) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be CUDA");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be CUDA");
  TORCH_CHECK(chunk_indices.is_cuda(), "chunk_indices must be CUDA");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
  TORCH_CHECK(chunk_indices.is_contiguous(), "chunk_indices must be contiguous");
  TORCH_CHECK(q.dim() == 3, "q must be [batch, heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must be [batch, heads, seq, head_dim]");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache must match k_cache shape");
  TORCH_CHECK(chunk_indices.dim() == 3, "chunk_indices must be [batch, heads, top_k]");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type(), "q and k_cache dtype must match");
  TORCH_CHECK(q.scalar_type() == v_cache.scalar_type(), "q and v_cache dtype must match");
  TORCH_CHECK(chunk_indices.scalar_type() == at::kInt || chunk_indices.scalar_type() == at::kLong,
              "chunk_indices must be int32 or int64");

  const int batch = static_cast<int>(q.size(0));
  const int heads = static_cast<int>(q.size(1));
  const int head_dim = static_cast<int>(q.size(2));
  const int seq_len = static_cast<int>(k_cache.size(2));
  const int top_k = static_cast<int>(chunk_indices.size(2));
  TORCH_CHECK(k_cache.size(0) == batch && k_cache.size(1) == heads && k_cache.size(3) == head_dim,
              "KV cache shape must match q");
  TORCH_CHECK(chunk_indices.size(0) == batch && chunk_indices.size(1) == heads,
              "chunk_indices batch/head dimensions must match q");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
  TORCH_CHECK(head_dim <= 1024, "head_dim > 1024 is not supported by the v1 kernel");

  c10::cuda::CUDAGuard device_guard(q.device());
  auto out = torch::empty_like(q);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      q.scalar_type(),
      "csa_decode_forward_cuda",
      [&] {
        if (chunk_indices.scalar_type() == at::kInt) {
          launch_csa_decode_forward<scalar_t, int>(
              q, k_cache, v_cache, chunk_indices, out, batch, heads, seq_len, head_dim, top_k,
              static_cast<int>(chunk_size));
        } else {
          launch_csa_decode_forward<scalar_t, int64_t>(
              q, k_cache, v_cache, chunk_indices, out, batch, heads, seq_len, head_dim, top_k,
              static_cast<int>(chunk_size));
        }
      });
  return out;
}

