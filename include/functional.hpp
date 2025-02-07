#pragma once
#include "operators.hpp"
#include "utils.hpp"

inline Tensor layer_norm_1d(Tensor &input, const TensorIntf &weight,
                            const TensorIntf &bias, float eps = 1e-5f) {
  const auto shape = input.shape();
  size_t dim1 = shape.size() - 2, dim2 = shape.size() - 1;

  input.permute(dim1, dim2);
  Tensor mean = reduce_mean<float, float, float>(input, dim1);
  if (mean.n_elems() == 1) {
    input = sub_<float>(input, *(mean.data<float>()));
  } else {
    input = sub_<float>( // inplace
        input, mean);
  }
  Tensor x_shift2 = mul<float, float>(input, input);
  Tensor var =
      reduce_mean<float, float, float>(x_shift2, x_shift2.shape().size() - 2);
  float var_scalar = *var.data<float>() + eps;
  var_scalar = 1.0f / std::sqrt(var_scalar);
  input = mul_<float>(input, var_scalar);
  input.permute(dim1, dim2);
  input = mul_<float>(input, weight);
  input = add_<float>(input, bias);
  spdlog::debug("end funcall layernorm {}", to_string(input.shape()));

  return input;
};

inline Tensor layer_norm(Tensor &input, const TensorIntf &weight,
                         const TensorIntf &bias, float eps = 1e-5f) {
  spdlog::debug("funcall layernorm input {} weight {} bias {}",
                to_string(input.shape()), to_string(weight.shape()),
                to_string(bias.shape()));
  const auto shape = input.shape();
  assert(shape.size() >= 2);
  if (shape[shape.size() - 1] == input.n_elems())
    return layer_norm_1d(input, weight, bias, eps);

  Tensor tmp = transpose<float>(input, shape.size() - 2,
                                shape.size() - 1); // shouldn't be necessary
  Tensor mean = reduce_mean<float, float, float>(tmp, shape.size() - 2);
  Tensor x_shift = sub_<float>( // inplace = tmp
      tmp, mean);
  Tensor x_shift2 = mul<float, float>(x_shift, x_shift);
  Tensor var =
      reduce_mean<float, float, float>(x_shift2, x_shift2.shape().size() - 2);
  var = add_<float>(var, eps);
  var = elementwise_unary<float>(
      var, [](const float &t) { return 1.0f / std::sqrt(t); }, "Rsqrt");
  Tensor den = mul<float, float>(x_shift, var);
  Tensor res =
      transpose<float>(den, den.shape().size() - 2, den.shape().size() - 1);
  res = mul_<float>(res, weight);
  res = add_<float>(res, bias);
  spdlog::debug("end funcall layernorm {}", to_string(res.shape()));

  return res;
}

namespace {
// inplace
inline Tensor softmax_impl(Tensor &input, size_t dim) {
  Tensor max = reduce_max<float, float, float>(input, dim);
  if (max.n_elems() == 1) {
    input = sub_<float>(input, *(max.data<float>()));
  } else {
    input = sub_<float>(input, max);
  }
  Tensor exp = elementwise_unary<float>( // inplace
      input, [](const float &x) { return std::exp(x); }, "Exp");
  Tensor sum = reduce_sum<float, float, float>(exp, dim);
  if (sum.n_elems() == 1) {
    float sum_inv = 1.0f / *(sum.data<float>());
    return mul_<float>(exp, sum_inv);
  }
  Tensor sum_inv = elementwise_unary<float>(
      sum, [](const float &t) { return 1.0f / t; }, "Recp");
  return mul_<float>(exp, sum_inv);
}
} // namespace

// inplace
inline Tensor softmax(Tensor &input, size_t dim) {
  spdlog::debug("funcall softmax input {} dim {}", to_string(input.shape()),
                dim);
  scoped_debug info{"end funcall softmax"};
  const auto shape = input.shape();
  assert(dim < shape.size());
  if (dim == shape.size() - 1) { // over last dim
    assert(shape.size() >= 2);
    size_t dim1 = shape.size() - 2, dim2 = shape.size() - 1;
    if (std::abs(int(dim1 - dim2)) == 1 &&
        (shape[dim1] == 1 ||
         shape[dim2] == 1)) { // can use reshape instead of transpose
      auto new_shape = shape;
      new_shape[dim1] = shape[dim2];
      new_shape[dim2] = shape[dim1];
      input.reshape(new_shape);
      input = softmax_impl(input, dim - 1);
      input.reshape(shape);
      return input;
    }
    Tensor tmp = transpose<float>(input, shape.size() - 2, shape.size() - 1);
    tmp = softmax_impl(tmp, dim - 1);
    return transpose<float>(tmp, shape.size() - 2, shape.size() - 1);
  }
  return softmax_impl(input, dim);
}

// needs to add bias
inline Tensor batch_matmul(const Tensor &lhs, const Tensor &rhs,
                           uint32_t batch_dims, bool transposed) {
  const auto &shape_in = lhs.shape();
  const auto &shape_weights = rhs.shape();
  assert(shape_in.size() - batch_dims == 2);
  assert(shape_weights.size() == shape_in.size());
  auto shape_out = shape_in;
  shape_out[batch_dims] = shape_in[batch_dims];
  shape_out[batch_dims + 1] =
      transposed ? shape_weights[batch_dims] : shape_weights[batch_dims + 1];
  size_t out_offset = 1, in_offset = 1, weights_offset = 1;
  for (size_t i = batch_dims; i < shape_in.size(); ++i) {
    out_offset *= shape_out[i];
    in_offset *= shape_in[i];
    weights_offset *= shape_weights[i];
  }
  size_t outer_loop = 1;
  for (size_t i = 0; i < batch_dims; ++i) {
    outer_loop *= shape_in[i];
    assert(shape_in[i] == shape_weights[i]);
  }
  spdlog::debug("funcall batch_matmul input {} batch {} weight {} batch_dims "
                "{} transposed {}",
                lhs.id(), outer_loop, rhs.id(), batch_dims, transposed);
  const size_t rows = shape_out[batch_dims];
  const size_t dots = shape_in[batch_dims + 1];
  const size_t cols = shape_out[batch_dims + 1];
  const std::vector<size_t> matmul_in{rows, dots};
  const std::vector<size_t> matmul_out{rows, cols};
  const size_t w_dim1 = transposed ? cols : dots;
  const size_t w_dim2 = transposed ? dots : cols;
  const std::vector<size_t> matmul_weights{w_dim1, w_dim2};
  Tensor output{lhs.dtype(), shape_out};
  float *out_ptr = output.data<float>();
  const float *in_ptr = lhs.cdata<float>();
  const float *weight_ptr = rhs.cdata<float>();
  // should use matmul, but it would require to create tensors for each batch
  // and the parallel loop is probably more efficient that way.
  // NISA version would need different tiles for each matmul.
  for (size_t b = 0; b < outer_loop; ++b)
    spdlog::trace("opcall matmul input {:x} {} weights {:x} {} transposed {} "
                  "output {:x} {}",
                  lhs.unique_id() + b * in_offset, matmul_in,
                  rhs.unique_id() + b * weights_offset, matmul_weights,
                  transposed, output.unique_id() + b * out_offset, matmul_out);
#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < outer_loop; ++b) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        float acc = 0.0;
        if (transposed) {
          for (size_t k = 0; k < dots; ++k) {
            acc += in_ptr[b * in_offset + i * dots + k] *
                   weight_ptr[b * weights_offset + j * dots + k];
          }
        } else {
          for (size_t k = 0; k < dots; ++k) {
            acc += in_ptr[b * in_offset + i * dots + k] *
                   weight_ptr[b * weights_offset + k * cols + j];
          }
        }
        out_ptr[b * out_offset + i * cols + j] = acc;
      }
    }
  }
  spdlog::debug("end funcall batch_matmul");
  return output;
}

// multihead
inline Tensor scaled_dot_product_attention(Tensor &input, uint32_t embed,
                                           uint32_t heads, const Tensor *mask,
                                           Tensor *k_cache, Tensor *v_cache) {
  spdlog::debug("funcall scaled_dot_product_attention input {} embed {} heads "
                "{} mask {}, k_cache {} v_cache {}",
                to_string(input.shape()), embed, heads, mask != nullptr,
                k_cache != nullptr, v_cache != nullptr);
  scoped_debug info{"end funcall scaled_dot_product_attention"};
  const auto shape = input.shape(); // [B, T, embed * 3]
  const size_t head_size = embed / heads;
  assert(shape.size() == 3);
  assert(shape[2] == 3 * embed); // Q K V are concatenated.
  size_t B = shape[0], T = shape[1], C3 = shape[2], C = C3 / 3;
  assert(C3 == 3 * heads * head_size);
  std::vector<size_t> qkv_shape{B, T, heads, head_size};
  std::vector<Tensor> qkv = input.split(C, 2);
  for (auto &t : qkv)
    t.reshape(qkv_shape);
  Tensor q = qkv[0];
  Tensor k = qkv[1];
  Tensor v = qkv[2];
  q = transpose<float>(q, 1, 2); // YZ [B, H, T, Hs]
  k = transpose<float>(k, 1, 2); // YZ [B, H, T, Hs]
  if (k_cache)
    k_cache->insert(k, {0, 0, 0, 0});
  v = transpose<float>(v, 1, 2); // YZ
  if (v_cache)
    v_cache->insert(v, {0, 0, 0, 0});

  Tensor att = batch_matmul(q, k, 2, true); // [B, H, T, T]
  float scaler = 1.0 / std::sqrt(head_size);
  mul_<float>(att, scaler);
  if (mask) {
    Tensor sliced_mask = mask->subtile({T, T}, {0, 0});
    // add_<float>(att, *mask); // inplace
    add_<float>(att, sliced_mask); // inplace
  }
  att = softmax(att, 3);
  Tensor y = batch_matmul(att, v, 2, false);
  Tensor res = transpose<float>(y, 1, 2); // YZ
  res.reshape({B, T, C});
  return res;
}

inline Tensor cached_scaled_dot_product_attention(
    Tensor &input, // [B, 1, C * 3]
    uint32_t embed, uint32_t heads,
    const Tensor *mask, // [T, T]
    Tensor &k_cache,    // [B, H, T, Hs]  ? // [B, H, Hs, T] not possible ...
    Tensor &v_cache,    // [B, H, T, Hs]
    uint32_t t) {
  if (t == 0) { // prefill mode
    return scaled_dot_product_attention(input, embed, heads, mask, &k_cache,
                                        &v_cache);
  }
  spdlog::debug(
      "funcall cached_scaled_dot_product_attention input {} embed {} heads "
      "{} mask {}, k_cache {} v_cache {}",
      to_string(input.shape()), embed, heads, mask != nullptr,
      to_string(k_cache.shape()), to_string(v_cache.shape()));
  scoped_debug info{"end funcall cached_scaled_dot_product_attention"};
  const auto shape = input.shape(); // [B, T, embed * 3] T = 1
  const size_t head_size = embed / heads;
  assert(shape.size() == 3);
  assert(shape[2] == 3 * embed); // Q K V are concatenated.
  size_t B = shape[0], T = shape[1], C3 = shape[2], C = C3 / 3;
  assert(B == 1); // otherwise we should use matmul
  assert(C3 == 3 * heads * head_size);
  assert(T == 1);
  std::vector<size_t> qkv_shape{B, heads, 1, head_size}; // T = 1
  std::vector<Tensor> qkv = input.split(C, 2);
  for (auto &t : qkv)
    t.reshape(qkv_shape);
  Tensor q = qkv[0]; // [B, H, 1, Hs]
  Tensor k = qkv[1];
  Tensor v = qkv[2];
  Tensor y{safetensors::kFLOAT32, {B, heads, 1, head_size}};
  const float scaler = 1.0 / std::sqrt(head_size);
  // #pragma omp parallel for
  for (size_t i = 0; i < heads; ++i) {
    Tensor q_i = q.subtile({B, 1, 1, head_size}, {0, i, 0, 0}); // [B, 1, 1, Hs]
    q_i.reshape({head_size});
    Tensor k_i = k.subtile({B, 1, 1, head_size}, {0, i, 0, 0});
    Tensor v_i = v.subtile({B, 1, 1, head_size}, {0, i, 0, 0});
    k_cache.insert(k_i, {0, i, t, 0});
    v_cache.insert(v_i, {0, i, t, 0});
    Tensor k_i_full = k_cache.subtile({B, 1, t + 1, head_size},
                                      {0, i, 0, 0}); // [B, 1, T, Hs]
    k_i_full = transpose<float>(k_i_full, 2, 3);     // XY [B, 1, Hs, T]
    k_i_full.reshape({head_size, t + 1});
    Tensor att_i = gemv<float, float, float>(q_i, k_i_full); // B, 1, 1, T
    mul_<float>(att_i, scaler);
    if (mask) {
      Tensor sliced_mask = mask->subtile({1, t + 1}, {t, 0});
      add_<float>(att_i, sliced_mask); // inplace
    }
    att_i = softmax(att_i, 1); // [B, H, 1, T]
    Tensor v_i_full = v_cache.subtile({B, 1, t + 1, head_size},
                                      {0, i, 0, 0}); // [B, 1, T, Hs]
    v_i_full.reshape({t + 1, head_size});
    att_i.reshape({t + 1});
    Tensor y_i = gemv<float, float, float>(att_i, v_i_full);
    y_i.reshape({B, 1, 1, head_size});
    y.insert(y_i, {0, i, 0, 0});
  }
  y.reshape({B, 1, C});
  return y;
}

inline Tensor multinomial(Tensor &input, size_t num_samples) {
  const auto &shape = input.shape();
  assert(shape.size() <= 2); // constraint from pytorch
  // compute cumulative sum of input
  Tensor cumsum = cumsum_rvv<float, float>(input);
  return sample_rvv<float, uint32_t>(cumsum, num_samples, 1337);
}
