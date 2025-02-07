#pragma once

#include "tensor.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>

template <typename T1, typename T2>
Tensor elementwise_unary(const TensorIntf &src,
                         std::function<T1(const T2 &)> op,
                         const std::string &name) {
  WARN_IF(src.n_elems() == 1, "elementwise_unary {} should be a scalar op",
          name);
  Tensor out{DataTypeTraits<T1>::type, src.shape()};
  spdlog::trace("opcall elementwise_unary {} input {} output {}", name,
                src.id(), out.id());
  T1 *out_ptr = out.data<T1>();
  const T2 *src_ptr = src.cdata<T2>();
#pragma omp parallel for
  for (size_t i = 0; i < src.n_elems(); ++i) {
    out_ptr[i] = op(src_ptr[i]);
  }
  return out;
}

// inplace version
template <typename T>
Tensor elementwise_unary(Tensor &src, std::function<T(const T &)> op,
                         const std::string &name) {
  spdlog::trace("opcall elementwise_unary inplace {} input {} output {}", name,
                src.id(), src.id());
  WARN_IF(src.n_elems() == 1, "elementwise_unary {} should be a scalar op",
          name);
  T *out_ptr = src.data<T>();
#pragma omp parallel for
  for (size_t i = 0; i < src.n_elems(); ++i) {
    out_ptr[i] = op(out_ptr[i]);
  }
  return src;
}

inline std::vector<size_t> align_shape(const std::vector<size_t> &s) {
  std::vector<size_t> res(4, 1);
  int offset = 4 - s.size();
  for (size_t i = 0; i < s.size(); ++i)
    res[offset + i] = s[i];
  return res;
}

template <typename T1, typename T2>
Tensor elementwise_binary(const TensorIntf &src1, const TensorIntf &src2,
                          std::function<T1(const T2 &, const T2 &)> op,
                          const std::string &name) {
  WARN_IF(src1.n_elems() == 1, "elementwise_binary {} should be a scalar op",
          name);
  WARN_IF(src2.n_elems() == 1, "elementwise_binary {} should be an imm op",
          name);
  assert(src1.dtype() == src2.dtype());
  std::vector<size_t> shape1 = align_shape(src1.shape());
  std::vector<size_t> shape2 = align_shape(src2.shape());
  std::vector<size_t> maxs = shape1;
  const T2 *in1_ptr = src1.cdata<T2>();
  const T2 *in2_ptr = src2.cdata<T2>();
  for (size_t i = 0; i < shape1.size(); ++i) {
    assert(shape1[i] == shape2[i] || shape2[i] == 1);
  }
  Tensor out{DataTypeTraits<T1>::type, shape1};
  T1 *out_ptr = out.data<T1>();
  TensorIndex idx1{shape1}, idx2{shape2}, outidx{maxs};
#pragma omp parallel for collapse(3)
  for (size_t i = 0; i < maxs[0]; ++i) {
    for (size_t j = 0; j < maxs[1]; ++j) {
      for (size_t k = 0; k < maxs[2]; ++k) {
        for (size_t l = 0; l < maxs[3]; ++l) {
          size_t i1 = std::min(i, shape1[0] - 1),
                 i2 = std::min(i, shape2[0] - 1);
          size_t j1 = std::min(j, shape1[1] - 1),
                 j2 = std::min(j, shape2[1] - 1);
          size_t k1 = std::min(k, shape1[2] - 1),
                 k2 = std::min(k, shape2[2] - 1);
          size_t l1 = std::min(l, shape1[3] - 1),
                 l2 = std::min(l, shape2[3] - 1);
          size_t out_index = outidx.index({i, j, k, l});
          size_t in1_index = idx1.index({i1, j1, k1, l1});
          size_t in2_index = idx2.index({i2, j2, k2, l2});
          T1 res = op(in1_ptr[in1_index], in2_ptr[in2_index]);
          out_ptr[out_index] = res;
        }
      }
    }
  }
  out.reshape(src1.shape());
  spdlog::trace("opcall elementwise_binary {} p1 {} p2 {} output {}", name,
                src1.id(), src2.id(), out.id());
  return out;
}

// inplace version
template <typename T>
Tensor elementwise_binary(Tensor &src1, const TensorIntf &src2,
                          std::function<T(const T &, const T &)> op,
                          const std::string &name) {
  WARN_IF(src1.n_elems() == 1,
          "elementwise_binary inplace {} should be a scalar op", name);
  WARN_IF(src2.n_elems() == 1,
          "elementwise_binary inplace {} should be an imm op", name);
  assert(src1.dtype() == src2.dtype());
  std::vector<size_t> shape1 = align_shape(src1.shape());
  std::vector<size_t> shape2 = align_shape(src2.shape());
  T *in1_ptr = src1.data<T>();
  const T *in2_ptr = src2.cdata<T>();
  for (size_t i = 0; i < shape1.size(); ++i) {
    assert(shape1[i] == shape2[i] || shape2[i] == 1);
  }
  TensorIndex idx1{shape1}, idx2{shape2};
#pragma omp parallel for collapse(4)
  for (size_t i = 0; i < shape1[0]; ++i) {
    for (size_t j = 0; j < shape1[1]; ++j) {
      for (size_t k = 0; k < shape1[2]; ++k) {
        for (size_t l = 0; l < shape1[3]; ++l) {
          size_t i2 = std::min(i, shape2[0] - 1);
          size_t j2 = std::min(j, shape2[1] - 1);
          size_t k2 = std::min(k, shape2[2] - 1);
          size_t l2 = std::min(l, shape2[3] - 1);
          size_t out_index = idx1.index({i, j, k, l});
          size_t in2_index = idx2.index({i2, j2, k2, l2});
          T res = op(in1_ptr[out_index], in2_ptr[in2_index]);
          in1_ptr[out_index] = res;
        }
      }
    }
  }
  spdlog::trace("opcall elementwise_binary inplace {} p1 {} p2 {} output {}",
                name, src1.id(), src2.id(), src1.id());
  return src1;
}

template <typename T1, typename T2>
Tensor elementwise_binary(const TensorIntf &src, const T2 &imm,
                          std::function<T1(const T2 &, const T2 &)> op,
                          const std::string &name) {
  WARN_IF(src.n_elems() == 1, "elementwise_binary {} should be a scalar op",
          name);
  const T2 *in_ptr = src.cdata<T2>();
  Tensor out{DataTypeTraits<T1>::type, src.shape()};
  spdlog::trace("opcall elementwise_binary {} p {} imm {} output {}", name,
                src.id(), imm, out.id());
  T1 *out_ptr = out.data<T1>();
#pragma omp parallel for
  for (size_t i = 0; i < src.n_elems(); ++i) {
    out_ptr[i] = op(in_ptr[i], imm);
  }
  return out;
}

// inplace version
template <typename T>
Tensor elementwise_binary(Tensor &src, const T &imm,
                          std::function<T(const T &, const T &)> op,
                          const std::string &name) {
  spdlog::trace("opcall elementwise_binary inplace {} p {} imm {} output {}",
                name, src.id(), imm, src.id());
  WARN_IF(src.n_elems() == 1, "elementwise_binary {} should be a scalar op",
          name);
  T *in_ptr = src.data<T>();
#pragma omp parallel for
  for (size_t i = 0; i < src.n_elems(); ++i) {
    in_ptr[i] = op(in_ptr[i], imm);
  }
  return src;
}

template <typename T1, typename T2>
Tensor mul(const TensorIntf &src1, const TensorIntf &src2) {
  return elementwise_binary<T1, T2>(
      src1, src2, [](const T1 &t1, const T2 &t2) { return t1 * t2; }, "Mul");
}

template <typename T> Tensor mul_(Tensor &src1, const TensorIntf &src2) {
  return elementwise_binary<T>(
      src1, src2, [](const T &t1, const T &t2) { return t1 * t2; }, "Mul");
}

template <typename T1, typename T2>
Tensor mul(const TensorIntf &src, const T2 &imm) {
  return elementwise_binary<T1, T2>(
      src, imm, [](const T1 &t1, const T2 &t2) { return t1 * t2; }, "Mul");
}

template <typename T> Tensor mul_(Tensor &src, const T &imm) {
  return elementwise_binary<T>(
      src, imm, [](const T &t1, const T &t2) { return t1 * t2; }, "Mul");
}

template <typename T1, typename T2>
Tensor add(const TensorIntf &src1, const TensorIntf &src2) {
  return elementwise_binary<T1, T2>(
      src1, src2, [](const T1 &t1, const T2 &t2) { return t1 + t2; }, "Add");
}

template <typename T> Tensor add_(Tensor &src1, const TensorIntf &src2) {
  return elementwise_binary<T>(
      src1, src2, [](const T &t1, const T &t2) { return t1 + t2; }, "Add");
}

template <typename T1, typename T2>
Tensor add(const TensorIntf &src, const T2 &imm) {
  return elementwise_binary<T1, T2>(
      src, imm, [](const T1 &t1, const T2 &t2) { return t1 + t2; }, "Add");
}

template <typename T> Tensor add_(Tensor &src, const T &imm) {
  return elementwise_binary<T>(
      src, imm, [](const T &t1, const T &t2) { return t1 + t2; }, "Add");
}

template <typename T1, typename T2>
Tensor sub(const TensorIntf &src, const T2 &imm) {
  return elementwise_binary<T1, T2>(
      src, imm, [](const T1 &t1, const T2 &t2) { return t1 - t2; }, "Sub");
}

template <typename T> Tensor sub_(Tensor &src1, const TensorIntf &src2) {
  return elementwise_binary<T>(
      src1, src2, [](const T &t1, const T &t2) { return t1 - t2; }, "Sub");
}

template <typename T1, typename T2>
Tensor sub(const TensorIntf &src1, const TensorIntf &src2) {
  return elementwise_binary<T1, T2>(
      src1, src2, [](const T1 &t1, const T2 &t2) { return t1 - t2; }, "Sub");
}

template <typename T> Tensor sub_(Tensor &src, const T &imm) {
  return elementwise_binary<T>(
      src, imm, [](const T &t1, const T &t2) { return t1 - t2; }, "Sub");
}

template <typename T>
Tensor transpose(const TensorIntf &input, size_t dim1, size_t dim2) {
  const auto input_shape = input.shape();
  assert(dim1 != dim2);
  assert(dim1 < input_shape.size());
  assert(dim2 < input_shape.size());
  auto output_shape = input_shape;
  output_shape[dim1] = input_shape[dim2];
  output_shape[dim2] = input_shape[dim1];
  WARN_IF(std::abs(int(dim1 - dim2)) == 1 &&
              (input_shape[dim1] == 1 || input_shape[dim2] == 1),
          "Transpose could be a reshape {} dim1 {} dim2 {}",
          to_string(input.shape()), dim1, dim2);
  std::vector<size_t> shape1 = align_shape(input_shape);
  std::vector<size_t> shape2 = align_shape(output_shape);
  size_t offset_dim1 = dim1 + shape1.size() - input_shape.size();
  size_t offset_dim2 = dim2 + shape1.size() - input_shape.size();
  Tensor out{DataTypeTraits<T>::type, output_shape};
  spdlog::trace("opcall transpose input {} dim1 {} dim2 {} output {}",
                input.id(), dim1, dim2, out.id());
  T *out_ptr = out.data<T>();
  const T *in_ptr = input.cdata<T>();
  TensorIndex idx1{shape1}, idx2{shape2};
#pragma omp parallel for collapse(4)
  for (size_t i = 0; i < shape1[0]; ++i) {
    for (size_t j = 0; j < shape1[1]; ++j) {
      for (size_t k = 0; k < shape1[2]; ++k) {
        for (size_t l = 0; l < shape1[3]; ++l) {
          std::vector<size_t> index{i, j, k, l};
          auto out_index = index;
          out_index[offset_dim1] = index[offset_dim2];
          out_index[offset_dim2] = index[offset_dim1];
          size_t in_index = idx1.index(index);
          size_t output_index = idx2.index(out_index);
          out_ptr[output_index] = in_ptr[in_index];
        }
      }
    }
  }
  return out;
}

template <typename Input, typename Output, typename Accum>
Tensor matmul(const TensorIntf &input, const TensorIntf &weights,
              const Accum *bias, bool transposed) {
  assert(bias);
  const auto &shape_in = input.shape();
  const auto &shape_weights = weights.shape();
  assert(shape_in.size() == 2);
  assert(shape_weights.size() == 2);
  const size_t rows = shape_in[0];
  const size_t dots = shape_in[1];
  const size_t cols = transposed ? shape_weights[0] : shape_weights[1];
  if (transposed) {
    assert(dots == shape_weights[1]);
  } else {
    assert(dots == shape_weights[0]);
  }
  WARN_IF(rows == 1, "Matmul should be a gemv input {} weights {}",
          to_string(input.shape()), to_string(weights.shape()));
  std::vector<size_t> shape_out{rows, cols};
  Tensor out{DataTypeTraits<Output>::type, shape_out};
  Output *out_ptr = out.data<Output>();
  const Input *in_ptr = input.cdata<Input>();
  const Input *weight_ptr = weights.cdata<Input>();
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      Accum acc = bias[j];
      if (transposed) {
        for (size_t k = 0; k < dots; ++k) {
          acc += in_ptr[i * dots + k] * weight_ptr[j * dots + k];
        }
      } else {
        for (size_t k = 0; k < dots; ++k) {
          acc += in_ptr[i * dots + k] * weight_ptr[k * cols + j];
        }
      }
      out_ptr[i * cols + j] = acc;
    }
  }
  return out;
}

template <typename Input, typename Output, typename Accum>
Tensor matmul(const TensorIntf &input, const TensorIntf &weights,
              const TensorIntf &bias, bool transposed) {
  Tensor res = matmul<Input, Output, Accum>(input, weights, bias.cdata<Accum>(),
                                            transposed);
  spdlog::trace(
      "opcall matmul input {} weights {} bias {} transposed {} output {}",
      input.id(), weights.id(), bias.id(), transposed, res.id());
  return res;
}

template <typename Input, typename Output, typename Accum>
Tensor matmul(const TensorIntf &input, const TensorIntf &weights,
              bool transposed) {
  const auto &shape_weights = weights.shape();
  assert(shape_weights.size() == 2);
  const size_t cols = transposed ? shape_weights[0] : shape_weights[1];
  const std::vector<Accum> bias_data(cols, 0);
  Tensor res = matmul<Input, Output, Accum>(input, weights, bias_data.data(),
                                            transposed);
  spdlog::trace("opcall matmul input {} weights {} transposed {} output {}",
                input.id(), weights.id(), transposed, res.id());
  return res;
}

template <typename Input, typename Output, typename Accum>
Tensor gemv(const TensorIntf &input, const TensorIntf &weights,
            const Accum *bias) {
  const auto &shape_in = input.shape();
  const auto &shape_weights = weights.shape();
  assert(shape_in.size() <= 2);
  if (shape_in.size() == 2)
    assert(shape_in[0] == 1);
  assert(shape_weights.size() == 2);
  const size_t dots = shape_in.size() == 2 ? shape_in[1] : shape_in[0];
  const size_t cols = shape_weights[1];
  assert(dots == shape_weights[0]);
  std::vector<size_t> shape_out{1, cols};
  Tensor out{DataTypeTraits<Output>::type, shape_out};
  Output *out_ptr = out.data<Output>();
  const Input *in_ptr = input.cdata<Input>();
  const Input *weight_ptr = weights.cdata<Input>();
#pragma omp parallel for
  for (size_t j = 0; j < cols; ++j) {
    Accum acc = bias ? bias[j] : 0;
    for (size_t k = 0; k < dots; ++k) {
      acc += in_ptr[k] * weight_ptr[k * cols + j];
    }
    out_ptr[j] = acc;
  }
  return out;
}

template <typename Input, typename Output, typename Accum>
Tensor gemv(const TensorIntf &input, const TensorIntf &weights) {
  Tensor res = gemv<Input, Output, Accum>(input, weights, nullptr);
  spdlog::trace("opcall gemv input {} weights {} no bias output {}", input.id(),
                weights.id(), res.id());
  return res;
}

template <typename Input, typename Output, typename Accum>
Tensor gemv(const TensorIntf &input, const TensorIntf &weights,
            const TensorIntf &bias) {
  Tensor res = gemv<Input, Output, Accum>(input, weights, bias.cdata<Accum>());
  spdlog::trace("opcall gemv input {} weights {} bias {} output {}", input.id(),
                weights.id(), bias.id(), res.id());
  return res;
}

template <typename Input, typename Index>
Tensor gather(const TensorIntf &data, const TensorIntf &indices) {
  const auto shape_data = data.shape();
  const auto shape_indices = indices.shape();
  assert(shape_indices.size() <= 2); // for now
  assert(shape_indices.size() <= shape_data.size());
  std::vector<size_t> shape_out{shape_data};
  shape_out[0] = shape_indices[0];
  if (shape_indices.size() == 2)
    shape_out[1] = 1;
  Tensor out{DataTypeTraits<Input>::type, shape_out};
  size_t block_size = 1;
  for (size_t i = shape_indices.size(); i < shape_out.size(); ++i)
    block_size *= shape_out[i];
  // copy each index to out
  spdlog::trace("opcall gather input {} elems {} output {}", data.id(),
                block_size * shape_indices[0], out.id());
  const Input *in_ptr = data.cdata<Input>();
  const Index *idx_ptr = indices.cdata<Index>();
  Input *out_ptr = out.data<Input>();
  if (shape_indices.size() == 1) {
    for (size_t i = 0; i < shape_indices[0]; ++i) {
      assert(idx_ptr[i] < shape_data[0]);
      size_t offset = block_size * idx_ptr[i];
      std::memcpy(&out_ptr[block_size * i], &in_ptr[offset],
                  sizeof(Input) * block_size);
    }
  } else {
    size_t i_stride = shape_data[1];
    for (size_t i = 0; i < shape_indices[0]; ++i) {
      Index idx1 = idx_ptr[i * shape_indices[1]];
      Index idx2 = idx_ptr[i * shape_indices[1] + 1];
      assert(idx1 < shape_data[0]);
      assert(idx2 < shape_data[1]);
      size_t offset = block_size * (idx1 * i_stride + idx2);
      std::memcpy(&out_ptr[block_size * i], &in_ptr[offset],
                  sizeof(Input) * block_size);
    }
  }
  return out;
}

template <typename Input, typename Index>
Tensor embeddings_gather(const TensorIntf &data, const TensorIntf &indices) {
  const auto shape_data = data.shape();
  const auto shape_indices = indices.shape();
  assert(shape_indices.size() <= 2); // for now
  assert(shape_data.size() == 2);
  std::vector<size_t> shape_out{shape_data};
  if (shape_indices.size() == 1)
    shape_out[0] = shape_indices[0];
  else {
    shape_out.insert(shape_out.begin(), shape_indices[0]);
    shape_out[1] = shape_indices[1];
  }
  Tensor out{DataTypeTraits<Input>::type, shape_out};
  size_t block_size = shape_data[1];
  // copy each index to out
  const Input *in_ptr = data.cdata<Input>();
  const Index *idx_ptr = indices.cdata<Index>();
  Input *out_ptr = out.data<Input>();
  if (shape_indices.size() == 1) {
    spdlog::trace("opcall embeddings_gather input {} elems {} output {}",
                  data.id(), block_size * shape_indices[0], out.id());
    for (size_t i = 0; i < shape_indices[0]; ++i) {
      assert(idx_ptr[i] < shape_data[0]);
      size_t offset = block_size * idx_ptr[i];
      std::memcpy(&out_ptr[block_size * i], &in_ptr[offset],
                  sizeof(Input) * block_size);
    }
  } else {
    size_t i_stride = shape_indices[1];
    spdlog::trace("opcall embeddings_gather input {} elems {} output {}",
                  data.id(), block_size * shape_indices[0] * shape_indices[1],
                  out.id());
    for (size_t i = 0; i < shape_indices[0]; ++i) { // batch dim
      for (size_t j = 0; j < shape_indices[1]; ++j) {
        Index idx = idx_ptr[i * i_stride + j];
        assert(idx < shape_data[0]);
        size_t offset = block_size * idx;
        std::memcpy(&out_ptr[block_size * (i * i_stride + j)], &in_ptr[offset],
                    sizeof(Input) * block_size);
      }
    }
  }
  return out;
}

// gather everything from 0 to size-1.
// Useful for positional embeddings in gpt2
template <typename Input>
Tensor gather_range(const TensorIntf &data, uint32_t size) {
  const auto shape_data = data.shape();
  std::vector<size_t> shape_out{shape_data};
  shape_out[0] = size;
  Tensor out{DataTypeTraits<Input>::type, shape_out};
  size_t block_size = 1;
  for (size_t i = 1; i < shape_out.size(); ++i)
    block_size *= shape_out[i];
  spdlog::trace("opcall gather_range input {} elems {} output {}", data.id(),
                block_size * size, out.id());
  // copy each index to out
  const Input *in_ptr = data.cdata<Input>();
  Input *out_ptr = out.data<Input>();
  std::memcpy(out_ptr, in_ptr, sizeof(Input) * block_size * size);
  return out;
}

template <typename Input, typename Output, typename Accum>
Tensor reduce(const TensorIntf &input, size_t dim, const Accum &init_value,
              std::function<Accum(const Accum &, const Input &)> op,
              const std::string &name,
              std::function<Output(const Accum &, size_t)> end_func) {
  auto dtout = DataTypeTraits<Output>::type;
  auto shape_in = input.shape();
  //  assert(dim < shape_in.size() - 1); // last dim is not allowed
  std::vector<size_t> shape_out{shape_in};
  shape_out[dim] = 1;
  Tensor out{DataTypeTraits<Output>::type, shape_out};
  spdlog::trace("opcall reduce {} input {} dim {} output {}", name, input.id(),
                dim, out.id());
  Output *out_ptr = out.data<Output>();
  const Input *src_ptr = input.cdata<Input>();
  size_t pre_dim = 1;
  for (size_t i = 0; i < dim; ++i)
    pre_dim *= shape_in[i];
  size_t post_stride = 1;
  for (size_t i = dim + 1; i < shape_in.size(); ++i)
    post_stride *= shape_in[i];
  size_t pre_stride = post_stride * shape_in[dim];

#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < pre_dim; ++i) {
    for (size_t j = 0; j < post_stride; ++j) {
      Accum accum = init_value;
      for (size_t k = 0; k < shape_in[dim]; ++k) {
        size_t src_addr = i * pre_stride + k * post_stride + j;
        accum = op(accum, src_ptr[src_addr]);
      }
      size_t out_addr = i * post_stride + j;
      out_ptr[out_addr] = end_func(accum, shape_in[dim]);
    }
  }
  return out;
}

template <typename Input, typename Output, typename Accum>
Tensor reduce_sum(const TensorIntf &input, size_t dim) {
  return reduce<Input, Output, Accum>(
      input, dim, 0.0,
      [](const Accum &accum, const Input &input) { return accum + input; },
      "Sum", [](const Accum &accum, size_t) { return accum; });
}

template <typename Input, typename Output, typename Accum>
Tensor reduce_var(const TensorIntf &input, size_t dim,
                  size_t correction = 1) { // unbiased = True
  return reduce<Input, Output, std::pair<Accum, Accum>>(
      input, dim, {0.0, 0.0},
      [](const std::pair<Accum, Accum> &accum, const Input &input) {
        return std::pair{(accum.first + input), (accum.second + input * input)};
      },
      "Var",
      [&](const std::pair<Accum, Accum> &accum, size_t dim) {
        Accum mu = accum.first / static_cast<Accum>(dim);
        Accum t = accum.second / static_cast<Accum>(dim);
        Accum sigma = t - mu * mu;
        return Accum(dim) / Accum(dim - correction) * sigma;
      });
}

template <typename Input, typename Output, typename Accum>
Tensor reduce_mul(const TensorIntf &input, size_t dim) {
  return reduce<Input, Output, Accum>(
      input, dim, 1.0,
      [](const Accum &accum, const Input &input) { return accum * input; },
      "Prod", [](const Accum &accum, size_t) { return accum; });
}

template <typename Input, typename Output, typename Accum>
Tensor reduce_mean(const TensorIntf &input, size_t dim) {
  return reduce<Input, Output, Accum>(
      input, dim, 0.0,
      [](const Accum &accum, const Input &input) { return accum + input; },
      "Mean",
      [](const Accum &accum, size_t dim) {
        return accum / static_cast<Accum>(dim);
      });
}

template <typename Input, typename Output, typename Accum>
Tensor reduce_max(const TensorIntf &input, size_t dim) {
  return reduce<Input, Output, Accum>(
      input, dim, std::numeric_limits<Output>::lowest(),
      [](const Accum &accum, const Input &input) {
        return std::max<Accum>(accum, input);
      },
      "Max", [](const Accum &accum, size_t) { return accum; });
}

template <typename Input, typename Output, typename Accum>
Tensor reduce_min(const TensorIntf &input, size_t dim) {
  return reduce<Input, Output, Accum>(
      input, dim, std::numeric_limits<Output>::max(),
      [](const Accum &accum, const Input &input) {
        return std::min<Accum>(accum, input);
      },
      "Min", [](const Accum &accum, size_t) { return accum; });
}

template <typename Input, typename Output>
Tensor argmax(const TensorIntf &input, size_t dim) {
  return reduce<Input, Output, std::tuple<int32_t, Input, int32_t>>(
      input, dim, {0, std::numeric_limits<Output>::lowest(), -1},
      [](const std::tuple<int32_t, Input, int32_t> &accum, const Input &input) {
        int32_t index, best_index;
        Input best_value;
        std::tie(index, best_value, best_index) = accum;
        if (input > best_value) {
          return std::tuple{(index + 1), input, index};
        }
        return std::tuple{(index + 1), best_value, best_index};
      },
      "ArgMax",
      [](const std::tuple<int32_t, Input, int32_t> &accum, size_t) {
        return std::get<2>(accum);
      });
}

template <typename Input, typename Output>
Tensor argmin(const TensorIntf &input, size_t dim) {
  return reduce<Input, Output, std::tuple<int32_t, Input, int32_t>>(
      input, dim, {0, std::numeric_limits<Output>::max(), -1},
      [](const std::tuple<int32_t, Input, int32_t> &accum, const Input &input) {
        int32_t index, best_index;
        Input best_value;
        std::tie(index, best_value, best_index) = accum;
        if (input < best_value) {
          return std::tuple{(index + 1), input, index};
        }
        return std::tuple{(index + 1), best_value, best_index};
      },
      "ArgMin",
      [](const std::tuple<int32_t, Input, int32_t> &accum, size_t) {
        return std::get<2>(accum);
      });
}

template <typename Input, typename Output> Tensor cumsum_rvv(Tensor &input) {
  const auto &shape = input.shape();
  assert(shape.size() <= 2);
  Tensor out{DataTypeTraits<Output>::type, shape};
  spdlog::trace("opcall cumsum_rvv input {} output {}", input.id(), out.id());
  const Input *in = input.cdata<Input>();
  Output *out_ptr = out.data<Output>();
  if (shape.size() == 1) {
    out_ptr[0] = in[0];
    for (size_t i = 1; i < shape[0]; ++i) {
      out_ptr[i] = out_ptr[i - 1] + in[i];
    }
  } else {
    for (size_t i = 0; i < shape[0]; ++i) {
      out_ptr[i * shape[1]] = in[i * shape[1]];
      for (size_t j = 1; j < shape[1]; ++j) {
        out_ptr[i * shape[1] + j] =
            in[i * shape[1] + j] + out_ptr[i * shape[1] + j - 1];
      }
    }
  }
  return out;
}

namespace {
template <typename Input, typename Output>
Output sample_1d(const Input *in, size_t n, uint64_t *rng_state) {
  float coin = random_f32(rng_state);
  size_t i = 0;
  while (i < n && coin > in[i]) // binary search would be more efficient
    ++i;
  return i < n ? i : i - 1;
}
} // namespace

template <typename Input, typename Output>
Tensor sample_rvv(Tensor &input, size_t num_samples, uint64_t seed) {
  const auto &shape = input.shape();
  assert(shape.size() <= 2);
  std::vector<size_t> output_shape = shape;
  output_shape[output_shape.size() - 1] = num_samples;
  Tensor out{DataTypeTraits<Output>::type, output_shape};
  spdlog::trace("opcall sample_rvv input {} sample {} output {}", input.id(),
                num_samples, out.id());
  const Input *in = input.cdata<Input>();
  static uint64_t rng_state = seed;
  Output *out_ptr = out.data<Output>();
  if (shape.size() == 1) {
    for (size_t i = 0; i < num_samples; ++i) {
      out_ptr[i] = sample_1d<Input, Output>(in, shape[0], &rng_state);
    }
  } else {
    for (size_t i = 0; i < shape[0]; ++i) {
      for (size_t j = 0; j < num_samples; ++j) {
        out_ptr[i * num_samples + j] =
            sample_1d<Input, Output>(&(in[i * shape[1]]), shape[1], &rng_state);
      }
    }
  }
  return out;
}
