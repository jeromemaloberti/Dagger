#pragma once
#include "safetensors-cpp/safetensors.hh"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>

#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ranges.h"

template <typename T> const T *to(const uint8_t *v) {
  return static_cast<const T *>(v);
}

inline uint64_t unix_time() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

inline std::string to_string(safetensors::dtype dtype, const uint8_t *data) {
  switch (dtype) {
  case safetensors::dtype::kBOOL: {
    return std::to_string(data[0] ? 1 : 0);
  }
  case safetensors::dtype::kUINT8: {
    return std::to_string(data[0]);
  }
  case safetensors::dtype::kINT8: {
    return std::to_string(*reinterpret_cast<const int8_t *>(data));
  }
  case safetensors::dtype::kUINT16: {
    return std::to_string(*reinterpret_cast<const uint16_t *>(data));
  }
  case safetensors::dtype::kINT16: {
    return std::to_string(*reinterpret_cast<const int16_t *>(data));
  }
  case safetensors::dtype::kUINT32: {
    return std::to_string(*reinterpret_cast<const uint32_t *>(data));
  }
  case safetensors::dtype::kINT32: {
    return std::to_string(*reinterpret_cast<const int32_t *>(data));
  }
  case safetensors::dtype::kUINT64: {
    return std::to_string(*reinterpret_cast<const uint64_t *>(data));
  }
  case safetensors::dtype::kINT64: {
    return std::to_string(*reinterpret_cast<const int64_t *>(data));
  }
  case safetensors::dtype::kFLOAT16: {
    return std::to_string(
        safetensors::fp16_to_float(*reinterpret_cast<const uint16_t *>(data)));
  }
  case safetensors::dtype::kBFLOAT16: {
    return std::to_string(safetensors::bfloat16_to_float(
        *reinterpret_cast<const int64_t *>(data)));
  }
  case safetensors::dtype::kFLOAT32: {
    return std::to_string(*reinterpret_cast<const float *>(data));
  }
  case safetensors::dtype::kFLOAT64: {
    return std::to_string(*reinterpret_cast<const double *>(data));
  }
  }
  return std::string("???");
}

inline std::string to_string_snipped(const safetensors::tensor_t &t,
                                     const uint8_t *databuffer, size_t N = 8) {
  std::stringstream ss;
  size_t nitems = safetensors::get_shape_size(t);
  size_t itembytes = safetensors::get_dtype_bytes(t.dtype);

  if ((N == 0) || ((N * 2) >= nitems)) {
    ss << "[";
    for (size_t i = 0; i < nitems; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }
    ss << "]";
  } else {
    ss << "[";
    size_t head_end = (std::min)(N, nitems);
    size_t tail_start = (std::max)(nitems - N, head_end);

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }

    ss << ", ..., ";

    for (size_t i = tail_start; i < nitems; i++) {
      if (i > tail_start) {
        ss << ", ";
      }
      ss << to_string(t.dtype, databuffer + t.data_offsets[0] + i * itembytes);
    }

    ss << "]";
  }

  return ss.str();
}

inline std::string to_string_snipped(size_t n_elems, safetensors::dtype dtype,
                                     const std::vector<uint8_t> &data,
                                     size_t N = 8) {
  std::stringstream ss;
  size_t elem_size = safetensors::get_dtype_bytes(dtype);

  if ((N == 0) || ((N * 2) >= n_elems)) {
    ss << "[";
    for (size_t i = 0; i < n_elems; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(dtype, data.data() + i * elem_size);
    }
    ss << "]";
  } else {
    ss << "[";
    size_t head_end = (std::min)(N, n_elems);
    size_t tail_start = (std::max)(n_elems - N, head_end);

    for (size_t i = 0; i < head_end; i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << to_string(dtype, data.data() + i * elem_size);
    }

    ss << ", ..., ";

    for (size_t i = tail_start; i < n_elems; i++) {
      if (i > tail_start) {
        ss << ", ";
      }
      ss << to_string(dtype, data.data() + i * elem_size);
    }

    ss << "]";
  }

  return ss.str();
}

template <typename T> std::string to_string(const T *data, size_t size) {
  std::string res = "[";
  for (size_t i = 0; i < size; ++i) {
    res += std::to_string(data[i]);
    if (i < size - 1)
      res += ",";
  }
  return res += "]";
}
template <typename T> std::string to_string(const std::vector<T> &input) {
  return to_string(input.data(), input.size());
}

class TensorIndex {
private:
  std::vector<size_t> dims_;
  size_t size_;
  std::vector<size_t> accum_;

  void print_indent(size_t dim, std::ostream &out) const {
    for (size_t i = 0; i < dim; ++i)
      out << " ";
  }

  template <typename T>
  void print_dim(const std::vector<T> &data, std::ostream &out, size_t dim,
                 size_t &offset) const {
    print_indent(dim, out);
    out << "[";
    if (dim == dims_.size() - 1) {
      for (size_t i = 0; i < dims_[dim]; ++i) {
        out << std::to_string(data[i + offset]);
        if (i < dims_[dim] - 1)
          out << ",";
      }
      offset += dims_[dim];
    } else {
      out << "\n";
      for (size_t i = 0; i < dims_[dim]; ++i)
        print_dim(data, out, dim + 1, offset);
      print_indent(dim, out);
    }
    out << "]\n";
  }
  template <typename T>
  void print_dim(const T *data, std::ostream &out, size_t dim, size_t &offset,
                 uint32_t limit) const {
    print_indent(dim, out);
    out << "[";
    if (dim == dims_.size() - 1) {
      if (limit == 0 || limit * 2 >= dims_[dim]) {
        for (size_t i = 0; i < dims_[dim]; ++i) {
          out << std::to_string(data[i + offset]);
          if (i < dims_[dim] - 1)
            out << ",";
        }
      } else {
        for (size_t i = 0; i < limit; ++i) {
          out << std::to_string(data[i + offset]);
          if (i < limit)
            out << ",";
        }
        out << " ..., ";
        for (size_t i = dims_[dim] - limit; i < dims_[dim]; ++i) {
          out << std::to_string(data[i + offset]);
          if (i < dims_[dim] - 1)
            out << ",";
        }
      }
      offset += dims_[dim];
    } else {
      out << "\n";
      if (limit == 0 || limit * 2 >= dims_[dim]) {
        for (size_t i = 0; i < dims_[dim]; ++i)
          print_dim(data, out, dim + 1, offset, limit);
      } else {
        for (size_t i = 0; i < limit; ++i)
          print_dim(data, out, dim + 1, offset, limit);
        print_indent(dim, out);
        out << " ... \n";
        offset += accum_[dim] * (dims_[dim] - 2 * limit);
        for (size_t i = dims_[dim] - limit; i < dims_[dim]; ++i)
          print_dim(data, out, dim + 1, offset, limit);
      }
      print_indent(dim, out);
    }
    out << "]\n";
  }

public:
  template <typename It> TensorIndex(It start, It end) {
    for (auto it = start; it != end; ++it) {
      dims_.push_back(*it);
    }
    accum_.resize(dims_.size());
    size_ = 1;
    for (int i = dims_.size() - 1; i >= 0; --i) {
      accum_[i] = size_;
      size_ *= dims_[i];
    }
  }
  TensorIndex(const std::initializer_list<size_t> &dims)
      : TensorIndex(dims.begin(), dims.end()) {}
  TensorIndex(const std::vector<size_t> &dims)
      : TensorIndex(dims.begin(), dims.end()) {}

  TensorIndex permute(const std::vector<size_t> &dims) const {
    assert(dims.size() == dims_.size());
    std::vector<size_t> indices(dims_.size());
    size_t i = 0;
    for (size_t d : dims) {
      indices[i] = dims_[d];
      ++i;
    }
    return TensorIndex(indices.begin(), indices.end());
  }

  size_t rank() const { return dims_.size(); }

  size_t size() const { return size_; }
  std::vector<size_t> dims() const { return dims_; }
  bool same_dims(const std::vector<size_t> &d) const {
    if (d.size() != dims_.size())
      return false;
    for (size_t i = 0; i < dims_.size(); ++i) {
      if (dims_[i] != d[i])
        return false;
    }
    return true;
  }

  std::string to_string() const { return ::to_string(dims_); }

  size_t index(const std::vector<size_t> &indices) const {
    size_t res = 0;
    size_t i = 0;
    for (size_t d : indices) {
      assert(d < dims_[i]);
      res += d * accum_[i];
      ++i;
    }
    return res;
  }

  void indices(size_t index, std::vector<size_t> &indices) const {
    assert(indices.size() == dims_.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = (index / accum_[i]) % dims_[i];
    }
  }

  template <typename T>
  void print_tensor(const std::vector<T> &data, std::ostream &out) const {
    assert(data.size() == size_);
    size_t offset = 0;
    print_dim(data, out, 0, offset);
  }

  template <typename T>
  void print_tensor(const T *data, std::ostream &out, uint32_t limit) const {
    size_t offset = 0;
    print_dim(data, out, 0, offset, limit);
  }
};

// Interface for constant tensor
class TensorIntf {
private:
  virtual const uint8_t *data_ptr() const = 0;
  uint64_t time_;

public:
  TensorIntf() : time_{unix_time()} {}
  virtual std::vector<size_t> shape() const = 0;
  virtual safetensors::dtype dtype() const = 0;
  template <typename T> const T *cdata() const {
    return reinterpret_cast<const T *>(data_ptr());
  }
  virtual uint64_t n_elems() const = 0;
  virtual uint64_t size() const = 0;
  uint64_t time() const { return time_; }
  uint64_t unique_id() const {
    return reinterpret_cast<uint64_t>(data_ptr()) + time_;
  }
  std::string id() const {
    return fmt::format("{:x} {}", unique_id(), shape());
  }
};

class Variable : public TensorIntf { //
private:
  safetensors::tensor_t tensor_;
  uint64_t n_elems_;
  uint64_t size_;
  const uint8_t *data_;

public:
  virtual std::vector<size_t> shape() const override { return tensor_.shape; }
  virtual safetensors::dtype dtype() const override { return tensor_.dtype; }
  Variable(const safetensors::safetensors_t &st, const std::string &key) {
    bool res = st.tensors.at(key, &tensor_);
    assert(res);
    n_elems_ = safetensors::get_shape_size(tensor_);
    size_ = safetensors::get_dtype_bytes(tensor_.dtype) * n_elems_;
    const uint8_t *databuffer{nullptr};
    if (st.mmaped) {
      databuffer = st.databuffer_addr;
    } else {
      databuffer = st.storage.data();
    }
    data_ = databuffer + tensor_.data_offsets[0];
  }
  virtual uint64_t n_elems() const override { return n_elems_; }
  virtual uint64_t size() const override { return size_; }

private:
  virtual const uint8_t *data_ptr() const override { return data_; }
};

inline size_t product(const std::vector<size_t> &v) {
  size_t res = 1;
  for (const auto &d : v)
    res *= d;
  return res;
}

class Tensor : public TensorIntf {
  safetensors::dtype dtype_;
  std::vector<size_t> shape_;
  std::vector<size_t> accum_;
  uint64_t n_elems_;
  std::vector<uint8_t> data_;

  void slice1d(uint8_t *out_ptr, size_t in_offset, size_t n_elems) const {
    const size_t dsize = safetensors::get_dtype_bytes(dtype_);
    std::memcpy(out_ptr, data_.data() + in_offset * dsize, n_elems * dsize);
  }
  void insert1d(uint8_t *out_ptr, size_t out_offset, size_t n_elems) const {
    const size_t dsize = safetensors::get_dtype_bytes(dtype_);
    std::memcpy(out_ptr + out_offset * dsize, data_.data(), n_elems * dsize);
  }
  void slice2d(uint8_t *out_ptr, size_t rows, size_t cols, size_t rows_offset,
               size_t cols_offset, size_t last_dim) const {
    const size_t dsize = safetensors::get_dtype_bytes(dtype_);
    for (size_t i = 0; i < rows; ++i) {
      size_t out_offset = i * cols * dsize;
      size_t in_offset = ((i + rows_offset) * last_dim + cols_offset);
      slice1d(out_ptr + out_offset, in_offset, cols);
    }
  }
  void insert2d(uint8_t *out_ptr, size_t out_cols, size_t rows_offset,
                size_t cols_offset) const {
    const size_t dsize = safetensors::get_dtype_bytes(dtype_);
    for (size_t i = 0; i < shape_[0]; ++i) {
      size_t in_offset = i * shape_[1] * dsize;
      size_t out_offset = ((i + rows_offset) * out_cols + cols_offset) * dsize;
      std::memcpy(out_ptr + out_offset, data_.data() + in_offset,
                  shape_[1] * dsize);
    }
  }
  void accum() {
    accum_.resize(shape_.size());
    n_elems_ = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
      accum_[i] = n_elems_;
      n_elems_ *= shape_[i];
    }
  }

public:
  Tensor(const safetensors::dtype dtype, const std::vector<size_t> &shape)
      : dtype_{dtype}, shape_{shape} {
    accum();
    const size_t size = n_elems_ * safetensors::get_dtype_bytes(dtype);
    data_.resize(size);
  }
  template <typename T>
  Tensor(const safetensors::dtype dtype, const std::vector<size_t> &shape,
         const std::vector<T> &data)
      : dtype_{dtype}, shape_{shape} {
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data.data());
    data_ = std::vector<uint8_t>(bytes, bytes + data.size() * sizeof(T));
    accum();
  }
  template <typename T>
  Tensor(const safetensors::dtype dtype, const std::vector<size_t> &shape,
         const T *data)
      : dtype_{dtype}, shape_{shape} { // assume data is contiguous
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
    accum();
    data_ = std::vector<uint8_t>(bytes, bytes + n_elems_ * sizeof(T));
  }
  size_t index(const std::vector<size_t> &indices) const {
    size_t res = 0;
    size_t i = 0;
    assert(indices.size() <= shape_.size());
    for (size_t d : indices) {
      assert(d < shape_[i]);
      res += d * accum_[i];
      ++i;
    }
    return res;
  }
  virtual std::vector<size_t> shape() const override { return shape_; }
  virtual safetensors::dtype dtype() const override { return dtype_; }
  virtual uint64_t n_elems() const override { return n_elems_; }
  virtual uint64_t size() const override { return data_.size(); }
  template <typename T> T *data() {
    return reinterpret_cast<T *>(data_.data());
  }
  // std::string to_string_snipped(const size_t N = 8) const {
  //   return to_string_snipped(n_elems_, dtype_, data_, N);
  // }
  void reshape(const std::vector<size_t> &new_shape) {
    assert(product(shape_) == product(new_shape));
    shape_ = new_shape;
    accum();
  }
  void permute(size_t dim1, size_t dim2) {
    assert(dim1 < shape_.size());
    assert(dim2 < shape_.size());
    auto new_shape = shape_;
    new_shape[dim1] = shape_[dim2];
    new_shape[dim2] = shape_[dim1];
    reshape(new_shape);
  }

  template <typename T> std::vector<T> vdata() const {
    const T *base = reinterpret_cast<const T *>(data_.data());
    return std::vector<T>(base, base + n_elems_);
  }
  template <typename T> std::string to_string(const uint32_t limit = 0) const {
    TensorIndex idx{shape_};
    std::stringstream ss;
    idx.print_tensor(reinterpret_cast<const T *>(data_.data()), ss, limit);
    return ss.str();
  }
  void insert(const Tensor &t, const std::vector<size_t> &offsets) {
    const auto &t_shape = t.shape();
    assert(t.dtype_ == dtype_);
    assert(t_shape.size() == offsets.size());
    assert(offsets.size() == shape_.size());
    for (size_t i = 0; i < t_shape.size(); ++i) {
      assert(t_shape[i] + offsets[i] <= shape_[i]);
    }
    const size_t dsize = safetensors::get_dtype_bytes(dtype_);
    if (offsets.size() == 1) {
      t.insert1d(data_.data(), offsets[0], t_shape[0]);
    } else {
      if (offsets.size() == 2) {
        t.insert2d(data_.data(), shape_[1], offsets[0], offsets[1]);
      } else {
        if (offsets.size() == 3) {
          for (size_t i = 0; i < t_shape[0]; ++i) {
            for (size_t j = 0; j < t_shape[1]; ++j) {
              size_t in_offset = t.index({i, j}) * dsize;
              size_t out_offset =
                  index({i + offsets[0], j + offsets[1], offsets[2]});
              std::memcpy(data_.data() + out_offset * dsize,
                          t.data_.data() + in_offset, t_shape[2] * dsize);
            }
          }
        } else {
          if (offsets.size() == 4) {
            for (size_t i = 0; i < t_shape[0]; ++i) {
              for (size_t j = 0; j < t_shape[1]; ++j) {
                for (size_t k = 0; k < t_shape[2]; ++k) {
                  size_t in_offset = t.index({i, j, k}) * dsize;
                  size_t out_offset = index({i + offsets[0], j + offsets[1],
                                             k + offsets[2], offsets[3]});
                  std::memcpy(data_.data() + out_offset * dsize,
                              t.data_.data() + in_offset, t_shape[3] * dsize);
                }
              }
            }
          } else {
            assert(false);
          }
        }
      }
    }
  }
  Tensor subtile(const std::vector<size_t> &dims,
                 const std::vector<size_t> &offsets) const {
    assert(dims.size() == offsets.size());
    assert(dims.size() == shape_.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      assert(dims[i] + offsets[i] <= shape_[i]);
    }
    Tensor out{dtype_, dims};
    uint8_t *out_ptr = out.data<uint8_t>();
    const size_t dsize = safetensors::get_dtype_bytes(dtype_);
    if (dims.size() == 1) {
      slice1d(out_ptr, offsets[0], dims[0]);
    } else {
      if (dims.size() == 2) {
        slice2d(out_ptr, dims[0], dims[1], offsets[0], offsets[1], shape_[1]);
      } else {
        if (dims.size() == 3) {
          for (size_t i = 0; i < dims[0]; ++i) {
            for (size_t j = 0; j < dims[1]; ++j) {
              size_t out_offset = out.index({i, j}) * dsize;
              size_t in_offset =
                  index({i + offsets[0], j + offsets[1], offsets[2]});
              slice1d(out_ptr + out_offset, in_offset, dims[2]);
            }
          }
        } else {
          if (dims.size() == 4) {
            for (size_t i = 0; i < dims[0]; ++i) {
              for (size_t j = 0; j < dims[1]; ++j) {
                for (size_t k = 0; k < dims[2]; ++k) {
                  size_t out_offset = out.index({i, j, k}) * dsize;
                  size_t in_offset = index({i + offsets[0], j + offsets[1],
                                            k + offsets[2], offsets[3]});
                  slice1d(out_ptr + out_offset, in_offset, dims[3]);
                }
              }
            }
          } else {
            assert(false);
          }
        }
      }
    }
    return out;
  }
  // can be done in NISA by creatind new tiles, so cost == 0
  std::vector<Tensor> split(size_t size, size_t dim = 0) {
    assert(dim < shape_.size());
    assert(size < shape_[dim]);
    assert((shape_[dim] % size) == 0);
    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = size;
    std::vector<Tensor> res(shape_[dim] / size, {dtype_, new_shape});
    size_t pre_dim = 1;
    for (size_t i = 0; i < dim; ++i)
      pre_dim *= shape_[i];
    size_t block_size = 1;
    for (size_t i = dim + 1; i < shape_.size(); ++i)
      block_size *= shape_[i];
    size_t in_stride = block_size * shape_[dim];
    size_t out_stride = block_size * size;

#pragma omp parallel for
    for (size_t i = 0; i < pre_dim; ++i) {
      for (size_t j = 0; j < res.size(); ++j) {
        float *out_ptr = res[j].data<float>();
        size_t src_idx = i * in_stride + j * out_stride;
        size_t out_idx = i * out_stride;
        std::memcpy(static_cast<void *>(&out_ptr[out_idx]),
                    &data_[src_idx * safetensors::get_dtype_bytes(dtype_)],
                    size * block_size * safetensors::get_dtype_bytes(dtype_));
      }
    }
    return res;
  }

private:
  virtual const uint8_t *data_ptr() const override { return data_.data(); }
};

inline bool eq(const std::vector<size_t> &shape1,
               const std::vector<size_t> &shape2) {
  assert(shape1.size() == shape2.size());
  assert(product(shape1) == product(shape2));
  bool res = true;
  for (auto i = 0; i < shape1.size(); ++i)
    res = res && (shape1[i] == shape2[i]);
  return res;
}

template <typename T>
std::vector<T> weights2matrix(const std::vector<T> &weights, const uint32_t M,
                              const uint32_t C, const uint32_t R,
                              const uint32_t S, const uint32_t block_size,
                              uint32_t &row_size) {
  const TensorIndex input_index{M, C, R, S};
  const uint32_t c_blocks = std::ceil(float(C) / block_size);
  const TensorIndex output_index{c_blocks, R, S, block_size, M};
  row_size = c_blocks * block_size * R * S;
  std::vector<T> res(output_index.size());
  for (size_t m = 0; m < M; ++m) {
    for (size_t c = 0; c < c_blocks; ++c) {
      const uint32_t c_max = C - c * block_size;
      for (size_t r = 0; r < R; ++r) {
        for (size_t s = 0; s < S; ++s) {
          for (size_t i = 0; i < block_size; ++i) {
            const uint32_t o_index = output_index.index({c, r, s, i, m});
            const uint32_t i_index =
                input_index.index({m, i + c * c_blocks, r, s});
            if (i < c_max) {
              res[o_index] = weights[i_index];
            } else {
              res[o_index] = T(0);
            }
          }
        }
      }
    }
  }
  return res;
}

template <typename T>
std::vector<T>
input2matrix(const std::vector<T> &input, const uint32_t N, const uint32_t C,
             const uint32_t H, const uint32_t W, const uint32_t R,
             const uint32_t S, const uint32_t pad_h, const uint32_t pad_w,
             const uint32_t dilation_h, const uint32_t dilation_w,
             const uint32_t stride_h, const uint32_t stride_w,
             const uint32_t block_size, const uint32_t row_size) {
  const TensorIndex input_index{N, C, H, W};
  const uint32_t c_blocks = std::ceil(float(C) / block_size);
  const uint32_t output_h =
      (H + 2 * pad_h - (dilation_h * (R - 1) + 1)) / stride_h + 1;
  const uint32_t output_w =
      (W + 2 * pad_w - (dilation_w * (S - 1) + 1)) / stride_w + 1;
  const TensorIndex output_index{N, output_h, output_w,  c_blocks,
                                 R, S,        block_size};
  std::vector<T> res(output_index.size());
  for (size_t n = 0; n < N; ++n) {
    for (size_t h_o = 0; h_o < output_h; ++h_o) {
      for (size_t w_o = 0; w_o < output_w; ++w_o) {

        for (size_t c = 0; c < c_blocks; ++c) {
          const uint32_t c_max = C - c * block_size;
          for (size_t r = 0; r < R; ++r) {
            int32_t h_in = h_o * stride_h - pad_h + r * dilation_h;
            for (size_t s = 0; s < S; ++s) {
              int32_t w_in = w_o * stride_w - pad_w + s * dilation_w;
              for (size_t i = 0; i < block_size; ++i) {
                size_t c_in = c * c_blocks + i;
                const uint32_t o_index =
                    output_index.index({n, h_o, w_o, c, r, s, i});
                if (i < c_max && h_in >= 0 && w_in >= 0 && h_in < H &&
                    w_in < W) {
                  const uint32_t i_index =
                      input_index.index({n, c_in, size_t(h_in), size_t(w_in)});
                  res[o_index] = input[i_index];
                } else {
                  res[o_index] = T(0);
                }
              }
            }
          }
        }
      }
    }
  }
  return res;
}

template <typename T> struct DataTypeTraits;

template <> struct DataTypeTraits<int8_t> {
  static constexpr safetensors::dtype type = safetensors::kINT8;
};

template <> struct DataTypeTraits<int32_t> {
  static constexpr safetensors::dtype type = safetensors::kINT32;
};

template <> struct DataTypeTraits<uint32_t> {
  static constexpr safetensors::dtype type = safetensors::kUINT32;
};

template <> struct DataTypeTraits<float> {
  static constexpr safetensors::dtype type = safetensors::kFLOAT32;
};
