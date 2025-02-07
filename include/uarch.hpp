#pragma once

#include "uarch/config.h"
#include <algorithm>

inline const std::tuple<size_t, size_t, size_t, size_t>
get_xyzw(const std::vector<size_t> &shape) {
  auto x = shape.size() > 1 ? shape[shape.size() - 2] : 1;
  auto y = shape.size() > 1 ? shape[shape.size() - 2] : 1;
  auto z = shape.size() > 2 ? shape[shape.size() - 3] : 1;
  auto w = shape.size() > 3 ? shape[shape.size() - 4] : 1;
  return {x, y, z, w};
}

constexpr bool is_quanttized(datatype dt) {
  switch (dt) {
  case datatype::i8:
  case datatype::u8:
  case datatype::fp8q:
    return true;
  default:
    return false;
  }
  return false;
}

inline auto get_data_size(datatype dt) {
  switch (dt) {
  case datatype::i8:
  case datatype::u8:
  case datatype::fp8q: {
    return 1U;
  }
  case datatype::i16:
  case datatype::u16:
  case datatype::fp16:
  case datatype::bf16: {
    return 2U;
  }
  case datatype::i32:
  case datatype::u32:
  case datatype::fp32: {
    return 4U;
  }
  case datatype::fp64: {
    return 8U;
  }
  default:
    std::cerr << "Unsupported datatype yet";
  }
  return 0U;
}

namespace latency {

inline uint64_t transpose(const config::UarchConfig *uarch_config,
                          const std::vector<size_t> &shape, const size_t dim1,
                          const size_t dim2, const datatype &dtype) {
  auto datatype_multiplier = get_data_size(dtype);
  auto output_shape = shape;
  output_shape[dim1] = shape[dim2];
  output_shape[dim2] = shape[dim1];
  const size_t shape_size = shape.size();
  auto [mtx, mty, mtz, mtw] = get_xyzw(output_shape);
  if (datatype_multiplier != 1 && datatype_multiplier != 2 &&
      datatype_multiplier != 4) {
    std::cerr << "Unsupported datatype by Reshaper";
  }
  if (dim1 == shape_size - 1 || dim2 == shape_size - 1) {
    uint32_t buffer_write_latency = (64 / datatype_multiplier);
    auto num_subtile_x_i = ceil((mtx * datatype_multiplier) /
                                uarch_config->getValue<unsigned>(
                                    config::ParameterName::rsh_buffer_width));
    auto num_subtile_x_o = ceil((mty * datatype_multiplier) /
                                uarch_config->getValue<unsigned>(
                                    config::ParameterName::rsh_buffer_width));
    auto num_subtile_y_i = mty * mtz;
    auto num_subtile_y_o = mtx * mtz;
    auto num_subtile_z = mtw;
    auto buffer_read_latency = 0;
    auto operation_latency = std::max(num_subtile_x_i * num_subtile_y_i,
                                      num_subtile_x_o * num_subtile_y_o) *
                             num_subtile_z;
    return buffer_write_latency + buffer_read_latency + operation_latency +
           uarch_config->getValue<unsigned>(
               config::ParameterName::rsh_input_fifo_latency) +
           uarch_config->getValue<unsigned>(
               config::ParameterName::rsh_output_fifo_latency) +
           uarch_config->getValue<unsigned>(
               config::ParameterName::rsh_sfr_latency);
  } else {
    auto num_subtile_x = ceil(mtx * datatype_multiplier) /
                         uarch_config->getValue<unsigned>(
                             config::ParameterName::rsh_buffer_width);
    auto num_subtile_y = mty * mtz;
    auto num_subtile_z = mtw;
    auto buffer_write_latency = 1;
    auto buffer_read_latency = 0;
    return buffer_write_latency + buffer_read_latency +
           (num_subtile_x * num_subtile_y * num_subtile_z) +
           uarch_config->getValue<unsigned>(
               config::ParameterName::rsh_input_fifo_latency) +
           uarch_config->getValue<unsigned>(
               config::ParameterName::rsh_output_fifo_latency) +
           uarch_config->getValue<unsigned>(
               config::ParameterName::rsh_sfr_latency);
  }
}

inline uint64_t reduce_mean(const config::UarchConfig *uarch_config,
                            const std::vector<size_t> &shape, const size_t dim,
                            const datatype &dtype) {
  // if dtype is quantized the formula is different.
  auto sram_latency = uarch_config->getValue<unsigned>(
                          config::ParameterName::sram_init_latency) +
                      uarch_config->getValue<unsigned>(
                          config::ParameterName::sram_done_latency);
  auto pipe_latency =
      uarch_config->getValue<unsigned>(config::ParameterName::vx_pipe_init) +
      uarch_config->getValue<unsigned>(config::ParameterName::vx_pipe_done);
  auto datasize = get_data_size(dtype);
  auto num_element =
      std::min<uint32_t>(uint32_t(uarch_config->getValue<unsigned>(
                             config::ParameterName::dlen_width)),
                         shape[shape.size() - 1] * datasize);
  auto [x, y, z, w] = get_xyzw(shape);
  auto mtx = (x / num_element) + ((x % num_element) != 0);
  if (dim == shape.size() - 3) {
    std::swap(y, z);
  } else {
    if (dim == shape.size() - 4)
      std::swap(y, w);
  }
  auto num_uop = 2;
  return (y + num_uop) * (mtx * z * w) + sram_latency + pipe_latency;
}

// for max/min/sum
inline uint64_t reduce(const config::UarchConfig *uarch_config,
                       const std::vector<size_t> &shape, const size_t dim,
                       const datatype &dtype, bool sum) {
  auto sram_latency = uarch_config->getValue<unsigned>(
                          config::ParameterName::sram_init_latency) +
                      uarch_config->getValue<unsigned>(
                          config::ParameterName::sram_done_latency);
  auto pipe_latency =
      uarch_config->getValue<unsigned>(config::ParameterName::vx_pipe_init) +
      uarch_config->getValue<unsigned>(config::ParameterName::vx_pipe_done);
  auto datasize = get_data_size(dtype);
  auto num_element =
      std::min<uint32_t>(uint32_t(uarch_config->getValue<unsigned>(
                             config::ParameterName::dlen_width)),
                         shape[shape.size() - 1] * datasize);
  auto [x, y, z, w] = get_xyzw(shape);
  auto mtx =
      ((x * datasize) / num_element) + (((x * datasize) % num_element) != 0);
  auto vl = 1;
  if (dtype == datatype::i8 || dtype == datatype::u8 ||
      dtype == datatype::fp8q) {
    vl = shape[shape.size() - 1] <= 32 ? 1 : 2;
  } else if (dtype == datatype::fp16) {
    vl = 1;
  } else {
    std::cerr << "Unsupported data type in VX\n";
    return 0;
  }
  // nisa can change the type in dest wrt source.
  if (dtype != datatype::fp8q && dtype != datatype::fp16) {
    std::cerr << "Unsupported datatype in VX \n";
    return 0;
  }
  if (dim == shape.size() - 3) {
    std::swap(y, z);
  } else {
    if (dim == shape.size() - 4)
      std::swap(y, w);
  }
  auto num_uop = sum && is_quanttized(dtype) ? 3 : 2;
  return (y + num_uop) * vl * (mtx * z * w) + sram_latency + pipe_latency;
}

// needs a destination dtype
inline uint64_t elementwise_binary(const config::UarchConfig *uarch_config,
                                   const std::vector<size_t> &shape_LHS,
                                   datatype dtype) {
  auto sram_latency = uarch_config->getValue<unsigned>(
                          config::ParameterName::sram_init_latency) +
                      uarch_config->getValue<unsigned>(
                          config::ParameterName::sram_done_latency);
  auto pipe_latency =
      uarch_config->getValue<unsigned>(config::ParameterName::vx_pipe_init) +
      uarch_config->getValue<unsigned>(config::ParameterName::vx_pipe_done);
  auto datasize = get_data_size(dtype);
  auto num_element =
      std::min<uint32_t>(uint32_t(uarch_config->getValue<unsigned>(
                             config::ParameterName::dlen_width)),
                         shape_LHS[shape_LHS.size() - 1] * datasize);
  auto [x, mty, mtz, mtw] = get_xyzw(shape_LHS);
  auto mtx = (x / num_element) + ((x % num_element) != 0);
  auto num_uop = 1;
  if (is_quanttized(dtype)) {
    if (dtype != datatype::fp8q)
      num_uop = 3; // if dest is also quantized +1
    auto vl = x <= 32 ? 1 : 2;
    num_uop *= vl;
  }
  return mtx * mty * mtz * mtw * num_uop + sram_latency + pipe_latency;
}

//     inline uint64_t load_dense(const config::UarchConfig *uarch_config) {
//   auto sram_bw =
//       uarch_config->getValue<unsigned>(config::ParameterName::sram_bw_bytepc);
//   auto dram_bw =
//       uarch_config->getValue<unsigned>(config::ParameterName::dram_bw_bytepc);
//   auto sram_width{64U};
//   auto dram_width{32U};
//   auto c{64U};
//   auto m{1U};
//   auto dram_mem_tile_info = sram_mem_tile_info;
//   auto dram_attribute = instr->template get_attribute<NX2::dram_attr_dense>(
//       NX2::AttributeId::dattr);
//   dram_mem_tile_info.set_addr(dram_attribute.get_daddr());
//   dram_mem_tile_info.set_stx(dram_attribute.get_dstx());
//   dram_mem_tile_info.set_sty(dram_attribute.get_dsty());
//   dram_mem_tile_info.set_stz(dram_attribute.get_dstz());
//   const auto sram_addr_ranges = get_memory_ranges(sram_mem_tile_info, c, m);
//   const auto dram_addr_ranges = get_memory_ranges(dram_mem_tile_info, c, m);
//   auto sram_cycles =
//       (get_number_requests_from_range(sram_addr_ranges, sram_width) *
//        sram_width) /
//       sram_bw;
//   auto dram_cycles =
//       (get_number_requests_from_range(dram_addr_ranges, dram_width) *
//        dram_width) /
//       dram_bw;
//   return std::max(sram_cycles, dram_cycles);
// }
}; // namespace latency
