// SPDX-License-Identifier: MIT
// Copyright 2023 - Present, Syoyo Fujita.
// Inspired from:
// https://gist.github.com/Narsil/5d6bf307995158ad2c4994f323967284
#pragma once
#include "minijson.hpp"

#ifdef __ANDROID__
#ifdef SAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS
#include <android/asset_manager.h>
#endif

#ifdef SAFETENSORS_CPP_IMPLEMENTATION
AAssetManager *asset_manager = nullptr;
#else
extern AAssetManager *asset_manager;
#endif
#endif

namespace safetensors {

constexpr size_t kMaxDim =
    8; // must be equal to SAFETENSORS_C_MAX_DIM in `safetensors-c.h`

enum dtype {
  kBOOL,
  kUINT8,
  kINT8,
  kINT16,
  kUINT16,
  kFLOAT16,
  kBFLOAT16,
  kINT32,
  kUINT32,
  kFLOAT32,
  kFLOAT64,
  kINT64,
  kUINT64,
};

template <typename T> using ordered_dict = minijson::ordered_dict<T>;

struct tensor_t {
  safetensors::dtype dtype;
  std::vector<size_t> shape;
  std::array<size_t, 2> data_offsets;
};

struct safetensors_t {
  // we need ordered dict(preserves the order of key insertion)
  // as done in Python's OrderedDict, since JSON data may not be sorted by its
  // key string.
  ordered_dict<tensor_t> tensors;
  ordered_dict<std::string> metadata;
  std::vector<uint8_t> storage; // empty when mmap'ed
  size_t header_size{0};        // JSON size

  bool mmaped{false};

  //
  // Following members are set when mmaped.
  //
  const uint8_t *mmap_addr{nullptr};
  size_t mmap_size{0};
  const uint8_t *databuffer_addr{nullptr}; // [mmap_addr + header_size + 8]
  size_t databuffer_size{0};               // mmap_size - header_size - 8
  // opaque pointer to safetensors_file and safetensors_mmap
  void *st_file{nullptr};
  void *st_mmap{nullptr};

  ~safetensors_t();
};

//
// Load safetensors from file.
// databuffer is copied to `safetensors_t::storage`.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool load_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err);

//
// Load safetensors data from memory.
// databuffer is copied to `safetensors_t::storage`.
//
// @param[in] addr Memory address of safetensors data.
// @param[in] nbytes The size in bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
//
bool load_from_memory(const uint8_t *addr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err);

//
// Load safetensors with memory mapping(i.e. zero-copy).
// databuffer is not copied to `safetensors_t` object, thus the app must hold
// file during `safetensor_t` object is live.
//
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err);

//
// Load safetensors from mmaped region.
// databuffer is not copied to `safetensors_t` object, thus the app must not
// free/unmap `addr` during `safetensor_t` object is live.
//
// @param[in] addr mmaped memory address of safetensors data.
// @param[in] nbytes mmap bytes.
// @param[in] filename Filename of corresponding memory data. Can be empty.
// @param[out] st safetensors data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool mmap_from_memory(const uint8_t *arr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err);

//
// Save safetensors to file.
//
// @param[in] st safetensors data.
// @param[in] filename Filepath. Assume UTF-8 filepath.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool save_to_file(const safetensors_t &st, const std::string &filename,
                  std::string *warn, std::string *err);

//
// Save safetensors to memory.
//
// @param[in] st safetensors data.
// @param[out] data_out Serialized safetensor data.
// @param[out] warn Warning message buffer(can be nullptr if you don't need
// warning message)
// @param[out] err Error message buffer(can be nullptr if you don't need error
// message)
//
// @return true upon success. `err` will be filled when false.
bool save_to_memory(const std::string &filename, std::vector<uint8_t> *data_out,
                    std::string *warn, std::string *err);

//
// Utility functions
//

// Returns shape[0] * shape[1] * ...
// Empty Tensor(any shape[i] is 0) returns 0.
// Zero-rank tensor([]) return 1.
size_t get_shape_size(const tensor_t &t);

// Returns dtype size in bytes.
size_t get_dtype_bytes(const safetensors::dtype dtype);
std::string get_dtype_str(const safetensors::dtype dtype);

// Validate data_offsets of all tensors in safetensors_t.
bool validate_data_offsets(const safetensors_t &st, std::string &err);

uint16_t float_to_bfloat16(float x);
float bfloat16_to_float(uint16_t x);

uint16_t float_to_fp16(float x);
float fp16_to_float(uint16_t x);

} // namespace safetensors

#if defined(SAFETENSORS_CPP_IMPLEMENTATION)

#include <cstring>
#include <fstream>
#include <memory>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <stdio.h> // for _fseeki64
#include <windows.h>
#endif

namespace safetensors {

// Max header(JSON) size. 100 MB as done in original safetensors implementation.
constexpr size_t kMaxJSONSize = 1024ull * 1024ull * 100ull;

namespace detail {

#ifdef _WIN32
std::wstring UTF8ToWchar(const std::string &str) {
  int wstr_size =
      MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), nullptr, 0);
  std::wstring wstr(size_t(wstr_size), 0);
  MultiByteToWideChar(CP_UTF8, 0, str.data(), int(str.size()), &wstr[0],
                      int(wstr.size()));
  return wstr;
}

std::string WcharToUTF8(const std::wstring &wstr) {
  int str_size = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()),
                                     nullptr, 0, nullptr, nullptr);
  std::string str(size_t(str_size), 0);
  WideCharToMultiByte(CP_UTF8, 0, wstr.data(), int(wstr.size()), &str[0],
                      int(str.size()), nullptr, nullptr);
  return str;
}
#endif

bool ReadWholeFile(std::vector<unsigned char> *out, std::string *err,
                   const std::string &filepath, void *) {
#ifdef SAFETENSORS_CPP_ANDROID_LOAD_FROM_ASSETS
  if (asset_manager) {
    AAsset *asset = AAssetManager_open(asset_manager, filepath.c_str(),
                                       AASSET_MODE_STREAMING);
    if (!asset) {
      if (err) {
        (*err) += "File open error : " + filepath + "\n";
      }
      return false;
    }
    size_t size = AAsset_getLength(asset);
    if (size == 0) {
      if (err) {
        (*err) += "Invalid file size : " + filepath +
                  " (does the path point to a directory?)";
      }
      return false;
    }
    out->resize(size);
    AAsset_read(asset, reinterpret_cast<char *>(&out->at(0)), size);
    AAsset_close(asset);
    return true;
  } else {
    if (err) {
      (*err) += "No asset manager specified : " + filepath + "\n";
    }
    return false;
  }
#else
#ifdef _WIN32
#if defined(__GLIBCXX__) // mingw
  int file_descriptor =
      _wopen(UTF8ToWchar(filepath).c_str(), _O_RDONLY | _O_BINARY);
  __gnu_cxx::stdio_filebuf<char> wfile_buf(file_descriptor, std::ios_base::in);
  std::istream f(&wfile_buf);
#elif defined(_MSC_VER) || defined(_LIBCPP_VERSION)
  // For libcxx, assume _LIBCPP_HAS_OPEN_WITH_WCHAR is defined to accept
  // `wchar_t *`
  std::ifstream f(UTF8ToWchar(filepath).c_str(), std::ifstream::binary);
#else
  // Unknown compiler/runtime
  std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
#else
  std::ifstream f(filepath.c_str(), std::ifstream::binary);
#endif
  if (!f) {
    if (err) {
      (*err) += "File open error : " + filepath + "\n";
    }
    return false;
  }

  // For directory(and pipe?), peek() will fail(Posix gnustl/libc++ only)
  f.peek();
  if (!f) {
    if (err) {
      (*err) +=
          "File read error. Maybe empty file or invalid file : " + filepath +
          "\n";
    }
    return false;
  }

  f.seekg(0, f.end);
  size_t sz = static_cast<size_t>(f.tellg());

  // std::cout << "sz = " << sz << "\n";
  f.seekg(0, f.beg);

  if (int64_t(sz) < 0) {
    if (err) {
      (*err) += "Invalid file size : " + filepath +
                " (does the path point to a directory?)";
    }
    return false;
  } else if (sz == 0) {
    if (err) {
      (*err) += "File is empty : " + filepath + "\n";
    }
    return false;
  } else if (sz >= (std::numeric_limits<std::streamoff>::max)()) {
    if (err) {
      (*err) += "Invalid file size : " + filepath + "\n";
    }
    return false;
  }

  out->resize(sz);
  f.read(reinterpret_cast<char *>(&out->at(0)),
         static_cast<std::streamsize>(sz));

  return true;
#endif
}

bool parse_metadata(const ::minijson::value &v, ordered_dict<std::string> &dst,
                    std::string *err) {
  if (auto po = v.as<::minijson::object>()) {
    for (size_t i = 0; i < po->size(); i++) {
      ::minijson::value ov;
      if (!po->at(i, &ov)) {
        if (err) {
          (*err) += "[Internal error] Invalid object found in __metadata__, at "
                    "index " +
                    std::to_string(i) + ".\n";
        }
        return false;
      }

      if (auto so = ov.as<std::string>()) {
        if (dst.count(po->keys()[i])) {
          // This should not be happen though
          if (err) {
            (*err) += "Duplicate key `" + po->keys()[i] +
                      "` found in __metadata__.\n";
          }
          return false;
        }

        dst.insert(po->keys()[i], *so);
      } else {
        if (err) {
          (*err) += "`" + po->keys()[i] + "` must be string value.\n";
        }
        return false;
      }
    }
  } else {
    if (err) {
      (*err) += "`__metadata__` value must be JSON object.\n";
    }
    return false;
  }

  return true;
}

bool parse_dtype(const ::minijson::value &v, safetensors::dtype &dtype,
                 std::string *err) {
  if (auto so = v.as<std::string>()) {
    if ((*so) == "BOOL") {
      dtype = safetensors::dtype::kBOOL;
    } else if ((*so) == "U8") {
      dtype = safetensors::dtype::kUINT8;
    } else if ((*so) == "I8") {
      dtype = safetensors::dtype::kINT8;
    } else if ((*so) == "U16") {
      dtype = safetensors::dtype::kUINT16;
    } else if ((*so) == "I16") {
      dtype = safetensors::dtype::kINT16;
    } else if ((*so) == "U32") {
      dtype = safetensors::dtype::kUINT32;
    } else if ((*so) == "I32") {
      dtype = safetensors::dtype::kINT32;
    } else if ((*so) == "U64") {
      dtype = safetensors::dtype::kUINT64;
    } else if ((*so) == "I64") {
      dtype = safetensors::dtype::kINT64;
    } else if ((*so) == "F16") {
      dtype = safetensors::dtype::kFLOAT16;
    } else if ((*so) == "BF16") {
      dtype = safetensors::dtype::kBFLOAT16;
    } else if ((*so) == "F32") {
      dtype = safetensors::dtype::kFLOAT32;
    } else if ((*so) == "F64") {
      dtype = safetensors::dtype::kFLOAT64;
    } else {
      if (err) {
        (*err) += "Unknown `dtype` string: " + *so + ".\n";
      }
      return false;
    }
  } else {
    if (err) {
      (*err) +=
          "`dtype` item should be string type but got " + v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_shape(const ::minijson::value &v, std::vector<size_t> &dst,
                 std::string *err) {
  // NOTE:
  // - Empty tensors (tensors with 1 dimension being 0) are allowed
  // - [] is allowed(0-Rank tensor = merely a scalar)
  if (auto pa = v.as<::minijson::array>()) {
    ::minijson::array::const_iterator i;

    for (i = pa->begin(); i != pa->end(); i++) {
      if (auto pn = i->as<::minijson::number>()) {
        if (dst.size() >= kMaxDim) {
          if (err) {
            (*err) += "`shape` length must be less than " +
                      std::to_string(kMaxDim) + " but got " +
                      std::to_string(dst.size()) + ".\n";
          }
          return false;
        }

        dst.push_back(size_t(*pn));

      } else {
        if (err) {
          (*err) += "Array item in `shape` must be number type, but got " +
                    i->type_name() + ".\n";
        }
        return false;
      }
    }
  } else {
    if (err) {
      (*err) +=
          "`shape` value must be JSON array, but got " + v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_data_offsets(const ::minijson::value &v, std::array<size_t, 2> &dst,
                        std::string *err) {
  if (auto pa = v.as<::minijson::array>()) {
    ::minijson::array::const_iterator i;
    size_t cnt = 0;

    for (i = pa->begin(); i != pa->end(); i++) {
      if (auto pn = i->as<::minijson::number>()) {
        if (cnt >= 2) {
          if (err) {
            (*err) += "`data_offsets` length must be 2.\n";
          }
          return false;
        }

        dst[cnt] = size_t(*pn);

        cnt++;

      } else {
        if (err) {
          (*err) +=
              "Array item in `data_offsets` must be number type, but got " +
              i->type_name() + ".\n";
        }
        return false;
      }
    }

    if (cnt != 2) {
      if (err) {
        (*err) += "`data_offsets` length must be 2.\n";
      }
      return false;
    }
  } else {
    if (err) {
      (*err) += "`data_offsets` value must be JSON array, but got " +
                v.type_name() + ".\n";
    }
    return false;
  }

  return true;
}

bool parse_tensor(const std::string &name, const ::minijson::value &v,
                  tensor_t &tensor, std::string *err) {
  if (auto po = v.as<::minijson::object>()) {

    bool dtype_found{false};
    bool shape_found{false};
    bool data_offsets_found{false};

    dtype dtype;
    std::vector<size_t> shape;
    std::array<size_t, 2> data_offsets{};

    for (size_t i = 0; i < po->size(); i++) {
      std::string key = po->keys()[i];

      if (key == "dtype") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `dtype` has invalid object.\n";
          }
          return false;
        }

        if (!parse_dtype(value, dtype, err)) {
          return false;
        }

        dtype_found = true;
      } else if (key == "shape") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `shape` has invalid object.\n";
          }
          return false;
        }

        if (!parse_shape(value, shape, err)) {
          return false;
        }

        shape_found = true;
      } else if (key == "data_offsets") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. `data_offsets` has invalid object.\n";
          }
          return false;
        }
        if (!parse_data_offsets(value, data_offsets, err)) {
          return false;
        }

        data_offsets_found = true;
      } else {
        // Unknown key. Report error?
      }
    }

    if (!dtype_found) {
      if (err) {
        (*err) += "`" + name + "` does not have `dtype` item.\n";
      }
      return false;
    }

    if (!shape_found) {
      if (err) {
        (*err) += "`" + name + "` does not have `shape` item.\n";
      }
      return false;
    }

    bool is_empty_tensor{false};
    if ((shape.size() > 0)) {
      for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0) {
          is_empty_tensor = true;
          break;
        }
      }
    }

    if (is_empty_tensor) {
      // They are not storing any data in the databuffer, yet retaining size in
      // the header. So ignore data_offsets
      if (data_offsets_found) {
        // TODO: make this warn instead of err?
        if (err) {
          (*err) +=
              "`" + name +
              "` is empty tensors(tensors with 1 dimension being 0), and no "
              "data in databuffer, but `data_offsets` item is provided.\n";
        }
        return false;
      }
    } else {
      if (!data_offsets_found) {
        if (err) {
          (*err) += "`" + name + "` does not have `data_offsets` item.\n";
        }
        return false;
      }
    }

    tensor.dtype = dtype;
    tensor.shape = shape;
    tensor.data_offsets = data_offsets;

  } else {
    if (err) {
      (*err) += "`" + name + "` value must be JSON object.\n";
    }
    return false;
  }

  return true;
}

// From llama.cpp
#if defined(_WIN32)
static std::string safetensors_format_win_err(DWORD err) {
  LPSTR buf;
  size_t size = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0,
      NULL);
  if (!size) {
    return "FormatMessageA failed";
  }
  std::string ret(buf, size);
  LocalFree(buf);
  return ret;
}
#endif

struct safetensors_file {
  // use FILE * so we don't have to re-open the file to mmap
  FILE *fp{nullptr};
  size_t size{0};
  mutable bool _valid{false};
  std::string _err;

  safetensors_file(const char *fname, const char *mode) {
    fp = std::fopen(fname, mode);
    if (fp == nullptr) {
      _err = "failed to open " + std::string(fname) + ":" +
             std::string(strerror(errno)) + "\n";
      _valid = false;
    } else {
      seek(0, SEEK_END);
      size = tell();
      seek(0, SEEK_SET);
      _valid = true;
    }
  }

  ~safetensors_file() {
    if (fp) {
      std::fclose(fp);
      fp = nullptr;
    }
  }

  size_t tell() const {
#ifdef _WIN32
    __int64 ret = _ftelli64(fp);
#else
    long ret = std::ftell(fp);
#endif
    if (ret == -1) {
      // this really shouldn't fail
      _valid = false;
      return 0;
    }

    return (size_t)ret;
  }

  void seek(size_t offset, int whence) const {
#ifdef _WIN32
    int ret = _fseeki64(fp, (__int64)offset, whence);
#else
    int ret = std::fseek(fp, (long)offset, whence);
#endif
    if (ret == 0) {
      _valid = false;
    }
  }

  bool &is_valid() const { return _valid; }

  const std::string &get_error() const { return _err; }
};

struct safetensors_mmap {
  uint8_t *addr{nullptr};
  size_t size{0};

  bool _valid{false};
  std::string _warn;
  std::string _err;

  const bool is_valid() const { return _valid; }

  const std::string &get_error() const { return _err; }

  const std::string &get_warning() const { return _warn; }

  safetensors_mmap(const safetensors_mmap &) = delete;

#ifdef _POSIX_MAPPED_FILES
  static constexpr bool SUPPORTED = true;

  safetensors_mmap(struct safetensors_file *file,
                   size_t prefetch = (size_t)-1 /* -1 = max value */,
                   bool numa = false) {
    size = file->size;
    int fd = fileno(file->fp);
    int flags = MAP_SHARED;
    // prefetch/readahead impairs performance on NUMA systems
    if (numa) {
      prefetch = 0;
    }
#ifdef __linux__
    if (prefetch) {
      flags |= MAP_POPULATE;
    }
#endif
    addr = reinterpret_cast<uint8_t *>(
        mmap(NULL, file->size, PROT_READ, flags, fd, 0));
    if (addr == MAP_FAILED) {
      _valid = false;
      _err = "mmap failed: " + std::string(strerror(errno)) + "\n";

      size = 0;
      addr = nullptr;

      return;
    }

    if (prefetch > 0) {
      // Advise the kernel to preload the mapped memory
      if (posix_madvise(addr, std::min(file->size, prefetch),
                        POSIX_MADV_WILLNEED)) {
        _warn += "posix_madvise(.., POSIX_MADV_WILLNEED) failed: " +
                 std::string(strerror(errno)) + "\n";
      }
    }
    if (numa) {
      // advise the kernel not to use readahead
      // (because the next page might not belong on the same node)
      if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
        _warn += "posix_madvise(.., POSIX_MADV_RANDOM) failed: " +
                 std::string(strerror(errno)) + "\n";
      }
    }

    _valid = true;
  }

  ~safetensors_mmap() {
    if (_valid) {
      munmap(addr, size);
    }
    size = 0;
    addr = nullptr;
    _valid = false;
  }

#elif defined(_WIN32)
  static constexpr bool SUPPORTED = true;

  safetensors_mmap(struct safetensors_file *file, bool prefetch = true,
                   bool numa = false) {
    (void)numa;

    size = file->size;

    HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

    HANDLE hMapping =
        CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    DWORD error = GetLastError();

    if (hMapping == NULL) {
      // TODO: get error message
      _err = "CreateFileMappingA failed: " + safetensors_format_win_err(error) +
             "\n";
      _valid = false;
      size = 0;
      addr = nullptr;
      return;
    }

    addr = reinterpret_cast<uint8_t *>(
        MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
    error = GetLastError();
    CloseHandle(hMapping);

    if (addr == NULL) {
      _err =
          "MapViewOfFile failed: " + safetensors_format_win_err(error) + "\n";
    }

    if (prefetch) {
      // PrefetchVirtualMemory is only present on Windows 8 and above, so we
      // dynamically load it
      BOOL(WINAPI * pPrefetchVirtualMemory)
      (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

      // may fail on pre-Windows 8 systems
      pPrefetchVirtualMemory =
          reinterpret_cast<decltype(pPrefetchVirtualMemory)>(
              GetProcAddress(hKernel32, "PrefetchVirtualMemory"));

      if (pPrefetchVirtualMemory) {
        // advise the kernel to preload the mapped memory
        WIN32_MEMORY_RANGE_ENTRY range;
        range.VirtualAddress = addr;
        range.NumberOfBytes = (SIZE_T)size;
        if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
          _warn += "PrefetchVirtualMemory failed: " +
                   safetensors_format_win_err(GetLastError()) + "\n";
        }
      }
    }
  }
  ~safetensors_mmap() {
    if (!UnmapViewOfFile(addr)) {
      _warn += "UnmapViewOfFile failed: " +
               safetensors_format_win_err(GetLastError()) + "\n";
    }
  }
#else
  static constexpr bool SUPPORTED = false;

  safetensors_mmap(struct safetensors_file *file, bool prefetch = true,
                   bool numa = false) {
    (void)file;
    (void)prefetch;
    (void)numa;

    _valid = false;
    _err = "mmap not supported\n";
    addr = nullptr;
    size = 0;
  }
#endif
};

// Based on MIOPen bfloat16
// https://github.com/ROCmSoftwarePlatform/MIOpen/blob/master/src/kernels/bfloat16_dev.hpp

/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

typedef union cvt_bf16_fp32 {
  uint32_t u32;
  uint16_t ushortvec[2];

  float f32;
} cvt_bf16_fp32_t;

float bfloat16_to_float(uint16_t src_val) {
  cvt_bf16_fp32_t target_val;

  target_val.ushortvec[0] = 0;
  target_val.ushortvec[1] = src_val;

  return target_val.f32;
}

uint16_t float_to_bfloat16(float src_val) {
  cvt_bf16_fp32_t target_val;
  target_val.f32 = src_val;
  // BF16 round and NaN preservation code matches
  // https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/develop/library/include/rocblas_bfloat16.h
  if ((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
  {
    // When all of the exponent bits are 1, the value is Inf or NaN.
    // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
    // mantissa bit. Quiet NaN is indicated by the most significant mantissa
    // bit being 1. Signaling NaN is indicated by the most significant
    // mantissa bit being 0 but some other bit(s) being 1. If any of the
    // lower 16 bits of the mantissa are 1, we set the least significant bit
    // of the bfloat16 mantissa, in order to preserve signaling NaN in case
    // the bloat16's mantissa bits are all 0.
    if ((target_val.u32 & 0xffff) != 0) {
      target_val.u32 |= 0x10000; // Preserve signaling NaN
    }
  } else {
#if 1 // MIOPEN_USE_RNE_BFLOAT16
      // When the exponent bits are not all 1s, then the value is zero, normal,
      // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
      // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
      // This causes the bfloat16's mantissa to be incremented by 1 if the 16
      // least significant bits of the float mantissa are greater than 0x8000,
      // or if they are equal to 0x8000 and the least significant bit of the
    // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
    // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
    // has the value 0x7f, then incrementing it causes it to become 0x00 and
    // the exponent is incremented by one, which is the next higher FP value
    // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
    // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
    // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
    // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
    // incrementing it causes it to become an exponent of 0xFF and a mantissa
    // of 0x00, which is Inf, the next higher value to the unrounded value.
    target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#endif // MIOPEN_USE_RNE_BFLOAT16
  }

  return target_val.ushortvec[1];
}

// half <-> float conversion based on: https://gist.github.com/rygorous/2156668
// (CC0 license)
//

// Little endian
union FP32le {
  unsigned int u;
  float f;
  struct {
    unsigned int Mantissa : 23;
    unsigned int Exponent : 8;
    unsigned int Sign : 1;
  } s;
};

// Little endian
union float16le {
  unsigned short u;
  struct {
    unsigned int Mantissa : 10;
    unsigned int Exponent : 5;
    unsigned int Sign : 1;
  } s;
};

float half_to_float_le(float16le h) {
  static const FP32le magic = {113 << 23};
  static const unsigned int shifted_exp = 0x7c00
                                          << 13; // exponent mask after shift
  FP32le o;

  o.u = (h.u & 0x7fffU) << 13U;          // exponent/mantissa bits
  unsigned int exp_ = shifted_exp & o.u; // just the exponent
  o.u += (127 - 15) << 23;               // exponent adjust

  // handle exponent special cases
  if (exp_ == shifted_exp)   // Inf/NaN?
    o.u += (128 - 16) << 23; // extra exp adjust
  else if (exp_ == 0)        // Zero/Denormal?
  {
    o.u += 1 << 23; // extra exp adjust
    o.f -= magic.f; // renormalize
  }

  o.u |= (h.u & 0x8000U) << 16U; // sign bit
  return o.f;
}

uint16_t float_to_half_full_le(float _f) {
  FP32le f;
  f.f = _f;
  float16le o = {0};

  // Based on ISPC reference code (with minor modifications)
  if (f.s.Exponent == 0) // Signed zero/denormal (which will underflow)
    o.s.Exponent = 0;
  else if (f.s.Exponent == 255) // Inf or NaN (all exponent bits set)
  {
    o.s.Exponent = 31;
    o.s.Mantissa = f.s.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
  } else                                     // Normalized number
  {
    // Exponent unbias the single, then bias the halfp
    int newexp = f.s.Exponent - 127 + 15;
    if (newexp >= 31) // Overflow, return signed infinity
      o.s.Exponent = 31;
    else if (newexp <= 0) // Underflow
    {
      if ((14 - newexp) <= 24) // Mantissa might be non-zero
      {
        unsigned int mant = f.s.Mantissa | 0x800000; // Hidden 1 bit
        o.s.Mantissa = mant >> (14 - newexp);
        if ((mant >> (13 - newexp)) & 1) // Check for rounding
          o.u++; // Round, might overflow into exp bit, but this is OK
      }
    } else {
      o.s.Exponent = static_cast<unsigned int>(newexp);
      o.s.Mantissa = f.s.Mantissa >> 13;
      if (f.s.Mantissa & 0x1000) // Check for rounding
        o.u++;                   // Round, might overflow to inf, this is OK
    }
  }

  o.s.Sign = f.s.Sign;

  return o.u;
}

bool parse_safetensors_header(const uint8_t *addr, const size_t nbytes,
                              const std::string &filename, safetensors_t *st,
                              std::string *warn, std::string *err) {
  if (nbytes < 16) {
    if (err) {
      (*err) += "Size is too short.\n";
    }
    return false;
  }

  uint64_t header_size{0};
  memcpy(reinterpret_cast<unsigned char *>(&header_size), addr,
         sizeof(uint64_t));

  if (header_size < 4) {
    if (err) {
      (*err) += "Header size is too short.\n";
    }
    return false;
  }

  if ((8 + header_size) > nbytes) {
    if (err) {
      (*err) += "Header size " + std::to_string(header_size) +
                " + 8 exceeds input size " + std::to_string(nbytes) + " .\n";
    }
    return false;
  }

  if (header_size > kMaxJSONSize) {
    if (err) {
      (*err) += "Header JSON size exceeds the limit(" +
                std::to_string(kMaxJSONSize) + ").\n";
    }
    return false;
  }

  // assume JSON data is small enough.
  std::string json_str(reinterpret_cast<const char *>(&addr[8]), header_size);
  const char *p = json_str.c_str();

  ::minijson::value v;
  ::minijson::error e = ::minijson::parse(p, v);

  if (e != ::minijson::no_error) {
    if (err) {
      std::string json_err(::minijson::errstr(e));
      (*err) += "JSON parse error: " + json_err + "\n";
    }

    return false;
  }

  ordered_dict<tensor_t> tensors;
  ordered_dict<std::string> metadata;

  // root element must be dict.
  if (auto po = v.as<::minijson::object>()) {
    for (size_t i = 0; i < po->size(); i++) {
      std::string key = po->keys()[i];

      if (key == "__metadata__") {
        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. Invalid object in __metadata__.\n";
          }
          return false;
        }

        if (!detail::parse_metadata(value, metadata, err)) {
          return false;
        }
      } else {
        // tensor

        if (tensors.count(key)) {
          if (err) {
            (*err) += "Duplicate key `" + key + "` found.\n";
          }
          return false;
        }

        ::minijson::value value;
        if (!po->at(i, &value)) {
          if (err) {
            (*err) += "Internal error. Invalid object in `" + key + "`.\n";
          }
          return false;
        }

        tensor_t tensor;
        if (!detail::parse_tensor(key, value, tensor, err)) {
          return false;
        }

        tensors.insert(key, std::move(tensor));
      }
    }
  } else {
    if (err) {
      (*err) += "JSON root elements must be object(dict)\n";
    }
  }

  st->tensors = std::move(tensors);
  st->metadata = std::move(metadata);
  st->header_size = header_size;

#if 0
  size_t databuffer_size = nbytes - header_size - 8;

  st->storage.resize(nbytes);
  memcpy(st->storage.data(), addr + 8 + header_size, nbytes);

  st->mmaped = false;
  st->mmap_addr = addr + 8 + header_size;
  st->mmap_size = 0;
#endif

  return true;
}

} // namespace detail

safetensors_t::~safetensors_t() {
  if (st_mmap) {
    detail::safetensors_mmap *p =
        reinterpret_cast<detail::safetensors_mmap *>(st_mmap);
    delete p;
    st_mmap = nullptr;
  }

  if (st_file) {
    detail::safetensors_file *p =
        reinterpret_cast<detail::safetensors_file *>(st_file);
    delete p;
    st_file = nullptr;
  }
}

//
// - 8byte: header_size
// - json data(header_size bytes)
// - tensor data(filesize - header_size)
//

bool load_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err) {
  std::vector<unsigned char> data;
  if (!detail::ReadWholeFile(&data, err, filename, nullptr)) {
    return false;
  }

  return load_from_memory(reinterpret_cast<const uint8_t *>(data.data()),
                          data.size(), filename, st, warn, err);
}

bool load_from_memory(const uint8_t *addr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err) {
  if (nbytes < 16) {
    if (err) {
      (*err) += "Size is too short.\n";
    }
    return false;
  }

  if (!detail::parse_safetensors_header(addr, nbytes, filename, st, warn,
                                        err)) {
    return false;
  }

  size_t databuffer_size = nbytes - st->header_size - 8;

  st->storage.resize(databuffer_size);
  memcpy(st->storage.data(), addr + 8 + st->header_size, databuffer_size);

  st->mmaped = false;
  st->mmap_addr = nullptr;
  st->mmap_size = 0;
  st->databuffer_addr = nullptr;
  st->databuffer_size = 0;

  return true;
}

bool mmap_from_file(const std::string &filename, safetensors_t *st,
                    std::string *warn, std::string *err) {
  if (!st) {
    return false;
  }

  detail::safetensors_file *pf =
      new detail::safetensors_file(filename.c_str(), "rb");
  if (!pf->is_valid()) {
    if (err) {
      (*err) += pf->get_error();
    }
    delete pf;
    return false;
  }

  // TODO: prefetch, numa
  detail::safetensors_mmap *pm = new detail::safetensors_mmap(pf);

  bool ret = mmap_from_memory(pm->addr, pm->size, filename, st, warn, err);

  if (!ret) {
    delete pm;
    delete pf;

    return false;
  }

  st->mmap_addr = pm->addr;
  st->mmap_size = pm->size;

  st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
  st->databuffer_size = st->mmap_size - (8 + st->header_size);

  // retain pointer
  st->st_file = pf;
  st->st_mmap = pm;

  st->mmaped = true;

  return true;
}

bool mmap_from_memory(const uint8_t *addr, const size_t nbytes,
                      const std::string &filename, safetensors_t *st,
                      std::string *warn, std::string *err) {
  if (!addr) {
    return false;
  }

  if (nbytes < 16) {
    return false;
  }

  if (!st) {
    return false;
  }

  if (!detail::parse_safetensors_header(addr, nbytes, filename, st, warn,
                                        err)) {
    return false;
  }

  size_t databuffer_size = nbytes - st->header_size - 8;

  st->mmaped = true;

  st->mmap_addr = addr;
  st->mmap_size = nbytes;

  st->databuffer_addr = st->mmap_addr + 8 + st->header_size;
  st->databuffer_size = st->mmap_size - (8 + st->header_size);

  return true;
}

float bfloat16_to_float(uint16_t x) { return detail::bfloat16_to_float(x); }

uint16_t float_to_bfloat16(float x) { return detail::float_to_bfloat16(x); }

float fp16_to_float(uint16_t x) {
  detail::float16le src;
  src.u = x;
  return detail::half_to_float_le(src);
}

uint16_t float_to_fp16(float x) { return detail::float_to_half_full_le(x); }

size_t get_dtype_bytes(const safetensors::dtype dtype) {
  size_t sz = 0;

  switch (dtype) {
  case safetensors::dtype::kBOOL:
    // Original Rust implementaion uses 1.
    sz = 1;
    break;
  case safetensors::dtype::kUINT8:
    sz = 1;
    break;
  case safetensors::dtype::kINT8:
    sz = 1;
    break;
  case safetensors::dtype::kUINT16:
    sz = 2;
    break;
  case safetensors::dtype::kINT16:
    sz = 2;
    break;
  case safetensors::dtype::kINT32:
    sz = 4;
    break;
  case safetensors::dtype::kUINT32:
    sz = 4;
    break;
  case safetensors::dtype::kFLOAT16:
    sz = 2;
    break;
  case safetensors::dtype::kBFLOAT16:
    sz = 2;
    break;
  case safetensors::dtype::kFLOAT32:
    sz = 4;
    break;
  case safetensors::dtype::kFLOAT64:
    sz = 8;
    break;
  case safetensors::dtype::kINT64:
    sz = 8;
    break;
  case safetensors::dtype::kUINT64:
    sz = 8;
    break;
  }

  return sz;
}

std::string get_dtype_str(const safetensors::dtype dtype) {
  switch (dtype) {
  case safetensors::dtype::kBOOL:
    return "BOOL";
  case safetensors::dtype::kUINT8:
    return "U8";
  case safetensors::dtype::kINT8:
    return "I8";
  case safetensors::dtype::kUINT16:
    return "U16";
  case safetensors::dtype::kINT16:
    return "I16";
  case safetensors::dtype::kINT32:
    return "I32";
  case safetensors::dtype::kUINT32:
    return "U32";
  case safetensors::dtype::kFLOAT16:
    return "F16";
  case safetensors::dtype::kBFLOAT16:
    return "BF16";
  case safetensors::dtype::kFLOAT32:
    return "F32";
  case safetensors::dtype::kFLOAT64:
    return "F64";
  case safetensors::dtype::kINT64:
    return "I64";
  case safetensors::dtype::kUINT64:
    return "U64";
  }
  return "???";
}

// Empty Tensor returns 0.
// Zero-rank Tensor reuturns 1(scalar)
size_t get_shape_size(const tensor_t &t) {
  if (t.shape.empty()) {
    return 1;
  }

  if (t.shape.size() >= kMaxDim) { // invalid ndim
    return 0;
  }

  size_t sz = 1;

  for (size_t i = 0; i < t.shape.size(); i++) {
    sz *= t.shape[i];
  }

  return sz;
}

bool validate_data_offsets(const safetensors_t &st, std::string &err) {
  bool valid{true};

  std::stringstream ss;

  size_t databuffersize;
  if (st.mmaped) {
    databuffersize = st.databuffer_size;
  } else {
    databuffersize = st.storage.size();
  }

  size_t ntensors{0};
  // Iterate with key insertion order.
  for (size_t i = 0; i < st.tensors.size(); i++) {

    std::string key = st.tensors.keys()[i];

    tensor_t tensor;
    if (!st.tensors.at(i, &tensor)) {
      ss << "Internal error: Failed to get tensor at [" << i << "]\n";
      valid = false;
      continue;
    }

    if (tensor.data_offsets[0] > tensor.data_offsets[1]) {
      ss << key << ".data_offsets.BEGIN " << tensor.data_offsets[0]
         << " must be less than or equal to data_offsets.END "
         << tensor.data_offsets[1] << "\n";
      valid = false;
    }

    size_t tensor_size = get_dtype_bytes(tensor.dtype) * get_shape_size(tensor);

    if (tensor_size == 0) {
      // OK
      continue;
    }

    // data_offsets are absolute offset from the databuffer(file)
    if (tensor.data_offsets[0] > databuffersize) {
      ss << "Tensor `" << key << "`.data_offset.BEGIN "
         << tensor.data_offsets[0] << " exceeds databuffer size "
         << databuffersize << ".\n";
      valid = false;
    }

    if (tensor.data_offsets[1] > databuffersize) {
      ss << "Tensor `" << key << "`.data_offset.END " << tensor.data_offsets[1]
         << " exceeds databuffer size " << databuffersize << ".\n";
      valid = false;
    }

    size_t data_size = tensor.data_offsets[1] - tensor.data_offsets[0];

    if (tensor_size != data_size) {
      ss << "Data size mismatch. The size in Tensor `" << key << "` is "
         << tensor_size << ", but the size from data_offsets is " << data_size
         << "\n";
      valid = false;
    }

    ntensors++;
    if (ntensors == st.tensors.size()) {
      // Last element's data_offsets[1] must be equal to databuffer size.
      if (tensor.data_offsets[1] != databuffersize) {
        ss << "The last tensor's data_offset.END(" << tensor.data_offsets[1]
           << ") must be equal to databufer size " << databuffersize << ".\n";
        valid = false;
      }
    }
  }

  if (!valid) {
    err = ss.str();
  }

  return valid;
}

bool save_to_memory(const safetensors_t &st, std::vector<uint8_t> *dst,
                    std::string *warn, std::string *err) {
  // directly serialize JSON string.
  std::stringstream ss;

  // NOTE: The last offset **must** be the end of the file,
  // so write __metadata__ first(if metadata part exists)

  std::string _err;
  if (!validate_data_offsets(st, _err)) {
    if (err) {
      (*err) += "Invalid safensors is provided.\n";
      (*err) += _err;
    }
    return false;
  }

  ss << "{";
  if (st.metadata.size()) {
    ss << "\"__metadata__\": {";
    size_t nmeta = 0;
    for (size_t i = 0; i < st.metadata.size(); i++) {
      std::string key = st.metadata.keys()[i];
      std::string value;
      st.metadata.at(i, &value);

      if (nmeta > 0) {
        ss << ", ";
      }
      ss << "\"" + key + "\": \"" << value << "\"";
      nmeta++;
    }
    ss << "}";

    if (st.tensors.size()) {
      ss << ", ";
    }
  }

  size_t ntensors = 0;
  {
    for (size_t i = 0; i < st.tensors.size(); i++) {

      std::string key = st.tensors.keys()[i];
      safetensors::tensor_t tensor;
      st.tensors.at(i, &tensor);

      if (tensor.shape.size() > safetensors::kMaxDim) {
        if (err) {
          (*err) += key + ".shape is too large.\n";
          (*err) += _err;
        }
        return false;
      }

      if (ntensors > 0) {
        ss << ", ";
      }
      ss << "\"" << key << "\": {";
      ss << "\"dtype\": \"" << safetensors::get_dtype_str(tensor.dtype)
         << "\", ";
      ss << "\"shape\": [";
      for (size_t i = 0; i < tensor.shape.size(); i++) {
        if (i > 0) {
          ss << ", ";
        }
        ss << tensor.shape[i];
      }
      ss << "]";
      ss << ", \"data_offsets\": [" << tensor.data_offsets[0] << ", "
         << tensor.data_offsets[1] << "]";
      ss << "}";
      ntensors++;
    }
  }
  ss << "}";

  std::string header_str = ss.str();

  uint64_t header_size = header_str.size(); // do not include '\n'

  const void *databuffer_addr{nullptr};
  size_t databuffer_size{0};
  if (st.mmaped) {
    databuffer_size = st.databuffer_size;
    databuffer_addr = st.databuffer_addr;
  } else {
    databuffer_size = st.storage.size();
    databuffer_addr = reinterpret_cast<const void *>(st.storage.data());
  }

  // make databuffer addr start from the multiple of 8.
  size_t pad_bytes = 0;
  if ((header_size % 8) != 0) {
    pad_bytes = 8 - (header_size % 8);
  }
  // printf("header_size = %d\n", int(header_size));
  // printf("pad_bytes = %d\n", int(pad_bytes));
  size_t padded_header_size = header_size + pad_bytes;
  dst->resize(8 + padded_header_size + databuffer_size);

  // write padded header_size
  memcpy(dst->data(), &padded_header_size, 8);

  // write header
  memcpy(dst->data() + 8, header_str.data(), header_size);

  // Use whitespace for trailing padding.
  memset(dst->data() + 8 + header_size, 0x20, pad_bytes);

  memcpy(dst->data() + 8 + padded_header_size, databuffer_addr,
         databuffer_size);

  return true;
}

bool save_to_file(const safetensors_t &st, const std::string &filename,
                  std::string *warn, std::string *err) {
  // TODO: Use more reliable io.
  std::ofstream ofs(filename, std::ios::binary);

  if (!ofs) {
    if (err) {
      (*err) += "Failed to open `" + filename +
                "` to write. File is either existing directory or "
                "write-protected, or disk is full?\n";
    }
    return false;
  }

  std::vector<uint8_t> buf;
  if (!save_to_memory(st, &buf, warn, err)) {
    return false;
  }

  ofs.write(reinterpret_cast<const char *>(buf.data()), buf.size());
  if (!ofs) {
    if (err) {
      (*err) += "Failed to write safetensor data to `" + filename +
                "`. Maybe no disk space available?(Required bytes : " +
                std::to_string(buf.size()) + "\n";
    }
    return false;
  }

  return true;
}

} // namespace safetensors

#endif
