#pragma once
#include "safetensors-cpp/safetensors.hh"
#include <cassert>
#include <string>
#include <vector>

class Tokenizer {
  std::vector<std::string> tokens_;

  inline bool load_from_file(const std::string &filename, std::string *warn,
                             std::string *err) {
    std::vector<unsigned char> data;
    if (!safetensors::detail::ReadWholeFile(&data, err, filename, nullptr)) {
      std::cerr << "Could not load " << filename << std::endl;
      return false;
    }
    if (data.size() < 16) {
      if (err) {
        (*err) += "Size is too short.\n";
      }
      return false;
    }
    std::string json_str(reinterpret_cast<const char *>(&data[0]), data.size());
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
    bool is_array = v.is<minijson::array>();
    assert(is_array);
    if (auto pa = v.as<::minijson::array>()) {
      ::minijson::array::const_iterator i;
      for (i = pa->begin(); i != pa->end(); i++) {
        bool is_string = i->is<minijson::string>();
        assert(is_string);
        const std::string *token = i->as<std::string>();
        tokens_.push_back(*token);
      }
    }
    std::cerr << "Tokens read: " << tokens_.size() << std::endl;
    return true;
  }

public:
  Tokenizer(const std::string &filename) {
    std::string err, warn;
    bool res = load_from_file(filename, &warn, &err);
    assert(res);
  }
  std::string decode(uint32_t token_id) const {
    assert(token_id < tokens_.size());
    return tokens_[token_id];
  }
};
