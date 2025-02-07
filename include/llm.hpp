#pragma once

#include "modules.hpp"

class MLP : public Module, public UnaryIntf { // FFN for Jay
  Linear c_fc_;
  NewGelu gelu_;
  Linear c_proj_;

public:
  MLP(uint32_t embed, const Module &parent)
      : Module{"MLP", "MLP", {&parent}}, c_fc_{"c", parent, embed, 4 * embed},
        gelu_{"gelu", c_fc_}, c_proj_{"proj", gelu_, 4 * embed, embed} {}
  void load(const safetensors::safetensors_t &st, const std::string &prefix) {
    const std::string p = prefix + "mlp.";
    c_fc_.load(st, p + "c_fc.weight", p + "c_fc.bias");
    c_proj_.load(st, p + "c_proj.weight", p + "c_proj.bias");
  }
  virtual Tensor forward(Tensor &input) override {
    Tensor tmp = c_fc_.forward(input);
    Tensor act = gelu_.forward(tmp);
    return c_proj_.forward(act);
  }
};

template <typename Att>
class CausalSelfAttention : public Module, public UnaryIntf {
  Linear c_attn_;
  Att attn_;
  Linear c_proj_;

public:
  CausalSelfAttention(const std::string &name, const Module &parent,
                      uint32_t embd, uint32_t heads, uint32_t seq_len)
      : Module{"SelfAttention", name, {&parent}},
        c_attn_{"c", parent, embd, 3 * embd},
        attn_{"attn", c_attn_, embd, heads, seq_len, true},
        c_proj_{"proj", attn_, embd, embd} {}
  virtual Tensor forward(Tensor &input) override {
    Tensor tmp = c_attn_.forward(input);
    Tensor att = attn_.forward(tmp);
    return c_proj_.forward(att);
  }
  void load(const safetensors::safetensors_t &st, const std::string &prefix) {
    const std::string p = prefix + "attn.";
    c_attn_.load(st, p + "c_attn.weight", p + "c_attn.bias");
    c_proj_.load(st, p + "c_proj.weight", p + "c_proj.bias");
  }
};
