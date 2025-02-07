#pragma once
#include "functional.hpp"
#include "operators.hpp"
#include "tensor.hpp"
#include <cassert>
#include <string>
#include <vector>

class ModuleIntf {
public:
  virtual Tensor forward(const std::vector<Tensor> &inputs) = 0;
  virtual std::string dot() const = 0;
  virtual std::string id() const = 0;
};

class Module : public virtual ModuleIntf {
protected:
  const std::string module_type_;
  std::string name_;
  const std::vector<const Module *> parents_;

public:
  Module(const std::string &module_type, const std::string &name,
         const std::vector<const Module *> parents)
      : module_type_{module_type}, name_{name}, parents_{parents} {}
  void set_name(const std::string &name) { name_ = name; }
  std::string name() const { return name_; }
  virtual std::string id() const override {
    return std::to_string(reinterpret_cast<const unsigned long>(this));
  };
  virtual std::string dot() const override {
    return id() + " [label=\"" + name_ + " : " + module_type_ + "\"];\n";
  }
  const std::vector<const Module *> parents() const { return parents_; }
};

class UnaryIntf : public virtual ModuleIntf {
public:
  virtual Tensor forward(Tensor &input) = 0;
  virtual Tensor forward(const std::vector<Tensor> &inputs) override {
    assert(inputs.size() == 1);
    return forward(*const_cast<Tensor *>(inputs.data()));
  }
};

class BinaryIntf : public virtual ModuleIntf {
public:
  virtual Tensor forward(Tensor &input1, Tensor &input2) = 0;
  virtual Tensor forward(const std::vector<Tensor> &inputs) override {
    assert(inputs.size() == 2);
    Tensor *ptr = const_cast<Tensor *>(inputs.data());
    return forward(ptr[0], ptr[1]);
  }
};

class Input : public Module, public UnaryIntf {
public:
  Input(const std::string &name) : Module{"Input", name, {}} {}
  virtual Tensor forward(Tensor &input) { return input; }
};

class Add : public Module, public BinaryIntf {
  bool inplace_;

public:
  Add(const std::string &name, const Module &parent1, const Module &parent2,
      bool inplace = false)
      : Module{"Add", name, {&parent1, &parent2}}, inplace_{inplace} {}
  virtual Tensor forward(Tensor &input1, Tensor &input2) override {
    assert(input1.dtype() == input2.dtype());
    // TODO: take care of different types (fp8/bias + fp16)
    if (inplace_) {
      return add_<float>(input1, input2);
    }
    return add<float, float>(input1, input2);
  }
};

class Linear : public Module, public UnaryIntf {
  uint32_t in_, out_;
  bool bias_;
  Variable *weight_ = 0; // [out_ x in_]
  Variable *bias_v_ = 0;
  Tensor *weight_t_ = 0; // cached transposed weights
  bool transposed_ = false;

  Tensor ref(const Tensor &input) {
    if (bias_)
      return matmul<float, float, float>(input, *weight_, *bias_v_,
                                         transposed_);
    return matmul<float, float, float>(input, *weight_, transposed_);
  }

  Tensor evaluate(const Tensor &input) {
    size_t rows = input.shape()[0];
    if (rows == 1) {
      if (!transposed_) {
        if (bias_v_ == nullptr)
          return gemv<float, float, float>(input, *weight_);
        return gemv<float, float, float>(input, *weight_, *bias_v_);
      }
      if (weight_t_ == nullptr) {
        Tensor tmp = transpose<float>(*weight_, 0, 1);
        weight_t_ = new Tensor(tmp.dtype(), tmp.shape(), tmp.cdata<float>());
      }
      if (bias_v_ == nullptr)
        return gemv<float, float, float>(input, *weight_t_);
      return gemv<float, float, float>(input, *weight_t_, *bias_v_);
    }
    if (bias_)
      return matmul<float, float, float>(input, *weight_, *bias_v_,
                                         transposed_);
    return matmul<float, float, float>(input, *weight_, transposed_);
  }

public:
  Linear(const std::string &name, const Module &parent, uint32_t in,
         uint32_t out, bool bias = true, bool transposed = false)
      : Module{"Linear", name, {&parent}}, in_{in}, out_{out}, bias_{bias},
        transposed_{transposed} {}
  void load(const safetensors::safetensors_t &st, const std::string &weight,
            const std::string &bias = "") {
    weight_ = new Variable{st, weight};
    if (!bias.empty())
      bias_v_ = new Variable(st, bias);
  }
  virtual Tensor forward(Tensor &input) override {
    assert(weight_ != 0); // weight was loaded
    assert(input.dtype() == weight_->dtype());
    auto shape_in = input.shape();
    if (shape_in.size() == 3) { // stacked matmul
      size_t B = shape_in[0];
      size_t T = shape_in[1];
      size_t C = shape_in[2];
      input.reshape({B * T, C});
      Tensor out = evaluate(input);
      auto shape_out = out.shape();
      assert(shape_out.size() == 2);
      out.reshape({B, T, shape_out[1]});
      return out;
    } else {
      return evaluate(input);
    }
  }
};

class Embedding : public Module, public UnaryIntf {
  uint32_t n_embed_, dim_;
  bool range_; // for positional embedding
  Variable *embeddings_ = 0;

public:
  Embedding(const std::string &name, uint32_t n_embed, uint32_t dim,
            bool range = false)
      : Module{"Embedding", name, {}}, n_embed_{n_embed}, dim_{dim},
        range_{range} {}
  void load(const safetensors::safetensors_t &st,
            const std::string &embeddings) {
    embeddings_ = new Variable{st, embeddings};
  }
  void set_range(bool range) { range_ = range; }
  virtual Tensor forward(Tensor &input) override {
    assert(embeddings_ != 0);
    if (range_) {
      // for positional embeddings
      const auto shape_indices = input.shape();
      assert(shape_indices.size() <= 2);
      if (shape_indices.size() == 1) {
        return gather_range<float>(*embeddings_, shape_indices[0]);
      }
      return gather_range<float>(*embeddings_, shape_indices[1]);
    }
    return embeddings_gather<float, uint32_t>(*embeddings_, input);
  }
};

class NewGelu : public Module, public UnaryIntf {
  bool inplace_;
  const float scaling_factor = std::sqrt(2.0f / M_PI);

public:
  NewGelu(const std::string &name, const Module &parent, bool inplace = false)
      : Module{"NewGelu", name, {&parent}}, inplace_{inplace} {}
  virtual Tensor forward(Tensor &input) override {
    if (inplace_) {
      return elementwise_unary<float>(
          input,
          [&](const float &x) {
            float cube = 0.044715f * x * x * x;
            return 0.5f * x * (1.0f + tanhf(scaling_factor * (x + cube)));
          },
          "NewGelu");
    }
    return elementwise_unary<float, float>(
        input,
        [&](const float &x) {
          float cube = 0.044715f * x * x * x;
          return 0.5f * x * (1.0f + tanhf(scaling_factor * (x + cube)));
        },
        "NewGelu");
  }
};

class Cluster : public Module {
public:
  Cluster(const std::string &module_type, const std::string &name,
          const std::vector<const Module *> &parents)
      : Module(module_type, name, parents) {}
  virtual std::string id() const override { return "cluster_" + Module::id(); };
};

class Sequence : public Cluster, public UnaryIntf {
  std::vector<UnaryIntf *> modules_;

protected:
  void add(UnaryIntf *module) { modules_.push_back(module); }

public:
  Sequence(const std::string &module_type, const std::string &name,
           const Module &parent)
      : Cluster{module_type, name, {&parent}} {}
  virtual Tensor forward(Tensor &input) override {
    auto output = input;
    for (auto *m : modules_) {
      Tensor tmp = m->forward(output);
      output = tmp;
    }
    return output;
  }
  virtual Tensor forward(const std::vector<Tensor> &inputs) override {
    assert(inputs.size() == 1);
    return forward(*const_cast<Tensor *>(inputs.data()));
  }

  virtual std::string dot() const override {
    auto res = "subgraph " + id() + " { \nlabel=" + name_ + ";\n";
    for (const auto *m : modules_) {
      res += m->dot();
    }
    auto current = modules_.cbegin();
    res += parents_.front()->id() + " -> " + (*current)->id();
    ++current;
    while (current != modules_.cend()) {
      res += " -> " + (*current)->id();
      ++current;
    }
    res += ";\n}\n";
    return res;
  }
};

class LayerNorm : public Module, public UnaryIntf {
  uint32_t embd_;
  float eps_;
  Variable *weight_ = 0;
  Variable *bias_ = 0;

public:
  LayerNorm(const std::string &name, const Module &parent, uint32_t embd,
            float eps = 1e-5)
      : Module{"LayerNorm", name, {&parent}}, embd_{embd}, eps_{eps} {}
  void load(const safetensors::safetensors_t &st, const std::string &weight,
            const std::string &bias) {
    weight_ = new Variable{st, weight};
    bias_ = new Variable(st, bias);
  }
  virtual Tensor forward(Tensor &input) override {
    assert(weight_ != 0);
    assert(bias_ != 0);
    return layer_norm(input, *weight_, *bias_, eps_);
  }
};
