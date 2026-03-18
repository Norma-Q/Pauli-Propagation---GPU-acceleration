#include <time.h>
extern "C" int timespec_get(struct timespec*, int);
#include <torch/extension.h>

#include <c10/util/Optional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int64_t kWordBits = 63;

static inline torch::ScalarType coerce_scalar_type(const py::object& coeff_dtype_obj) {
  auto s = py::str(coeff_dtype_obj).cast<std::string>();

  if (s == "torch.float64" || s == "torch.double") return torch::kFloat64;
  if (s == "torch.float32" || s == "torch.float") return torch::kFloat32;
  if (s == "torch.float16" || s == "torch.half") return torch::kFloat16;
  if (s == "torch.bfloat16") return torch::kBFloat16;
  if (s == "torch.complex64") return torch::kComplexFloat;
  if (s == "torch.complex128") return torch::kComplexDouble;
  if (s == "torch.int64" || s == "torch.long") return torch::kInt64;
  if (s == "torch.int32" || s == "torch.int") return torch::kInt32;
  if (s == "torch.bool") return torch::kBool;

  throw std::runtime_error("Unsupported coeff_dtype: " + s);
}

static inline void require_1d_i64(const torch::Tensor& x, const char* name) {
  if (x.scalar_type() != torch::kInt64) {
    throw std::runtime_error(std::string(name) + " must be int64");
  }
  if (x.dim() != 1) {
    throw std::runtime_error(std::string(name) + " must be a 1D int64 tensor");
  }
}

static inline void require_2d_i64(const torch::Tensor& x, const char* name) {
  if (x.scalar_type() != torch::kInt64) {
    throw std::runtime_error(std::string(name) + " must be int64");
  }
  if (x.dim() != 2) {
    throw std::runtime_error(std::string(name) + " must be a 2D int64 tensor");
  }
}

struct PauliKey1D {
  int64_t x;
  int64_t z;

  bool operator==(const PauliKey1D& other) const noexcept {
    return x == other.x && z == other.z;
  }
};

struct PauliKey1DHash {
  std::size_t operator()(const PauliKey1D& key) const noexcept {
    std::size_t h1 = std::hash<int64_t>{}(key.x);
    std::size_t h2 = std::hash<int64_t>{}(key.z);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

struct PauliKeyMW {
  std::vector<int64_t> words;

  bool operator==(const PauliKeyMW& other) const noexcept {
    return words == other.words;
  }
};

struct PauliKeyMWHash {
  std::size_t operator()(const PauliKeyMW& key) const noexcept {
    std::size_t seed = 0;
    for (int64_t word : key.words) {
      std::size_t h = std::hash<int64_t>{}(word);
      seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
merge_rotation_same_and_sin_1d(const torch::Tensor& same_x,
                               const torch::Tensor& same_z,
                               const torch::Tensor& sin_x,
                               const torch::Tensor& sin_z) {
  require_1d_i64(same_x, "same_x");
  require_1d_i64(same_z, "same_z");
  require_1d_i64(sin_x, "sin_x");
  require_1d_i64(sin_z, "sin_z");

  auto device = same_x.device();
  if (same_z.device() != device || sin_x.device() != device || sin_z.device() != device) {
    throw std::runtime_error("merge_rotation_same_and_sin_1d expects all tensors on the same device");
  }

  auto opts_dev = torch::TensorOptions().dtype(torch::kInt64).device(device);
  if (device.is_cuda()) {
    auto same_keys = torch::stack({same_x, same_z}, 1);
    auto sin_keys = torch::stack({sin_x, sin_z}, 1);
    auto all_keys = torch::cat({same_keys, sin_keys}, 0);

    py::module torch_mod = py::module::import("torch");
    py::tuple uniq_out = torch_mod.attr("unique")(all_keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                  py::arg("sorted") = false)
                             .cast<py::tuple>();
    auto uniq = uniq_out[0].cast<torch::Tensor>();
    auto inv = uniq_out[1].cast<torch::Tensor>().to(torch::kInt64);

    int64_t n_same = same_x.size(0);
    auto same_inv = inv.slice(0, 0, n_same);
    auto sin_inv = inv.slice(0, n_same);
    auto keyid_to_row = torch::full({uniq.size(0)}, -1, opts_dev);
    if (n_same > 0) {
      auto same_rows = torch::arange(n_same, opts_dev);
      keyid_to_row.index_put_({same_inv}, same_rows);
    }
    auto row_sin = keyid_to_row.index_select(0, sin_inv);
    auto novel_key_mask = row_sin.lt(0);
    if (!(novel_key_mask.any().item<bool>())) {
      return {same_x, same_z, row_sin};
    }

    auto novel_key_ids =
        torch_mod.attr("unique")(sin_inv.index({novel_key_mask}), py::arg("sorted") = false).cast<torch::Tensor>();
    auto novel_rows = torch::arange(n_same, n_same + novel_key_ids.numel(), opts_dev);
    keyid_to_row.index_put_({novel_key_ids}, novel_rows);
    row_sin = keyid_to_row.index_select(0, sin_inv);

    auto novel_keys = uniq.index_select(0, novel_key_ids);
    auto new_x = torch::cat({same_x, novel_keys.select(1, 0)}, 0);
    auto new_z = torch::cat({same_z, novel_keys.select(1, 1)}, 0);
    return {new_x, new_z, row_sin};
  }

  auto opts_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto same_x_cpu = same_x.contiguous().cpu();
  auto same_z_cpu = same_z.contiguous().cpu();
  auto sin_x_cpu = sin_x.contiguous().cpu();
  auto sin_z_cpu = sin_z.contiguous().cpu();

  const auto* same_x_ptr = same_x_cpu.data_ptr<int64_t>();
  const auto* same_z_ptr = same_z_cpu.data_ptr<int64_t>();
  const auto* sin_x_ptr = sin_x_cpu.data_ptr<int64_t>();
  const auto* sin_z_ptr = sin_z_cpu.data_ptr<int64_t>();

  int64_t n_same = same_x_cpu.size(0);
  int64_t n_sin = sin_x_cpu.size(0);

  std::unordered_map<PauliKey1D, int64_t, PauliKey1DHash> same_rows;
  same_rows.reserve(static_cast<std::size_t>(std::max<int64_t>(1, (n_same + n_sin) * 2)));
  for (int64_t i = 0; i < n_same; ++i) {
    same_rows.emplace(PauliKey1D{same_x_ptr[i], same_z_ptr[i]}, i);
  }

  std::vector<int64_t> row_sin;
  std::vector<int64_t> novel_x_vals;
  std::vector<int64_t> novel_z_vals;
  row_sin.reserve(static_cast<std::size_t>(n_sin));
  novel_x_vals.reserve(static_cast<std::size_t>(n_sin));
  novel_z_vals.reserve(static_cast<std::size_t>(n_sin));

  int64_t next_row = n_same;
  for (int64_t i = 0; i < n_sin; ++i) {
    PauliKey1D key{sin_x_ptr[i], sin_z_ptr[i]};
    auto insert_out = same_rows.emplace(key, next_row);
    row_sin.push_back(insert_out.first->second);
    if (insert_out.second) {
      novel_x_vals.push_back(key.x);
      novel_z_vals.push_back(key.z);
      ++next_row;
    }
  }

  auto row_sin_t = row_sin.empty()
      ? torch::empty({0}, opts_dev)
      : torch::tensor(row_sin, opts_cpu).to(device);

  if (novel_x_vals.empty()) {
    return {same_x, same_z, row_sin_t};
  }

  auto novel_x = torch::tensor(novel_x_vals, opts_cpu).to(device);
  auto novel_z = torch::tensor(novel_z_vals, opts_cpu).to(device);
  auto new_x = torch::cat({same_x, novel_x}, 0);
  auto new_z = torch::cat({same_z, novel_z}, 0);
  return {new_x, new_z, row_sin_t};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
merge_rotation_same_and_sin_mw(const torch::Tensor& same_x,
                               const torch::Tensor& same_z,
                               const torch::Tensor& sin_x,
                               const torch::Tensor& sin_z) {
  require_2d_i64(same_x, "same_x");
  require_2d_i64(same_z, "same_z");
  require_2d_i64(sin_x, "sin_x");
  require_2d_i64(sin_z, "sin_z");

  auto device = same_x.device();
  if (same_z.device() != device || sin_x.device() != device || sin_z.device() != device) {
    throw std::runtime_error("merge_rotation_same_and_sin_mw expects all tensors on the same device");
  }

  auto opts_dev = torch::TensorOptions().dtype(torch::kInt64).device(device);
  if (device.is_cuda()) {
    auto same_keys = torch::cat({same_x, same_z}, 1);
    auto sin_keys = torch::cat({sin_x, sin_z}, 1);
    auto all_keys = torch::cat({same_keys, sin_keys}, 0);

    py::module torch_mod = py::module::import("torch");
    py::tuple uniq_out = torch_mod.attr("unique")(all_keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                  py::arg("sorted") = false)
                             .cast<py::tuple>();
    auto uniq = uniq_out[0].cast<torch::Tensor>();
    auto inv = uniq_out[1].cast<torch::Tensor>().to(torch::kInt64);

    int64_t n_same = same_x.size(0);
    int64_t n_words = same_x.size(1);
    auto same_inv = inv.slice(0, 0, n_same);
    auto sin_inv = inv.slice(0, n_same);
    auto keyid_to_row = torch::full({uniq.size(0)}, -1, opts_dev);
    if (n_same > 0) {
      auto same_rows = torch::arange(n_same, opts_dev);
      keyid_to_row.index_put_({same_inv}, same_rows);
    }
    auto row_sin = keyid_to_row.index_select(0, sin_inv);
    auto novel_key_mask = row_sin.lt(0);
    if (!(novel_key_mask.any().item<bool>())) {
      return {same_x, same_z, row_sin};
    }

    auto novel_key_ids =
        torch_mod.attr("unique")(sin_inv.index({novel_key_mask}), py::arg("sorted") = false).cast<torch::Tensor>();
    auto novel_rows = torch::arange(n_same, n_same + novel_key_ids.numel(), opts_dev);
    keyid_to_row.index_put_({novel_key_ids}, novel_rows);
    row_sin = keyid_to_row.index_select(0, sin_inv);

    auto novel_keys = uniq.index_select(0, novel_key_ids);
    auto new_x = torch::cat({same_x, novel_keys.slice(1, 0, n_words)}, 0);
    auto new_z = torch::cat({same_z, novel_keys.slice(1, n_words, 2 * n_words)}, 0);
    return {new_x, new_z, row_sin};
  }

  auto opts_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto same_x_cpu = same_x.contiguous().cpu();
  auto same_z_cpu = same_z.contiguous().cpu();
  auto sin_x_cpu = sin_x.contiguous().cpu();
  auto sin_z_cpu = sin_z.contiguous().cpu();

  auto same_x_acc = same_x_cpu.accessor<int64_t, 2>();
  auto same_z_acc = same_z_cpu.accessor<int64_t, 2>();
  auto sin_x_acc = sin_x_cpu.accessor<int64_t, 2>();
  auto sin_z_acc = sin_z_cpu.accessor<int64_t, 2>();

  int64_t n_same = same_x_cpu.size(0);
  int64_t n_sin = sin_x_cpu.size(0);
  int64_t n_words = same_x_cpu.size(1);

  std::unordered_map<PauliKeyMW, int64_t, PauliKeyMWHash> same_rows;
  same_rows.reserve(static_cast<std::size_t>(std::max<int64_t>(1, (n_same + n_sin) * 2)));
  for (int64_t i = 0; i < n_same; ++i) {
    PauliKeyMW key;
    key.words.reserve(static_cast<std::size_t>(2 * n_words));
    for (int64_t w = 0; w < n_words; ++w) {
      key.words.push_back(same_x_acc[i][w]);
    }
    for (int64_t w = 0; w < n_words; ++w) {
      key.words.push_back(same_z_acc[i][w]);
    }
    same_rows.emplace(std::move(key), i);
  }

  std::vector<int64_t> row_sin;
  std::vector<int64_t> novel_x_vals;
  std::vector<int64_t> novel_z_vals;
  row_sin.reserve(static_cast<std::size_t>(n_sin));
  novel_x_vals.reserve(static_cast<std::size_t>(n_sin * n_words));
  novel_z_vals.reserve(static_cast<std::size_t>(n_sin * n_words));

  int64_t next_row = n_same;
  for (int64_t i = 0; i < n_sin; ++i) {
    PauliKeyMW key;
    key.words.reserve(static_cast<std::size_t>(2 * n_words));
    for (int64_t w = 0; w < n_words; ++w) {
      key.words.push_back(sin_x_acc[i][w]);
    }
    for (int64_t w = 0; w < n_words; ++w) {
      key.words.push_back(sin_z_acc[i][w]);
    }

    auto insert_out = same_rows.emplace(key, next_row);
    row_sin.push_back(insert_out.first->second);
    if (insert_out.second) {
      for (int64_t w = 0; w < n_words; ++w) {
        novel_x_vals.push_back(key.words[static_cast<std::size_t>(w)]);
      }
      for (int64_t w = 0; w < n_words; ++w) {
        novel_z_vals.push_back(key.words[static_cast<std::size_t>(n_words + w)]);
      }
      ++next_row;
    }
  }

  auto row_sin_t = row_sin.empty()
      ? torch::empty({0}, opts_dev)
      : torch::tensor(row_sin, opts_cpu).to(device);

  if (novel_x_vals.empty()) {
    return {same_x, same_z, row_sin_t};
  }

  int64_t n_novel = static_cast<int64_t>(novel_x_vals.size()) / n_words;
  auto novel_x = torch::tensor(novel_x_vals, opts_cpu).view({n_novel, n_words}).to(device);
  auto novel_z = torch::tensor(novel_z_vals, opts_cpu).view({n_novel, n_words}).to(device);
  auto new_x = torch::cat({same_x, novel_x}, 0);
  auto new_z = torch::cat({same_z, novel_z}, 0);
  return {new_x, new_z, row_sin_t};
}

static inline std::pair<int64_t, int64_t> word_bit(int64_t q) {
  if (q < 0) {
    throw std::runtime_error("qubit index must be >= 0");
  }
  return {q / kWordBits, q % kWordBits};
}

torch::Tensor popcount_u64(const torch::Tensor& x) {
  auto count = torch::zeros_like(x, torch::TensorOptions().dtype(torch::kInt64));
  for (int i = 0; i < 64; i++) {
    count = count + torch::bitwise_and(torch::bitwise_right_shift(x, i), 1);
  }
  return count;
}

torch::Tensor popcount_sum_words(const torch::Tensor& x) {
  auto c = popcount_u64(x);
  if (x.dim() == 2) {
    return c.sum(1);
  }
  return c;
}

bool truncation_is_effective(const torch::Tensor& x_mask, int64_t max_weight, double weight_x, double weight_y,
                             double weight_z) {
  if (weight_x < 0.0 || weight_y < 0.0 || weight_z < 0.0) {
    return true;
  }
  double max_axis_weight = std::max({weight_x, weight_y, weight_z});
  if (max_axis_weight <= 0.0) {
    return false;
  }
  int64_t n_words = x_mask.dim() == 2 ? x_mask.size(1) : 1;
  double max_possible_weight = static_cast<double>(n_words * kWordBits) * max_axis_weight;
  return static_cast<double>(max_weight) < max_possible_weight;
}

torch::Tensor truncate_terms_mask(const torch::Tensor& x_mask, const torch::Tensor& z_mask, int64_t max_weight,
                                  double weight_x, double weight_y, double weight_z) {
  if (!truncation_is_effective(x_mask, max_weight, weight_x, weight_y, weight_z)) {
    return torch::ones({x_mask.size(0)}, torch::TensorOptions().dtype(torch::kBool).device(x_mask.device()));
  }
  auto x_cnt = popcount_sum_words(x_mask & (~z_mask)).to(torch::kFloat64);
  auto y_cnt = popcount_sum_words(x_mask & z_mask).to(torch::kFloat64);
  auto z_cnt = popcount_sum_words((~x_mask) & z_mask).to(torch::kFloat64);
  auto weighted = x_cnt * weight_x + y_cnt * weight_y + z_cnt * weight_z;
  return weighted <= static_cast<double>(max_weight);
}

torch::Tensor phase_sign_tensor(const torch::Tensor& x_mask, const torch::Tensor& z_mask, const torch::Tensor& gx_t,
                                const torch::Tensor& gz_t) {
  auto pX = x_mask & (~z_mask);
  auto pY = x_mask & z_mask;
  auto pZ = (~x_mask) & z_mask;

  auto gX = gx_t & (~gz_t);
  auto gY = gx_t & gz_t;
  auto gZ = (~gx_t) & gz_t;

  auto cnt_xy = popcount_u64(pX & gY);
  auto cnt_zx = popcount_u64(pZ & gX);
  auto cnt_yz = popcount_u64(pY & gZ);
  auto cnt_yx = popcount_u64(pY & gX);
  auto cnt_xz = popcount_u64(pX & gZ);
  auto cnt_zy = popcount_u64(pZ & gY);

  auto s = (cnt_xy + cnt_zx + cnt_yz + 3 * (cnt_yx + cnt_xz + cnt_zy)) & 3;

  auto neg_one = -torch::ones_like(s, torch::TensorOptions().dtype(torch::kInt64));
  auto pos_one = torch::ones_like(s, torch::TensorOptions().dtype(torch::kInt64));
  auto zero = torch::zeros_like(s, torch::TensorOptions().dtype(torch::kInt64));
  return torch::where(s == 1, pos_one, torch::where(s == 3, neg_one, zero));
}

torch::Tensor phase_sign_tensor_mw(const torch::Tensor& x_mask, const torch::Tensor& z_mask,
                                   const torch::Tensor& gx_words, const torch::Tensor& gz_words) {
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (gx_words.scalar_type() != torch::kInt64 || gz_words.scalar_type() != torch::kInt64 || gx_words.dim() != 1 ||
      gz_words.dim() != 1) {
    throw std::runtime_error("gx_words/gz_words must be 1D int64 tensors");
  }
  if (x_mask.size(1) != gx_words.numel() || z_mask.size(1) != gz_words.numel()) {
    throw std::runtime_error("gx_words/gz_words length must match mask word dimension");
  }

  auto gx_t = gx_words.unsqueeze(0);
  auto gz_t = gz_words.unsqueeze(0);

  auto pX = x_mask & (~z_mask);
  auto pY = x_mask & z_mask;
  auto pZ = (~x_mask) & z_mask;

  auto gX = gx_t & (~gz_t);
  auto gY = gx_t & gz_t;
  auto gZ = (~gx_t) & gz_t;

  auto cnt_xy = popcount_sum_words(pX & gY);
  auto cnt_zx = popcount_sum_words(pZ & gX);
  auto cnt_yz = popcount_sum_words(pY & gZ);
  auto cnt_yx = popcount_sum_words(pY & gX);
  auto cnt_xz = popcount_sum_words(pX & gZ);
  auto cnt_zy = popcount_sum_words(pZ & gY);

  auto s = (cnt_xy + cnt_zx + cnt_yz + 3 * (cnt_yx + cnt_xz + cnt_zy)) & 3;

  auto neg_one = -torch::ones_like(s, torch::TensorOptions().dtype(torch::kInt64));
  auto pos_one = torch::ones_like(s, torch::TensorOptions().dtype(torch::kInt64));
  auto zero = torch::zeros_like(s, torch::TensorOptions().dtype(torch::kInt64));
  return torch::where(s == 1, pos_one, torch::where(s == 3, neg_one, zero));
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_clifford(const std::string& symbol,
                                                                                const std::vector<int64_t>& qubits,
                                                                                const torch::Tensor& x_mask,
                                                                                const torch::Tensor& z_mask,
                                                                                torch::ScalarType coeff_dtype) {
  auto device = x_mask.device();
  auto n = x_mask.numel();
  auto coeff_sign = torch::ones({n}, torch::TensorOptions().dtype(coeff_dtype).device(device));

  if (symbol == "H") {
    int64_t q = qubits.at(0);
    if (q < 0 || q >= 63) {
      throw std::runtime_error("apply_clifford(1D) supports q in [0, 62]");
    }
    auto bit = torch::scalar_tensor(int64_t(1) << q, torch::TensorOptions().dtype(torch::kInt64).device(device));

    auto x_bit = (x_mask & bit).ne(0);
    auto z_bit = (z_mask & bit).ne(0);
    auto swap = x_bit ^ z_bit;

    auto new_x = x_mask ^ (swap.to(torch::kInt64) * bit);
    auto new_z = z_mask ^ (swap.to(torch::kInt64) * bit);

    auto y_mask = x_bit & z_bit;
    auto ones = torch::ones_like(coeff_sign);
    auto neg_ones = -torch::ones_like(coeff_sign);
    coeff_sign = coeff_sign * torch::where(y_mask, neg_ones, ones);
    return {new_x, new_z, coeff_sign};
  }

  if (symbol == "S") {
    int64_t q = qubits.at(0);
    if (q < 0 || q >= 63) {
      throw std::runtime_error("apply_clifford(1D) supports q in [0, 62]");
    }
    auto bit = torch::scalar_tensor(int64_t(1) << q, torch::TensorOptions().dtype(torch::kInt64).device(device));

    auto x_bit = (x_mask & bit).ne(0);
    auto z_bit = (z_mask & bit).ne(0);

    auto x_only = x_bit & (~z_bit);
    auto new_z = z_mask ^ (x_bit.to(torch::kInt64) * bit);

    auto ones = torch::ones_like(coeff_sign);
    auto neg_ones = -torch::ones_like(coeff_sign);
    coeff_sign = coeff_sign * torch::where(x_only, neg_ones, ones);
    return {x_mask, new_z, coeff_sign};
  }

  if (symbol == "CNOT") {
    int64_t c = qubits.at(0);
    int64_t t = qubits.at(1);
    if (c < 0 || c >= 63 || t < 0 || t >= 63) {
      throw std::runtime_error("apply_clifford(1D) supports qubits in [0, 62]");
    }
    auto bit_c = torch::scalar_tensor(int64_t(1) << c, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto bit_t = torch::scalar_tensor(int64_t(1) << t, torch::TensorOptions().dtype(torch::kInt64).device(device));

    auto xc = torch::bitwise_right_shift(torch::bitwise_and(x_mask, bit_c), c).to(torch::kInt64);
    auto zc = torch::bitwise_right_shift(torch::bitwise_and(z_mask, bit_c), c).to(torch::kInt64);
    auto xt = torch::bitwise_right_shift(torch::bitwise_and(x_mask, bit_t), t).to(torch::kInt64);
    auto zt = torch::bitwise_right_shift(torch::bitwise_and(z_mask, bit_t), t).to(torch::kInt64);

    auto c_idx = torch::bitwise_xor(xc + torch::bitwise_left_shift(zc, 1), zc);
    auto t_idx = torch::bitwise_xor(xt + torch::bitwise_left_shift(zt, 1), zt);
    auto lookup = t_idx + torch::bitwise_left_shift(c_idx, 2);

    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto opts_f = torch::TensorOptions().dtype(coeff_dtype).device(device);

    auto tab0 = torch::tensor(std::vector<int64_t>({0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}), opts_i64);
    auto tab1 = torch::tensor(std::vector<int64_t>({0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0}), opts_i64);
    auto tab2 = torch::tensor(std::vector<int64_t>({0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0}), opts_i64);
    auto tab3 = torch::tensor(std::vector<int64_t>({0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}), opts_i64);
    auto tab4 = torch::tensor(std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1}), opts_f);

    auto nxc = tab0.index_select(0, lookup);
    auto nzc = tab1.index_select(0, lookup);
    auto nxt = tab2.index_select(0, lookup);
    auto nzt = tab3.index_select(0, lookup);
    auto sign = tab4.index_select(0, lookup);

    auto new_x = x_mask & (~bit_c) & (~bit_t);
    auto new_z = z_mask & (~bit_c) & (~bit_t);
    new_x = new_x | torch::bitwise_left_shift(nxc.to(torch::kInt64), c) |
            torch::bitwise_left_shift(nxt.to(torch::kInt64), t);
    new_z = new_z | torch::bitwise_left_shift(nzc.to(torch::kInt64), c) |
            torch::bitwise_left_shift(nzt.to(torch::kInt64), t);

    coeff_sign = coeff_sign * sign;
    return {new_x, new_z, coeff_sign};
  }

  return {x_mask, z_mask, coeff_sign};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> apply_clifford_mw(const std::string& symbol,
                                                                                   const std::vector<int64_t>& qubits,
                                                                                   const torch::Tensor& x_mask,
                                                                                   const torch::Tensor& z_mask,
                                                                                   torch::ScalarType coeff_dtype) {
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (x_mask.sizes() != z_mask.sizes()) {
    throw std::runtime_error("x_mask/z_mask shape mismatch");
  }

  auto device = x_mask.device();
  auto n = x_mask.size(0);
  auto n_words = x_mask.size(1);
  auto coeff_sign = torch::ones({n}, torch::TensorOptions().dtype(coeff_dtype).device(device));
  auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

  if (symbol == "H") {
    int64_t q = qubits.at(0);
    auto [w, b] = word_bit(q);
    if (w >= n_words) {
      throw std::runtime_error("H gate qubit index exceeds mask word dimension");
    }
    auto bit = torch::scalar_tensor(int64_t(1) << b, opts_i64);

    auto xw = x_mask.select(1, w);
    auto zw = z_mask.select(1, w);
    auto x_bit = (torch::bitwise_and(xw, bit)).ne(0);
    auto z_bit = (torch::bitwise_and(zw, bit)).ne(0);
    auto swap = x_bit ^ z_bit;

    auto new_x = x_mask.clone();
    auto new_z = z_mask.clone();
    new_x.select(1, w).copy_(xw ^ (swap.to(torch::kInt64) * bit));
    new_z.select(1, w).copy_(zw ^ (swap.to(torch::kInt64) * bit));

    auto y_mask = x_bit & z_bit;
    auto ones = torch::ones_like(coeff_sign);
    auto neg_ones = -torch::ones_like(coeff_sign);
    coeff_sign = coeff_sign * torch::where(y_mask, neg_ones, ones);
    return {new_x, new_z, coeff_sign};
  }

  if (symbol == "S") {
    int64_t q = qubits.at(0);
    auto [w, b] = word_bit(q);
    if (w >= n_words) {
      throw std::runtime_error("S gate qubit index exceeds mask word dimension");
    }
    auto bit = torch::scalar_tensor(int64_t(1) << b, opts_i64);
    auto xw = x_mask.select(1, w);
    auto zw = z_mask.select(1, w);
    auto x_bit = (torch::bitwise_and(xw, bit)).ne(0);
    auto z_bit = (torch::bitwise_and(zw, bit)).ne(0);
    auto x_only = x_bit & (~z_bit);

    auto new_z = z_mask.clone();
    new_z.select(1, w).copy_(zw ^ (x_bit.to(torch::kInt64) * bit));

    auto ones = torch::ones_like(coeff_sign);
    auto neg_ones = -torch::ones_like(coeff_sign);
    coeff_sign = coeff_sign * torch::where(x_only, neg_ones, ones);
    return {x_mask, new_z, coeff_sign};
  }

  if (symbol == "CNOT") {
    int64_t c = qubits.at(0);
    int64_t t = qubits.at(1);
    auto [wc, bc] = word_bit(c);
    auto [wt, bt] = word_bit(t);
    if (wc >= n_words || wt >= n_words) {
      throw std::runtime_error("CNOT qubit index exceeds mask word dimension");
    }

    auto bit_c = torch::scalar_tensor(int64_t(1) << bc, opts_i64);
    auto bit_t = torch::scalar_tensor(int64_t(1) << bt, opts_i64);

    auto xc = torch::bitwise_and(x_mask.select(1, wc), bit_c).ne(0).to(torch::kInt64);
    auto zc = torch::bitwise_and(z_mask.select(1, wc), bit_c).ne(0).to(torch::kInt64);
    auto xt = torch::bitwise_and(x_mask.select(1, wt), bit_t).ne(0).to(torch::kInt64);
    auto zt = torch::bitwise_and(z_mask.select(1, wt), bit_t).ne(0).to(torch::kInt64);

    auto c_idx = torch::bitwise_xor(xc + torch::bitwise_left_shift(zc, 1), zc);
    auto t_idx = torch::bitwise_xor(xt + torch::bitwise_left_shift(zt, 1), zt);
    auto lookup = t_idx + torch::bitwise_left_shift(c_idx, 2);

    auto tab0 = torch::tensor(std::vector<int64_t>({0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0}), opts_i64);
    auto tab1 = torch::tensor(std::vector<int64_t>({0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0}), opts_i64);
    auto tab2 = torch::tensor(std::vector<int64_t>({0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0}), opts_i64);
    auto tab3 = torch::tensor(std::vector<int64_t>({0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1}), opts_i64);
    auto tab4 = torch::tensor(std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1}),
                              torch::TensorOptions().dtype(coeff_dtype).device(device));

    auto nxc = tab0.index_select(0, lookup);
    auto nzc = tab1.index_select(0, lookup);
    auto nxt = tab2.index_select(0, lookup);
    auto nzt = tab3.index_select(0, lookup);
    auto sign = tab4.index_select(0, lookup);

    auto new_x = x_mask.clone();
    auto new_z = z_mask.clone();

    auto clear_word = [&](torch::Tensor& m, int64_t w, const torch::Tensor& clear_bits) {
      auto col = m.select(1, w);
      col = torch::bitwise_and(col, torch::bitwise_not(clear_bits));
      m.select(1, w).copy_(col);
    };

    if (wc == wt) {
      auto clear_bits = torch::bitwise_or(bit_c, bit_t);
      clear_word(new_x, wc, clear_bits);
      clear_word(new_z, wc, clear_bits);
    } else {
      clear_word(new_x, wc, bit_c);
      clear_word(new_x, wt, bit_t);
      clear_word(new_z, wc, bit_c);
      clear_word(new_z, wt, bit_t);
    }

    auto set_bits = [&](torch::Tensor& m, int64_t w, const torch::Tensor& bit_val, int64_t b) {
      if (bit_val.numel() == 0) {
        return;
      }
      auto col = m.select(1, w);
      col = torch::bitwise_or(col, torch::bitwise_left_shift(bit_val.to(torch::kInt64), b));
      m.select(1, w).copy_(col);
    };

    set_bits(new_x, wc, nxc, bc);
    set_bits(new_z, wc, nzc, bc);
    set_bits(new_x, wt, nxt, bt);
    set_bits(new_z, wt, nzt, bt);

    coeff_sign = coeff_sign * sign;
    return {new_x, new_z, coeff_sign};
  }

  return {x_mask, z_mask, coeff_sign};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, py::object>
build_clifford_step_cpp(const std::string& symbol, const std::vector<int64_t>& qubits, const torch::Tensor& x_mask,
                        const torch::Tensor& z_mask, const py::object& coeff_dtype_obj, c10::optional<double> min_abs,
                        c10::optional<torch::Tensor> coeffs_cache, int64_t max_weight,
                        double weight_x, double weight_y, double weight_z) {
  (void)max_weight;
  (void)weight_x;
  (void)weight_y;
  (void)weight_z;
  require_1d_i64(x_mask, "x_mask");
  require_1d_i64(z_mask, "z_mask");
  auto n_in = x_mask.numel();

  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);

  auto [new_x0, new_z0, coeff_sign0] = apply_clifford(symbol, qubits, x_mask, z_mask, coeff_dtype);

  torch::Tensor new_x = new_x0;
  torch::Tensor new_z = new_z0;
  torch::Tensor coeff_sign = coeff_sign0;
  torch::Tensor col;

  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  if (do_prune) {
    if (!coeffs_cache.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto pre_mask = (coeff_sign * coeffs_prev).abs() >= min_abs.value();
    auto idx = torch::nonzero(pre_mask).flatten();

    new_x = new_x.index_select(0, idx);
    new_z = new_z.index_select(0, idx);
    coeff_sign = coeff_sign.index_select(0, idx);
    col = idx.to(torch::kInt64);
  } else {
    col = torch::arange(n_in, torch::TensorOptions().dtype(torch::kInt64).device(x_mask.device()));
  }

  auto keys = torch::stack({new_x, new_z}, 1);
  py::module torch_mod = py::module::import("torch");
  py::tuple uniq_out = torch_mod.attr("unique")(keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                  py::arg("sorted") = false)
                           .cast<py::tuple>();
  torch::Tensor uniq = uniq_out[0].cast<torch::Tensor>();
  torch::Tensor inv = uniq_out[1].cast<torch::Tensor>();

  new_x = uniq.select(1, 0);
  new_z = uniq.select(1, 1);

  torch::Tensor row = inv.to(torch::kInt64);

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.numel()};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    auto contrib = coeff_sign * coeffs_prev.index_select(0, col);
    out.index_add_(0, row, contrib);
    coeffs_cache_out = py::cast(out);
  }

  return {new_x, new_z, row, col, coeff_sign, coeffs_cache_out};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, py::object>
build_clifford_step_mw_cpp(const std::string& symbol, const std::vector<int64_t>& qubits, const torch::Tensor& x_mask,
                           const torch::Tensor& z_mask, const py::object& coeff_dtype_obj, c10::optional<double> min_abs,
                           c10::optional<torch::Tensor> coeffs_cache, int64_t max_weight,
                           double weight_x, double weight_y, double weight_z) {
  (void)max_weight;
  (void)weight_x;
  (void)weight_y;
  (void)weight_z;
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (x_mask.sizes() != z_mask.sizes()) {
    throw std::runtime_error("x_mask/z_mask shape mismatch");
  }

  auto n_in = x_mask.size(0);

  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);

  auto [new_x0, new_z0, coeff_sign0] = apply_clifford_mw(symbol, qubits, x_mask, z_mask, coeff_dtype);

  torch::Tensor new_x = new_x0;
  torch::Tensor new_z = new_z0;
  torch::Tensor coeff_sign = coeff_sign0;
  torch::Tensor col;

  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  if (do_prune) {
    if (!coeffs_cache.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto pre_mask = (coeff_sign * coeffs_prev).abs() >= min_abs.value();
    auto idx = torch::nonzero(pre_mask).flatten();

    new_x = new_x.index_select(0, idx);
    new_z = new_z.index_select(0, idx);
    coeff_sign = coeff_sign.index_select(0, idx);
    col = idx.to(torch::kInt64);
  } else {
    col = torch::arange(n_in, torch::TensorOptions().dtype(torch::kInt64).device(x_mask.device()));
  }

  auto keys = torch::cat({new_x, new_z}, 1);
  py::module torch_mod = py::module::import("torch");
  py::tuple uniq_out = torch_mod.attr("unique")(keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                  py::arg("sorted") = false)
                           .cast<py::tuple>();
  torch::Tensor uniq = uniq_out[0].cast<torch::Tensor>();
  torch::Tensor inv = uniq_out[1].cast<torch::Tensor>();

  auto n_words = new_x.size(1);
  new_x = uniq.slice(1, 0, n_words);
  new_z = uniq.slice(1, n_words, 2 * n_words);

  torch::Tensor row = inv.to(torch::kInt64);

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.size(0)};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    auto contrib = coeff_sign * coeffs_prev.index_select(0, col);
    out.index_add_(0, row, contrib);
    coeffs_cache_out = py::cast(out);
  }

  return {new_x, new_z, row, col, coeff_sign, coeffs_cache_out};
}

static std::tuple<torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  py::object>
build_pauli_rotation_step_cpp(int64_t gx, int64_t gz, int64_t param_idx, const torch::Tensor& x_mask,
                              const torch::Tensor& z_mask, const py::object& coeff_dtype_obj,
                              c10::optional<double> min_abs, c10::optional<torch::Tensor> coeffs_cache,
                              c10::optional<torch::Tensor> thetas_t, int64_t max_weight,
                              double weight_x, double weight_y, double weight_z) {
  require_1d_i64(x_mask, "x_mask");
  require_1d_i64(z_mask, "z_mask");
  auto device = x_mask.device();

  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);

  auto gx_t = torch::scalar_tensor(gx, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto gz_t = torch::scalar_tensor(gz, torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto symp = popcount_u64((x_mask & gz_t) ^ (z_mask & gx_t)) & 1;
  auto comm_mask = symp.eq(0);
  auto anti_mask = symp.eq(1);

  auto comm_idx = torch::nonzero(comm_mask).flatten().to(torch::kInt64);
  auto anti_idx = torch::nonzero(anti_mask).flatten().to(torch::kInt64);

  auto comm_x = x_mask.index_select(0, comm_idx);
  auto comm_z = z_mask.index_select(0, comm_idx);

  auto anti_x = x_mask.index_select(0, anti_idx);
  auto anti_z = z_mask.index_select(0, anti_idx);

  auto cos_x = anti_x;
  auto cos_z = anti_z;
  auto sin_x = anti_x ^ gx_t;
  auto sin_z = anti_z ^ gz_t;

  torch::Tensor cos_idx = anti_idx;
  torch::Tensor sin_idx = anti_idx;

  torch::Tensor sin_sign;
  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  torch::Tensor cos_t;
  torch::Tensor sin_t;

  if (do_prune) {
    if (!coeffs_cache.has_value() || !thetas_t.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache/thetas_t is None");
    }
    coeffs_prev = coeffs_cache.value();

    auto theta = thetas_t.value().index({param_idx});
    cos_t = torch::cos(theta);
    sin_t = torch::sin(theta);

    auto mask_comm = coeffs_prev.index_select(0, comm_idx).abs() >= min_abs.value();
    auto mask_cos = (coeffs_prev.index_select(0, anti_idx) * cos_t).abs() >= min_abs.value();

    auto sin_sign_all = phase_sign_tensor(anti_x, anti_z, gx_t, gz_t).to(coeffs_prev.dtype());
    auto mask_sin = (coeffs_prev.index_select(0, anti_idx) * sin_t * sin_sign_all).abs() >= min_abs.value();

    auto comm_keep = torch::nonzero(mask_comm).flatten().to(torch::kInt64);
    comm_x = comm_x.index_select(0, comm_keep);
    comm_z = comm_z.index_select(0, comm_keep);
    comm_idx = comm_idx.index_select(0, comm_keep);

    auto cos_keep = torch::nonzero(mask_cos).flatten().to(torch::kInt64);
    cos_x = cos_x.index_select(0, cos_keep);
    cos_z = cos_z.index_select(0, cos_keep);
    cos_idx = anti_idx.index_select(0, cos_keep);

    auto sin_keep = torch::nonzero(mask_sin).flatten().to(torch::kInt64);
    sin_x = sin_x.index_select(0, sin_keep);
    sin_z = sin_z.index_select(0, sin_keep);
    sin_idx = anti_idx.index_select(0, sin_keep);
    sin_sign = sin_sign_all.index_select(0, sin_keep);
  } else {
    sin_sign = phase_sign_tensor(anti_x, anti_z, gx_t, gz_t);
  }

  auto cat_x = torch::cat({comm_x, cos_x, sin_x}, 0);
  auto cat_z = torch::cat({comm_z, cos_z, sin_z}, 0);
  auto keys = torch::stack({cat_x, cat_z}, 1);

  py::module torch_mod = py::module::import("torch");
  py::tuple uniq_out = torch_mod.attr("unique")(keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                  py::arg("sorted") = false)
                           .cast<py::tuple>();
  torch::Tensor uniq = uniq_out[0].cast<torch::Tensor>();
  torch::Tensor inv = uniq_out[1].cast<torch::Tensor>();

  auto new_x = uniq.select(1, 0);
  auto new_z = uniq.select(1, 1);

  auto n_comm = comm_idx.numel();
  auto n_cos = cos_idx.numel();
  auto n_sin = sin_idx.numel();

  auto row_comm = inv.slice(0, 0, n_comm).to(torch::kInt64);
  auto row_cos = inv.slice(0, n_comm, n_comm + n_cos).to(torch::kInt64);
  auto row_sin = inv.slice(0, n_comm + n_cos).to(torch::kInt64);

  auto col_comm = comm_idx;
  auto col_cos = cos_idx;
  auto col_sin = sin_idx;

  std::vector<int64_t> comm_sz{row_comm.numel()};
  std::vector<int64_t> cos_sz{row_cos.numel()};
  auto val_comm = torch::ones(comm_sz, torch::TensorOptions().dtype(coeff_dtype).device(device));
  auto val_cos = torch::ones(cos_sz, torch::TensorOptions().dtype(coeff_dtype).device(device));
  auto val_sin = sin_sign.to(coeff_dtype);

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.numel()};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    out.index_add_(0, row_comm, coeffs_prev.index_select(0, col_comm));
    out.index_add_(0, row_cos, coeffs_prev.index_select(0, col_cos) * cos_t);
    out.index_add_(0, row_sin, coeffs_prev.index_select(0, col_sin) * sin_t * sin_sign.to(coeffs_prev.dtype()));
    coeffs_cache_out = py::cast(out);
  }

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);

    auto filter_triplet = [&](torch::Tensor& row, torch::Tensor& col, torch::Tensor& val) {
      if (row.numel() == 0) {
        return;
      }
      auto keep_rows = mask.index_select(0, row);
      auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
      row = row.index_select(0, keep_idx);
      col = col.index_select(0, keep_idx);
      val = val.index_select(0, keep_idx);
      row = row_map.index_select(0, row);
    };

    filter_triplet(row_comm, col_comm, val_comm);
    filter_triplet(row_cos, col_cos, val_cos);
    filter_triplet(row_sin, col_sin, val_sin);

    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);

    if (do_prune) {
      auto out_coeffs = coeffs_cache_out.cast<torch::Tensor>();
      out_coeffs = out_coeffs.index_select(0, out_idx);
      coeffs_cache_out = py::cast(out_coeffs);
    }
  }

  return std::make_tuple(new_x,
                         new_z,
                         row_comm,
                         col_comm,
                         val_comm,
                         row_cos,
                         col_cos,
                         val_cos,
                         row_sin,
                         col_sin,
                         val_sin,
                         coeffs_cache_out);
}

static std::tuple<torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  py::object>
build_pauli_rotation_step_mw_cpp(const torch::Tensor& gx_words, const torch::Tensor& gz_words, int64_t param_idx,
                                 const torch::Tensor& x_mask, const torch::Tensor& z_mask,
                                 const py::object& coeff_dtype_obj, c10::optional<double> min_abs,
                                 c10::optional<torch::Tensor> coeffs_cache, c10::optional<torch::Tensor> thetas_t,
                                 int64_t max_weight, double weight_x, double weight_y, double weight_z) {
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (x_mask.sizes() != z_mask.sizes()) {
    throw std::runtime_error("x_mask/z_mask shape mismatch");
  }
  if (gx_words.scalar_type() != torch::kInt64 || gz_words.scalar_type() != torch::kInt64 || gx_words.dim() != 1 ||
      gz_words.dim() != 1) {
    throw std::runtime_error("gx_words/gz_words must be 1D int64 tensors");
  }
  if (x_mask.size(1) != gx_words.numel() || z_mask.size(1) != gz_words.numel()) {
    throw std::runtime_error("gx_words/gz_words length must match mask word dimension");
  }

  auto device = x_mask.device();

  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);

  auto gx_t = gx_words.unsqueeze(0);
  auto gz_t = gz_words.unsqueeze(0);

  auto symp = popcount_u64((torch::bitwise_and(x_mask, gz_t)) ^ (torch::bitwise_and(z_mask, gx_t))).sum(1) & 1;
  auto comm_mask = symp.eq(0);
  auto anti_mask = symp.eq(1);

  auto comm_idx = torch::nonzero(comm_mask).flatten().to(torch::kInt64);
  auto anti_idx = torch::nonzero(anti_mask).flatten().to(torch::kInt64);

  auto comm_x = x_mask.index_select(0, comm_idx);
  auto comm_z = z_mask.index_select(0, comm_idx);

  auto anti_x = x_mask.index_select(0, anti_idx);
  auto anti_z = z_mask.index_select(0, anti_idx);

  auto cos_x = anti_x;
  auto cos_z = anti_z;
  auto sin_x = anti_x ^ gx_t;
  auto sin_z = anti_z ^ gz_t;

  torch::Tensor cos_idx = anti_idx;
  torch::Tensor sin_idx = anti_idx;

  torch::Tensor sin_sign;
  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  torch::Tensor cos_t;
  torch::Tensor sin_t;

  if (do_prune) {
    if (!coeffs_cache.has_value() || !thetas_t.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache/thetas_t is None");
    }
    coeffs_prev = coeffs_cache.value();

    auto theta = thetas_t.value().index({param_idx});
    cos_t = torch::cos(theta);
    sin_t = torch::sin(theta);

    auto mask_comm = coeffs_prev.index_select(0, comm_idx).abs() >= min_abs.value();
    auto mask_cos = (coeffs_prev.index_select(0, anti_idx) * cos_t).abs() >= min_abs.value();

    auto sin_sign_all = phase_sign_tensor_mw(anti_x, anti_z, gx_words, gz_words).to(coeffs_prev.dtype());
    auto mask_sin = (coeffs_prev.index_select(0, anti_idx) * sin_t * sin_sign_all).abs() >= min_abs.value();

    auto comm_keep = torch::nonzero(mask_comm).flatten().to(torch::kInt64);
    comm_x = comm_x.index_select(0, comm_keep);
    comm_z = comm_z.index_select(0, comm_keep);
    comm_idx = comm_idx.index_select(0, comm_keep);

    auto cos_keep = torch::nonzero(mask_cos).flatten().to(torch::kInt64);
    cos_x = cos_x.index_select(0, cos_keep);
    cos_z = cos_z.index_select(0, cos_keep);
    cos_idx = anti_idx.index_select(0, cos_keep);

    auto sin_keep = torch::nonzero(mask_sin).flatten().to(torch::kInt64);
    sin_x = sin_x.index_select(0, sin_keep);
    sin_z = sin_z.index_select(0, sin_keep);
    sin_idx = anti_idx.index_select(0, sin_keep);
    sin_sign = sin_sign_all.index_select(0, sin_keep);
  } else {
    sin_sign = phase_sign_tensor_mw(anti_x, anti_z, gx_words, gz_words);
  }

  auto cat_x = torch::cat({comm_x, cos_x, sin_x}, 0);
  auto cat_z = torch::cat({comm_z, cos_z, sin_z}, 0);
  auto keys = torch::cat({cat_x, cat_z}, 1);

  py::module torch_mod = py::module::import("torch");
  py::tuple uniq_out = torch_mod.attr("unique")(keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                  py::arg("sorted") = false)
                           .cast<py::tuple>();
  torch::Tensor uniq = uniq_out[0].cast<torch::Tensor>();
  torch::Tensor inv = uniq_out[1].cast<torch::Tensor>();

  auto n_words = x_mask.size(1);
  auto new_x = uniq.slice(1, 0, n_words);
  auto new_z = uniq.slice(1, n_words, 2 * n_words);

  auto n_comm = comm_idx.numel();
  auto n_cos = cos_idx.numel();
  auto n_sin = sin_idx.numel();

  auto row_comm = inv.slice(0, 0, n_comm).to(torch::kInt64);
  auto row_cos = inv.slice(0, n_comm, n_comm + n_cos).to(torch::kInt64);
  auto row_sin = inv.slice(0, n_comm + n_cos).to(torch::kInt64);

  auto col_comm = comm_idx;
  auto col_cos = cos_idx;
  auto col_sin = sin_idx;

  std::vector<int64_t> comm_sz{row_comm.numel()};
  std::vector<int64_t> cos_sz{row_cos.numel()};
  auto val_comm = torch::ones(comm_sz, torch::TensorOptions().dtype(coeff_dtype).device(device));
  auto val_cos = torch::ones(cos_sz, torch::TensorOptions().dtype(coeff_dtype).device(device));
  auto val_sin = sin_sign.to(coeff_dtype);

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.size(0)};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    out.index_add_(0, row_comm, coeffs_prev.index_select(0, col_comm));
    out.index_add_(0, row_cos, coeffs_prev.index_select(0, col_cos) * cos_t);
    out.index_add_(0, row_sin, coeffs_prev.index_select(0, col_sin) * sin_t * sin_sign.to(coeffs_prev.dtype()));
    coeffs_cache_out = py::cast(out);
  }

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);

    auto filter_triplet = [&](torch::Tensor& row, torch::Tensor& col, torch::Tensor& val) {
      if (row.numel() == 0) {
        return;
      }
      auto keep_rows = mask.index_select(0, row);
      auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
      row = row.index_select(0, keep_idx);
      col = col.index_select(0, keep_idx);
      val = val.index_select(0, keep_idx);
      row = row_map.index_select(0, row);
    };

    filter_triplet(row_comm, col_comm, val_comm);
    filter_triplet(row_cos, col_cos, val_cos);
    filter_triplet(row_sin, col_sin, val_sin);

    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);

    if (do_prune) {
      auto out_coeffs = coeffs_cache_out.cast<torch::Tensor>();
      out_coeffs = out_coeffs.index_select(0, out_idx);
      coeffs_cache_out = py::cast(out_coeffs);
    }
  }

  return std::make_tuple(new_x,
                         new_z,
                         row_comm,
                         col_comm,
                         val_comm,
                         row_cos,
                         col_cos,
                         val_cos,
                         row_sin,
                         col_sin,
                         val_sin,
                         coeffs_cache_out);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_pauli_rotation_anti_sin_cpp(int64_t gx, int64_t gz, const py::object& coeff_dtype_obj,
                                  const torch::Tensor& x_mask, const torch::Tensor& z_mask) {
  require_1d_i64(x_mask, "x_mask");
  require_1d_i64(z_mask, "z_mask");
  auto device = x_mask.device();
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);

  auto gx_t = torch::scalar_tensor(gx, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto gz_t = torch::scalar_tensor(gz, torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto sin_x = x_mask ^ gx_t;
  auto sin_z = z_mask ^ gz_t;
  auto sin_sign = phase_sign_tensor(x_mask, z_mask, gx_t, gz_t).to(coeff_dtype);
  return std::make_tuple(sin_x, sin_z, sin_sign);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
build_pauli_rotation_anti_sin_mw_cpp(const torch::Tensor& gx_words, const torch::Tensor& gz_words,
                                     const py::object& coeff_dtype_obj, const torch::Tensor& x_mask,
                                     const torch::Tensor& z_mask) {
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (x_mask.sizes() != z_mask.sizes()) {
    throw std::runtime_error("x_mask/z_mask shape mismatch");
  }
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);
  auto gx_t = gx_words.unsqueeze(0);
  auto gz_t = gz_words.unsqueeze(0);

  auto sin_x = x_mask ^ gx_t;
  auto sin_z = z_mask ^ gz_t;
  auto sin_sign = phase_sign_tensor_mw(x_mask, z_mask, gx_words, gz_words).to(coeff_dtype);
  return std::make_tuple(sin_x, sin_z, sin_sign);
}

static std::tuple<torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  py::object>
build_pauli_rotation_step_implicit_cpp(int64_t gx, int64_t gz, int64_t param_idx, const torch::Tensor& x_mask,
                                       const torch::Tensor& z_mask, const py::object& coeff_dtype_obj,
                                       c10::optional<double> min_abs, c10::optional<torch::Tensor> coeffs_cache,
                                       c10::optional<torch::Tensor> thetas_t, int64_t max_weight,
                                       double weight_x, double weight_y, double weight_z) {
  require_1d_i64(x_mask, "x_mask");
  require_1d_i64(z_mask, "z_mask");
  auto device = x_mask.device();
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);

  auto gx_t = torch::scalar_tensor(gx, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto gz_t = torch::scalar_tensor(gz, torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto symp = popcount_u64((x_mask & gz_t) ^ (z_mask & gx_t)) & 1;
  auto comm_idx = torch::nonzero(symp.eq(0)).flatten().to(torch::kInt64);
  auto anti_idx = torch::nonzero(symp.eq(1)).flatten().to(torch::kInt64);

  auto anti_x = x_mask.index_select(0, anti_idx);
  auto anti_z = z_mask.index_select(0, anti_idx);
  auto sin_x = anti_x ^ gx_t;
  auto sin_z = anti_z ^ gz_t;

  auto same_idx = comm_idx;
  auto cos_idx = anti_idx;
  auto sin_idx = anti_idx;

  torch::Tensor sin_sign;
  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  torch::Tensor cos_t;
  torch::Tensor sin_t;

  if (do_prune) {
    if (!coeffs_cache.has_value() || !thetas_t.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache/thetas_t is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto theta = thetas_t.value().index({param_idx});
    cos_t = torch::cos(theta);
    sin_t = torch::sin(theta);

    auto mask_comm = coeffs_prev.index_select(0, comm_idx).abs() >= min_abs.value();
    auto mask_cos = (coeffs_prev.index_select(0, anti_idx) * cos_t).abs() >= min_abs.value();
    auto sin_sign_all = phase_sign_tensor(anti_x, anti_z, gx_t, gz_t).to(coeffs_prev.dtype());
    auto mask_sin = (coeffs_prev.index_select(0, anti_idx) * sin_t * sin_sign_all).abs() >= min_abs.value();

    auto comm_keep = torch::nonzero(mask_comm).flatten().to(torch::kInt64);
    auto cos_keep = torch::nonzero(mask_cos).flatten().to(torch::kInt64);
    auto sin_keep = torch::nonzero(mask_sin).flatten().to(torch::kInt64);

    same_idx = torch::cat({comm_idx.index_select(0, comm_keep), anti_idx.index_select(0, cos_keep)}, 0);
    cos_idx = anti_idx.index_select(0, cos_keep);
    sin_x = sin_x.index_select(0, sin_keep);
    sin_z = sin_z.index_select(0, sin_keep);
    sin_idx = anti_idx.index_select(0, sin_keep);
    sin_sign = sin_sign_all.index_select(0, sin_keep);
  } else {
    same_idx = torch::cat({comm_idx, anti_idx}, 0);
    cos_idx = anti_idx;
    sin_sign = phase_sign_tensor(anti_x, anti_z, gx_t, gz_t);
  }

  auto same_x = x_mask.index_select(0, same_idx);
  auto same_z = z_mask.index_select(0, same_idx);
  auto n_same = same_idx.numel();
  auto merge_out = merge_rotation_same_and_sin_1d(same_x, same_z, sin_x, sin_z);
  auto new_x = std::get<0>(merge_out);
  auto new_z = std::get<1>(merge_out);
  auto row_sin = std::get<2>(merge_out);
  auto anti_same_pos = torch::arange(n_same - cos_idx.numel(), n_same,
                                     torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto val_sin = sin_sign.to(coeff_dtype);

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    auto same_comm_n = same_idx.numel() - cos_idx.numel();
    auto same_vals = torch::cat({
        coeffs_prev.index_select(0, same_idx.slice(0, 0, same_comm_n)),
        coeffs_prev.index_select(0, cos_idx) * cos_t,
    }, 0);
    std::vector<int64_t> out_sz{new_x.numel()};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    if (same_vals.numel() > 0) {
      out.index_put_({torch::arange(n_same, torch::TensorOptions().dtype(torch::kInt64).device(device))}, same_vals);
    }
    if (sin_idx.numel() > 0) {
      out.index_add_(0, row_sin, coeffs_prev.index_select(0, sin_idx) * sin_t * sin_sign.to(coeffs_prev.dtype()));
    }
    coeffs_cache_out = py::cast(out);
  }

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    auto keep_same = mask.slice(0, 0, n_same);
    same_idx = same_idx.index({keep_same});
    if (anti_same_pos.numel() > 0) {
      auto anti_keep = keep_same.index_select(0, anti_same_pos);
      auto kept_anti = anti_same_pos.index({anti_keep});
      anti_same_pos = row_map.index_select(0, kept_anti);
    }
    if (row_sin.numel() > 0) {
      auto keep_rows = mask.index_select(0, row_sin);
      auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
      row_sin = row_sin.index_select(0, keep_idx);
      sin_idx = sin_idx.index_select(0, keep_idx);
      val_sin = val_sin.index_select(0, keep_idx);
      row_sin = row_map.index_select(0, row_sin);
    }
    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);
    if (do_prune) {
      auto out_coeffs = coeffs_cache_out.cast<torch::Tensor>();
      out_coeffs = out_coeffs.index_select(0, out_idx);
      coeffs_cache_out = py::cast(out_coeffs);
    }
  }

  return std::make_tuple(new_x, new_z, same_idx, anti_same_pos, row_sin, sin_idx, val_sin, coeffs_cache_out);
}

static std::tuple<torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor,
                  py::object>
build_pauli_rotation_step_implicit_mw_cpp(const torch::Tensor& gx_words, const torch::Tensor& gz_words, int64_t param_idx,
                                          const torch::Tensor& x_mask, const torch::Tensor& z_mask,
                                          const py::object& coeff_dtype_obj, c10::optional<double> min_abs,
                                          c10::optional<torch::Tensor> coeffs_cache, c10::optional<torch::Tensor> thetas_t,
                                          int64_t max_weight, double weight_x, double weight_y, double weight_z) {
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (x_mask.sizes() != z_mask.sizes()) {
    throw std::runtime_error("x_mask/z_mask shape mismatch");
  }
  auto device = x_mask.device();
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);
  auto gx_t = gx_words.unsqueeze(0);
  auto gz_t = gz_words.unsqueeze(0);

  auto symp = popcount_u64((torch::bitwise_and(x_mask, gz_t)) ^ (torch::bitwise_and(z_mask, gx_t))).sum(1) & 1;
  auto comm_idx = torch::nonzero(symp.eq(0)).flatten().to(torch::kInt64);
  auto anti_idx = torch::nonzero(symp.eq(1)).flatten().to(torch::kInt64);

  auto anti_x = x_mask.index_select(0, anti_idx);
  auto anti_z = z_mask.index_select(0, anti_idx);
  auto sin_x = anti_x ^ gx_t;
  auto sin_z = anti_z ^ gz_t;

  auto same_idx = comm_idx;
  auto cos_idx = anti_idx;
  auto sin_idx = anti_idx;

  torch::Tensor sin_sign;
  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  torch::Tensor cos_t;
  torch::Tensor sin_t;

  if (do_prune) {
    if (!coeffs_cache.has_value() || !thetas_t.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache/thetas_t is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto theta = thetas_t.value().index({param_idx});
    cos_t = torch::cos(theta);
    sin_t = torch::sin(theta);

    auto mask_comm = coeffs_prev.index_select(0, comm_idx).abs() >= min_abs.value();
    auto mask_cos = (coeffs_prev.index_select(0, anti_idx) * cos_t).abs() >= min_abs.value();
    auto sin_sign_all = phase_sign_tensor_mw(anti_x, anti_z, gx_words, gz_words).to(coeffs_prev.dtype());
    auto mask_sin = (coeffs_prev.index_select(0, anti_idx) * sin_t * sin_sign_all).abs() >= min_abs.value();

    auto comm_keep = torch::nonzero(mask_comm).flatten().to(torch::kInt64);
    auto cos_keep = torch::nonzero(mask_cos).flatten().to(torch::kInt64);
    auto sin_keep = torch::nonzero(mask_sin).flatten().to(torch::kInt64);

    same_idx = torch::cat({comm_idx.index_select(0, comm_keep), anti_idx.index_select(0, cos_keep)}, 0);
    cos_idx = anti_idx.index_select(0, cos_keep);
    sin_x = sin_x.index_select(0, sin_keep);
    sin_z = sin_z.index_select(0, sin_keep);
    sin_idx = anti_idx.index_select(0, sin_keep);
    sin_sign = sin_sign_all.index_select(0, sin_keep);
  } else {
    same_idx = torch::cat({comm_idx, anti_idx}, 0);
    cos_idx = anti_idx;
    sin_sign = phase_sign_tensor_mw(anti_x, anti_z, gx_words, gz_words);
  }

  auto same_x = x_mask.index_select(0, same_idx);
  auto same_z = z_mask.index_select(0, same_idx);
  auto n_same = same_idx.numel();
  auto merge_out = merge_rotation_same_and_sin_mw(same_x, same_z, sin_x, sin_z);
  auto new_x = std::get<0>(merge_out);
  auto new_z = std::get<1>(merge_out);
  auto row_sin = std::get<2>(merge_out);
  auto anti_same_pos = torch::arange(n_same - cos_idx.numel(), n_same,
                                     torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto val_sin = sin_sign.to(coeff_dtype);

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    auto same_comm_n = same_idx.numel() - cos_idx.numel();
    auto same_vals = torch::cat({
        coeffs_prev.index_select(0, same_idx.slice(0, 0, same_comm_n)),
        coeffs_prev.index_select(0, cos_idx) * cos_t,
    }, 0);
    std::vector<int64_t> out_sz{new_x.size(0)};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    if (same_vals.numel() > 0) {
      out.index_put_({torch::arange(n_same, torch::TensorOptions().dtype(torch::kInt64).device(device))}, same_vals);
    }
    if (sin_idx.numel() > 0) {
      out.index_add_(0, row_sin, coeffs_prev.index_select(0, sin_idx) * sin_t * sin_sign.to(coeffs_prev.dtype()));
    }
    coeffs_cache_out = py::cast(out);
  }

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    auto keep_same = mask.slice(0, 0, n_same);
    same_idx = same_idx.index({keep_same});
    if (anti_same_pos.numel() > 0) {
      auto anti_keep = keep_same.index_select(0, anti_same_pos);
      auto kept_anti = anti_same_pos.index({anti_keep});
      anti_same_pos = row_map.index_select(0, kept_anti);
    }
    if (row_sin.numel() > 0) {
      auto keep_rows = mask.index_select(0, row_sin);
      auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
      row_sin = row_sin.index_select(0, keep_idx);
      sin_idx = sin_idx.index_select(0, keep_idx);
      val_sin = val_sin.index_select(0, keep_idx);
      row_sin = row_map.index_select(0, row_sin);
    }
    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);
    if (do_prune) {
      auto out_coeffs = coeffs_cache_out.cast<torch::Tensor>();
      out_coeffs = out_coeffs.index_select(0, out_idx);
      coeffs_cache_out = py::cast(out_coeffs);
    }
  }

  return std::make_tuple(new_x, new_z, same_idx, anti_same_pos, row_sin, sin_idx, val_sin, coeffs_cache_out);
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, py::object>
build_depolarizing_step_cpp(int64_t qubit, double px, double py, double pz, const torch::Tensor& x_mask,
                            const torch::Tensor& z_mask, const py::object& coeff_dtype_obj,
                            c10::optional<double> min_abs, c10::optional<torch::Tensor> coeffs_cache,
                            int64_t max_weight, double weight_x, double weight_y, double weight_z) {
  require_1d_i64(x_mask, "x_mask");
  require_1d_i64(z_mask, "z_mask");
  if (qubit < 0 || qubit >= 63) {
    throw std::runtime_error("build_depolarizing_step_cpp supports qubit in [0, 62]");
  }

  auto device = x_mask.device();
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);
  auto n = x_mask.numel();

  auto bit = torch::scalar_tensor(int64_t(1) << qubit, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto x_bit = (x_mask & bit).ne(0);
  auto z_bit = (z_mask & bit).ne(0);
  auto x_only = x_bit & (~z_bit);
  auto y_only = x_bit & z_bit;
  auto z_only = (~x_bit) & z_bit;

  auto opts = torch::TensorOptions().dtype(coeff_dtype).device(device);
  auto one = torch::scalar_tensor(1.0, opts);
  auto sx = torch::scalar_tensor(1.0 - 2.0 * py - 2.0 * pz, opts);
  auto sy = torch::scalar_tensor(1.0 - 2.0 * px - 2.0 * pz, opts);
  auto sz = torch::scalar_tensor(1.0 - 2.0 * px - 2.0 * py, opts);

  auto val = torch::ones({n}, opts);
  val = torch::where(x_only, sx, val);
  val = torch::where(y_only, sy, val);
  val = torch::where(z_only, sz, val);

  auto col = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto row = col.clone();
  auto new_x = x_mask;
  auto new_z = z_mask;

  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  if (do_prune) {
    if (!coeffs_cache.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto keep = (val * coeffs_prev).abs() >= min_abs.value();
    auto keep_idx = torch::nonzero(keep).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, keep_idx);
    new_z = new_z.index_select(0, keep_idx);
    row = row.index_select(0, keep_idx);
    col = col.index_select(0, keep_idx);
    val = val.index_select(0, keep_idx);
  }

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto keep_rows = mask.index_select(0, row);
    auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
    row = row.index_select(0, keep_idx);
    col = col.index_select(0, keep_idx);
    val = val.index_select(0, keep_idx);

    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    row = row_map.index_select(0, row);

    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);
  }

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.numel()};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    auto contrib = val * coeffs_prev.index_select(0, col);
    out.index_add_(0, row, contrib);
    coeffs_cache_out = py::cast(out);
  }

  return {new_x, new_z, row, col, val, coeffs_cache_out};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, py::object>
build_depolarizing_step_mw_cpp(int64_t qubit, double px, double py, double pz, const torch::Tensor& x_mask,
                               const torch::Tensor& z_mask, const py::object& coeff_dtype_obj,
                               c10::optional<double> min_abs, c10::optional<torch::Tensor> coeffs_cache,
                               int64_t max_weight, double weight_x, double weight_y, double weight_z) {
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (x_mask.sizes() != z_mask.sizes()) {
    throw std::runtime_error("x_mask/z_mask shape mismatch");
  }

  auto [w, b] = word_bit(qubit);
  if (w >= x_mask.size(1)) {
    throw std::runtime_error("depolarizing qubit index exceeds mask word dimension");
  }

  auto device = x_mask.device();
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);
  auto n = x_mask.size(0);

  auto bit = torch::scalar_tensor(int64_t(1) << b, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto xw = x_mask.select(1, w);
  auto zw = z_mask.select(1, w);
  auto x_bit = (xw & bit).ne(0);
  auto z_bit = (zw & bit).ne(0);
  auto x_only = x_bit & (~z_bit);
  auto y_only = x_bit & z_bit;
  auto z_only = (~x_bit) & z_bit;

  auto opts = torch::TensorOptions().dtype(coeff_dtype).device(device);
  auto sx = torch::scalar_tensor(1.0 - 2.0 * py - 2.0 * pz, opts);
  auto sy = torch::scalar_tensor(1.0 - 2.0 * px - 2.0 * pz, opts);
  auto sz = torch::scalar_tensor(1.0 - 2.0 * px - 2.0 * py, opts);

  auto val = torch::ones({n}, opts);
  val = torch::where(x_only, sx, val);
  val = torch::where(y_only, sy, val);
  val = torch::where(z_only, sz, val);

  auto col = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto row = col.clone();
  auto new_x = x_mask;
  auto new_z = z_mask;

  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  if (do_prune) {
    if (!coeffs_cache.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto keep = (val * coeffs_prev).abs() >= min_abs.value();
    auto keep_idx = torch::nonzero(keep).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, keep_idx);
    new_z = new_z.index_select(0, keep_idx);
    row = row.index_select(0, keep_idx);
    col = col.index_select(0, keep_idx);
    val = val.index_select(0, keep_idx);
  }

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto keep_rows = mask.index_select(0, row);
    auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
    row = row.index_select(0, keep_idx);
    col = col.index_select(0, keep_idx);
    val = val.index_select(0, keep_idx);

    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    row = row_map.index_select(0, row);

    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);
  }

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.size(0)};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    auto contrib = val * coeffs_prev.index_select(0, col);
    out.index_add_(0, row, contrib);
    coeffs_cache_out = py::cast(out);
  }

  return {new_x, new_z, row, col, val, coeffs_cache_out};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, py::object>
build_amplitude_damping_step_cpp(int64_t qubit, double gamma, const torch::Tensor& x_mask,
                                 const torch::Tensor& z_mask, const py::object& coeff_dtype_obj,
                                 c10::optional<double> min_abs, c10::optional<torch::Tensor> coeffs_cache,
                                 int64_t max_weight, double weight_x, double weight_y, double weight_z) {
  require_1d_i64(x_mask, "x_mask");
  require_1d_i64(z_mask, "z_mask");
  if (qubit < 0 || qubit >= 63) {
    throw std::runtime_error("build_amplitude_damping_step_cpp supports qubit in [0, 62]");
  }
  if (gamma < 0.0 || gamma > 1.0) {
    throw std::runtime_error("gamma must be in [0, 1]");
  }

  auto device = x_mask.device();
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);
  auto n = x_mask.numel();

  auto bit = torch::scalar_tensor(int64_t(1) << qubit, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto x_bit = (x_mask & bit).ne(0);
  auto z_bit = (z_mask & bit).ne(0);
  auto x_only = x_bit & (~z_bit);
  auto y_only = x_bit & z_bit;
  auto z_only = (~x_bit) & z_bit;

  auto opts = torch::TensorOptions().dtype(coeff_dtype).device(device);
  auto sq = std::sqrt(std::max(0.0, 1.0 - gamma));
  auto s_xy = torch::scalar_tensor(sq, opts);
  auto s_z = torch::scalar_tensor(1.0 - gamma, opts);

  auto idx_all = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto val_base = torch::ones({n}, opts);
  val_base = torch::where(x_only | y_only, s_xy, val_base);
  val_base = torch::where(z_only, s_z, val_base);

  auto z_idx = torch::nonzero(z_only).flatten().to(torch::kInt64);
  auto x_i = x_mask.index_select(0, z_idx);
  auto z_i = z_mask.index_select(0, z_idx) ^ bit;
  auto col_i = z_idx;
  auto val_i = torch::full({z_idx.numel()}, gamma, opts);

  auto cat_x = torch::cat({x_mask, x_i}, 0);
  auto cat_z = torch::cat({z_mask, z_i}, 0);
  auto cat_col = torch::cat({idx_all, col_i}, 0);
  auto cat_val = torch::cat({val_base, val_i}, 0);

  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  if (do_prune) {
    if (!coeffs_cache.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto keep = (cat_val * coeffs_prev.index_select(0, cat_col)).abs() >= min_abs.value();
    auto keep_idx = torch::nonzero(keep).flatten().to(torch::kInt64);
    cat_x = cat_x.index_select(0, keep_idx);
    cat_z = cat_z.index_select(0, keep_idx);
    cat_col = cat_col.index_select(0, keep_idx);
    cat_val = cat_val.index_select(0, keep_idx);
  }

  auto keys = torch::stack({cat_x, cat_z}, 1);
  py::module torch_mod = py::module::import("torch");
  py::tuple uniq_out = torch_mod.attr("unique")(keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                 py::arg("sorted") = false)
                           .cast<py::tuple>();
  auto uniq = uniq_out[0].cast<torch::Tensor>();
  auto inv = uniq_out[1].cast<torch::Tensor>().to(torch::kInt64);

  auto new_x = uniq.select(1, 0);
  auto new_z = uniq.select(1, 1);
  auto row = inv;
  auto col = cat_col;
  auto val = cat_val;

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto keep_rows = mask.index_select(0, row);
    auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
    row = row.index_select(0, keep_idx);
    col = col.index_select(0, keep_idx);
    val = val.index_select(0, keep_idx);

    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    row = row_map.index_select(0, row);

    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);
  }

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.numel()};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    auto contrib = val * coeffs_prev.index_select(0, col);
    out.index_add_(0, row, contrib);
    coeffs_cache_out = py::cast(out);
  }

  return {new_x, new_z, row, col, val, coeffs_cache_out};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, py::object>
build_amplitude_damping_step_mw_cpp(int64_t qubit, double gamma, const torch::Tensor& x_mask,
                                    const torch::Tensor& z_mask, const py::object& coeff_dtype_obj,
                                    c10::optional<double> min_abs, c10::optional<torch::Tensor> coeffs_cache,
                                    int64_t max_weight, double weight_x, double weight_y, double weight_z) {
  require_2d_i64(x_mask, "x_mask");
  require_2d_i64(z_mask, "z_mask");
  if (x_mask.sizes() != z_mask.sizes()) {
    throw std::runtime_error("x_mask/z_mask shape mismatch");
  }
  if (gamma < 0.0 || gamma > 1.0) {
    throw std::runtime_error("gamma must be in [0, 1]");
  }

  auto [w, b] = word_bit(qubit);
  if (w >= x_mask.size(1)) {
    throw std::runtime_error("amplitude damping qubit index exceeds mask word dimension");
  }

  auto device = x_mask.device();
  auto coeff_dtype = coerce_scalar_type(coeff_dtype_obj);
  auto n = x_mask.size(0);

  auto bit = torch::scalar_tensor(int64_t(1) << b, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto xw = x_mask.select(1, w);
  auto zw = z_mask.select(1, w);
  auto x_bit = (xw & bit).ne(0);
  auto z_bit = (zw & bit).ne(0);
  auto x_only = x_bit & (~z_bit);
  auto y_only = x_bit & z_bit;
  auto z_only = (~x_bit) & z_bit;

  auto opts = torch::TensorOptions().dtype(coeff_dtype).device(device);
  auto sq = std::sqrt(std::max(0.0, 1.0 - gamma));
  auto s_xy = torch::scalar_tensor(sq, opts);
  auto s_z = torch::scalar_tensor(1.0 - gamma, opts);

  auto idx_all = torch::arange(n, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto val_base = torch::ones({n}, opts);
  val_base = torch::where(x_only | y_only, s_xy, val_base);
  val_base = torch::where(z_only, s_z, val_base);

  auto z_idx = torch::nonzero(z_only).flatten().to(torch::kInt64);
  auto x_i = x_mask.index_select(0, z_idx);
  auto z_i = z_mask.index_select(0, z_idx).clone();
  auto z_iw = z_i.select(1, w);
  z_iw = torch::bitwise_and(z_iw, torch::bitwise_not(bit));
  z_i.select(1, w).copy_(z_iw);

  auto col_i = z_idx;
  auto val_i = torch::full({z_idx.numel()}, gamma, opts);

  auto cat_x = torch::cat({x_mask, x_i}, 0);
  auto cat_z = torch::cat({z_mask, z_i}, 0);
  auto cat_col = torch::cat({idx_all, col_i}, 0);
  auto cat_val = torch::cat({val_base, val_i}, 0);

  torch::Tensor coeffs_prev;
  bool do_prune = min_abs.has_value();
  if (do_prune) {
    if (!coeffs_cache.has_value()) {
      throw std::runtime_error("min_abs provided but coeffs_cache is None");
    }
    coeffs_prev = coeffs_cache.value();
    auto keep = (cat_val * coeffs_prev.index_select(0, cat_col)).abs() >= min_abs.value();
    auto keep_idx = torch::nonzero(keep).flatten().to(torch::kInt64);
    cat_x = cat_x.index_select(0, keep_idx);
    cat_z = cat_z.index_select(0, keep_idx);
    cat_col = cat_col.index_select(0, keep_idx);
    cat_val = cat_val.index_select(0, keep_idx);
  }

  auto keys = torch::cat({cat_x, cat_z}, 1);
  py::module torch_mod = py::module::import("torch");
  py::tuple uniq_out = torch_mod.attr("unique")(keys, py::arg("dim") = 0, py::arg("return_inverse") = true,
                                                 py::arg("sorted") = false)
                           .cast<py::tuple>();
  auto uniq = uniq_out[0].cast<torch::Tensor>();
  auto inv = uniq_out[1].cast<torch::Tensor>().to(torch::kInt64);

  auto n_words = x_mask.size(1);
  auto new_x = uniq.slice(1, 0, n_words);
  auto new_z = uniq.slice(1, n_words, 2 * n_words);
  auto row = inv;
  auto col = cat_col;
  auto val = cat_val;

  auto mask = truncate_terms_mask(new_x, new_z, max_weight, weight_x, weight_y, weight_z);
  if (!(mask.all().item<bool>())) {
    auto keep_rows = mask.index_select(0, row);
    auto keep_idx = torch::nonzero(keep_rows).flatten().to(torch::kInt64);
    row = row.index_select(0, keep_idx);
    col = col.index_select(0, keep_idx);
    val = val.index_select(0, keep_idx);

    auto row_map = torch::cumsum(mask.to(torch::kInt64), 0) - 1;
    row = row_map.index_select(0, row);

    auto out_idx = torch::nonzero(mask).flatten().to(torch::kInt64);
    new_x = new_x.index_select(0, out_idx);
    new_z = new_z.index_select(0, out_idx);
  }

  py::object coeffs_cache_out = py::none();
  if (do_prune) {
    std::vector<int64_t> out_sz{new_x.size(0)};
    auto out = torch::zeros(out_sz,
                            torch::TensorOptions().dtype(coeffs_prev.dtype()).device(coeffs_prev.device()));
    auto contrib = val * coeffs_prev.index_select(0, col);
    out.index_add_(0, row, contrib);
    coeffs_cache_out = py::cast(out);
  }

  return {new_x, new_z, row, col, val, coeffs_cache_out};
}

}  // namespace

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _pps_tensor_backend
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_clifford_step_cpp", &build_clifford_step_cpp,
        py::arg("symbol"), py::arg("qubits"), py::arg("x_mask"), py::arg("z_mask"), py::arg("coeff_dtype"),
      py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(), py::arg("max_weight"),
      py::arg("weight_x"), py::arg("weight_y"), py::arg("weight_z"));

  m.def("build_clifford_step_mw_cpp", &build_clifford_step_mw_cpp,
        py::arg("symbol"), py::arg("qubits"), py::arg("x_mask"), py::arg("z_mask"), py::arg("coeff_dtype"),
      py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(), py::arg("max_weight"),
      py::arg("weight_x"), py::arg("weight_y"), py::arg("weight_z"));

  m.def("build_pauli_rotation_step_cpp", &build_pauli_rotation_step_cpp,
        py::arg("gx"), py::arg("gz"), py::arg("param_idx"), py::arg("x_mask"), py::arg("z_mask"),
        py::arg("coeff_dtype"), py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(),
      py::arg("thetas_t") = py::none(), py::arg("max_weight"), py::arg("weight_x"), py::arg("weight_y"),
      py::arg("weight_z"));

  m.def("build_pauli_rotation_step_mw_cpp", &build_pauli_rotation_step_mw_cpp,
        py::arg("gx_words"), py::arg("gz_words"), py::arg("param_idx"), py::arg("x_mask"), py::arg("z_mask"),
        py::arg("coeff_dtype"), py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(),
      py::arg("thetas_t") = py::none(), py::arg("max_weight"), py::arg("weight_x"), py::arg("weight_y"),
      py::arg("weight_z"));

  m.def("build_pauli_rotation_step_implicit_cpp", &build_pauli_rotation_step_implicit_cpp,
        py::arg("gx"), py::arg("gz"), py::arg("param_idx"), py::arg("x_mask"), py::arg("z_mask"),
        py::arg("coeff_dtype"), py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(),
      py::arg("thetas_t") = py::none(), py::arg("max_weight"), py::arg("weight_x"), py::arg("weight_y"),
      py::arg("weight_z"));

  m.def("build_pauli_rotation_anti_sin_cpp", &build_pauli_rotation_anti_sin_cpp,
        py::arg("gx"), py::arg("gz"), py::arg("coeff_dtype"), py::arg("x_mask"), py::arg("z_mask"));

  m.def("merge_pauli_query_into_base_cpp", &merge_rotation_same_and_sin_1d,
        py::arg("base_x"), py::arg("base_z"), py::arg("query_x"), py::arg("query_z"));

  m.def("build_pauli_rotation_step_implicit_mw_cpp", &build_pauli_rotation_step_implicit_mw_cpp,
        py::arg("gx_words"), py::arg("gz_words"), py::arg("param_idx"), py::arg("x_mask"), py::arg("z_mask"),
        py::arg("coeff_dtype"), py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(),
      py::arg("thetas_t") = py::none(), py::arg("max_weight"), py::arg("weight_x"), py::arg("weight_y"),
      py::arg("weight_z"));

  m.def("build_pauli_rotation_anti_sin_mw_cpp", &build_pauli_rotation_anti_sin_mw_cpp,
        py::arg("gx_words"), py::arg("gz_words"), py::arg("coeff_dtype"), py::arg("x_mask"), py::arg("z_mask"));

  m.def("merge_pauli_query_into_base_mw_cpp", &merge_rotation_same_and_sin_mw,
        py::arg("base_x"), py::arg("base_z"), py::arg("query_x"), py::arg("query_z"));

    m.def("build_depolarizing_step_cpp", &build_depolarizing_step_cpp,
      py::arg("qubit"), py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("x_mask"), py::arg("z_mask"),
      py::arg("coeff_dtype"), py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(),
      py::arg("max_weight"), py::arg("weight_x"), py::arg("weight_y"), py::arg("weight_z"));

    m.def("build_depolarizing_step_mw_cpp", &build_depolarizing_step_mw_cpp,
      py::arg("qubit"), py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("x_mask"), py::arg("z_mask"),
      py::arg("coeff_dtype"), py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(),
      py::arg("max_weight"), py::arg("weight_x"), py::arg("weight_y"), py::arg("weight_z"));

    m.def("build_amplitude_damping_step_cpp", &build_amplitude_damping_step_cpp,
      py::arg("qubit"), py::arg("gamma"), py::arg("x_mask"), py::arg("z_mask"), py::arg("coeff_dtype"),
      py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(), py::arg("max_weight"),
      py::arg("weight_x"), py::arg("weight_y"), py::arg("weight_z"));

    m.def("build_amplitude_damping_step_mw_cpp", &build_amplitude_damping_step_mw_cpp,
      py::arg("qubit"), py::arg("gamma"), py::arg("x_mask"), py::arg("z_mask"), py::arg("coeff_dtype"),
      py::arg("min_abs") = py::none(), py::arg("coeffs_cache") = py::none(), py::arg("max_weight"),
      py::arg("weight_x"), py::arg("weight_y"), py::arg("weight_z"));
}
