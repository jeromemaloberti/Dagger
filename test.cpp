#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_vector.hpp"
#include "functional.hpp"
#include "operators.hpp"
#include <numeric>

TEST_CASE("to_string", "[tensor]") {
  std::vector<int32_t> data(48);
  std::iota(data.begin(), data.end(), 1);
  Tensor t{safetensors::kINT32, {48}, data};
  std::string ans = t.to_string<int32_t>(4);
  REQUIRE(ans == "[1,2,3,4, ..., 45,46,47,48]\n");
  Tensor t2{safetensors::kINT32, {4, 12}, data};
  ans = t2.to_string<int32_t>(4);
  REQUIRE(ans == "[\n [1,2,3,4, ..., 9,10,11,12]\n [13,14,15,16, ..., "
                 "21,22,23,24]\n [25,26,27,28, ..., 33,34,35,36]\n "
                 "[37,38,39,40, ..., 45,46,47,48]\n]\n");
  Tensor t3{safetensors::kINT32, {12, 4}, data};
  ans = t3.to_string<int32_t>(4);
  REQUIRE(ans ==
          "[\n [1,2,3,4]\n [5,6,7,8]\n [9,10,11,12]\n [13,14,15,16]\n ... \n "
          "[33,34,35,36]\n [37,38,39,40]\n [41,42,43,44]\n [45,46,47,48]\n]\n");
}

TEST_CASE("insert", "[tensor]") {
  std::vector<uint32_t> data(48), insert(8);
  std::iota(insert.begin(), insert.end(), 1);
  Tensor t{safetensors::kUINT32, {48}, data};
  Tensor i{safetensors::kUINT32, {8}, insert};
  t.insert(i, {8});
  REQUIRE_THAT(t.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{
                   0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  t.reshape({4, 12});
  i.reshape({2, 4});
  t.insert(i, {2, 0});
  REQUIRE_THAT(t.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{
                   0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                   0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0,
                   0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0}));
  Tensor t2{safetensors::kUINT32, {2, 4, 6}, data};
  i.reshape({1, 2, 4});
  t2.insert(i, {1, 2, 2});
  REQUIRE_THAT(t2.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8}));
  Tensor t3{safetensors::kUINT32, {2, 4, 2, 3}, data};
  i.reshape({1, 2, 2, 2});
  t3.insert(i, {1, 2, 0, 1});
  REQUIRE_THAT(t3.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 7, 8}));
}

TEST_CASE("subtile", "[tensor]") {
  std::vector<float> data(48);
  std::iota(data.begin(), data.end(), 1);
  Tensor t{safetensors::kFLOAT32, {48}, data};
  Tensor ans = t.subtile({8}, {0});
  REQUIRE(ans.shape().size() == 1);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}));
  ans = t.subtile({8}, {10});
  REQUIRE(ans.shape().size() == 1);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f}));
  Tensor t2{safetensors::kFLOAT32, {4, 12}, data};
  ans = t2.subtile({4, 4}, {0, 0});
  REQUIRE(ans.shape().size() == 2);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   1.0f, 2.0f, 3.0f, 4.0f, 13.0f, 14.0f, 15.0f, 16.0f, 25.0f,
                   26.0f, 27.0f, 28.0f, 37.0f, 38.0f, 39.0f, 40.0f}));
  ans = t2.subtile({2, 4}, {1, 4});
  REQUIRE(ans.shape().size() == 2);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   17.0f, 18.0f, 19.0f, 20.0f, 29.0f, 30.0f, 31.0f, 32.0f}));
  Tensor t3{safetensors::kFLOAT32, {2, 4, 6}, data};
  ans = t3.subtile({1, 1, 6}, {0, 0, 0});
  REQUIRE(ans.shape().size() == 3);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  ans = t3.subtile({1, 1, 6}, {1, 3, 0});
  REQUIRE(ans.shape().size() == 3);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f}));
  ans = t3.subtile({1, 2, 3}, {1, 1, 2});
  REQUIRE(ans.shape().size() == 3);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   33.0f, 34.0f, 35.0f, 39.0f, 40.0f, 41.0f}));
  Tensor t4{safetensors::kFLOAT32, {2, 4, 2, 3}, data};
  ans = t4.subtile({1, 1, 2, 3}, {0, 1, 0, 0});
  REQUIRE(ans.shape().size() == 4);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
}

TEST_CASE("split", "[tensor]") {
  std::vector<float> data(48);
  std::iota(data.begin(), data.end(), 1);
  Tensor t{safetensors::kFLOAT32, {2, 4, 6}, data};
  std::vector<Tensor> s = t.split(2, 2);
  REQUIRE(s.size() == 3);
  REQUIRE_THAT(s[0].vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   1.0f, 2.0f, 7.0f, 8.0f, 13.0f, 14.0f, 19.0f, 20.0f, 25.0f,
                   26.0f, 31.0f, 32.0f, 37.0f, 38.0f, 43.0f, 44.0f}));
  REQUIRE_THAT(s[1].vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   3.0f, 4.0f, 9.0f, 10.0f, 15.0f, 16.0f, 21.0f, 22.0f, 27.0f,
                   28.0f, 33.0f, 34.0f, 39.0f, 40.0f, 45.0f, 46.0f}));
  REQUIRE_THAT(s[2].vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   5.0f, 6.0f, 11.0f, 12.0f, 17.0f, 18.0f, 23.0f, 24.0f, 29.0f,
                   30.0f, 35.0f, 36.0f, 41.0f, 42.0f, 47.0f, 48.0f}));
  s = t.split(2, 1);
  REQUIRE(s.size() == 2);
  REQUIRE_THAT(s[0].vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                   9.0f,  10.0f, 11.0f, 12.0f, 25.0f, 26.0f, 27.0f, 28.0f,
                   29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f}));
  REQUIRE_THAT(s[1].vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                   21.0f, 22.0f, 23.0f, 24.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                   41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f}));
  s = t.split(1, 0);
  REQUIRE(s.size() == 2);
  REQUIRE_THAT(
      s[0].vdata<float>(),
      Catch::Matchers::Equals(std::vector<float>{
          1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
          10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
          19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f

      }));
  REQUIRE_THAT(s[1].vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f,
                   33.0f, 34.0f, 35.0f, 36.0f, 37.0f, 38.0f, 39.0f, 40.0f,
                   41.0f, 42.0f, 43.0f, 44.0f, 45.0f, 46.0f, 47.0f, 48.0f}));
}

TEST_CASE("align shape", "[operators]") {
  std::vector<size_t> s1{3, 2};
  auto ans = align_shape(s1);
  REQUIRE(ans.size() == 4);
  REQUIRE_THAT(ans, Catch::Matchers::Equals(std::vector<size_t>{1, 1, 3, 2}));
}

TEST_CASE("matmul", "[operators]") {
  Tensor t{safetensors::kFLOAT32,
           {2, 3},
           std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor ans = matmul<float, float, float>(t, t, true);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{14, 32, 32, 77}));

  Tensor t2{safetensors::kFLOAT32,
            {3, 2},
            std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor ans2 = matmul<float, float, float>(t, t2, false);
  REQUIRE_THAT(ans2.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{22, 28, 49, 64}));
  Tensor bias{safetensors::kFLOAT32, {2}, std::vector<float>{1.0, 2.0}};
  Tensor ans3 = matmul<float, float, float>(t, t, bias, true);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{15, 34, 33, 79}));
  Tensor ans4 = matmul<float, float, float>(t, t2, bias, false);
  REQUIRE_THAT(ans4.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{23, 30, 50, 66}));
}

TEST_CASE("gemv", "[operators]") {
  std::vector<float> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Tensor t{safetensors::kFLOAT32, {1, 6}, data};
  Tensor w{safetensors::kFLOAT32, {6, 1}, data};
  Tensor ans = matmul<float, float, float>(t, w, false);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{91.0f}));
  ans = gemv<float, float, float>(t, w, nullptr);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{91.0f}));
  // Tensor t2{safetensors::kFLOAT32,
  //           {3, 2},
  //           std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  // Tensor ans2 = matmul<float, float, float>(t, t2, false);
  // REQUIRE_THAT(ans2.vdata<float>(),
  //              Catch::Matchers::Equals(std::vector<float>{22, 28, 49, 64}));
  // Tensor bias{safetensors::kFLOAT32, {2}, std::vector<float>{1.0, 2.0}};
  // Tensor ans3 = matmul<float, float, float>(t, t, bias, true);
  // REQUIRE_THAT(ans3.vdata<float>(),
  //              Catch::Matchers::Equals(std::vector<float>{15, 34, 33, 79}));
  // Tensor ans4 = matmul<float, float, float>(t, t2, bias, false);
  // REQUIRE_THAT(ans4.vdata<float>(),
  //              Catch::Matchers::Equals(std::vector<float>{23, 30, 50, 66}));
}

TEST_CASE("batch_matmul", "[functional]") {
  std::vector<float> data(30);
  std::iota(data.begin(), data.end(), 1);
  Tensor t{safetensors::kFLOAT32, {2, 3, 5}, data};
  Tensor ans = batch_matmul(t, t, 1, true);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   55.0f, 130.0f, 205.0f, 130.0f, 330.0f, 530.0f, 205.0f,
                   530.0f, 855.0f, 1630.0f, 2080.0f, 2530.0f, 2080.0f, 2655.0f,
                   3230.0f, 2530.0f, 3230.0f, 3930.0f}));
  Tensor t2 = transpose<float>(t, 1, 2);
  Tensor ans2 = batch_matmul(t, t2, 1, false);
  REQUIRE_THAT(ans2.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{
                   55.0f, 130.0f, 205.0f, 130.0f, 330.0f, 530.0f, 205.0f,
                   530.0f, 855.0f, 1630.0f, 2080.0f, 2530.0f, 2080.0f, 2655.0f,
                   3230.0f, 2530.0f, 3230.0f, 3930.0f}));
  std::vector<float> data2(60);
  std::iota(data2.begin(), data2.end(), 1);
  Tensor t3{safetensors::kFLOAT32, {2, 2, 3, 5}, data2};
  Tensor ans3 = batch_matmul(t3, t3, 2, true);
  REQUIRE_THAT(
      ans3.vdata<float>(),
      Catch::Matchers::Equals(std::vector<float>{
          55.0f,    130.0f,   205.0f,   130.0f,   330.0f,   530.0f,
          205.0f,   530.0f,   855.0f,   1630.0f,  2080.0f,  2530.0f,
          2080.0f,  2655.0f,  3230.0f,  2530.0f,  3230.0f,  3930.0f,
          5455.0f,  6280.0f,  7105.0f,  6280.0f,  7230.0f,  8180.0f,
          7105.0f,  8180.0f,  9255.0f,  11530.0f, 12730.0f, 13930.0f,
          12730.0f, 14055.0f, 15380.0f, 13930.0f, 15380.0f, 16830.0f}));
  Tensor t4 = transpose<float>(t3, 2, 3);
  Tensor ans4 = batch_matmul(t3, t4, 2, false);
  REQUIRE_THAT(
      ans4.vdata<float>(),
      Catch::Matchers::Equals(std::vector<float>{
          55.0f,    130.0f,   205.0f,   130.0f,   330.0f,   530.0f,
          205.0f,   530.0f,   855.0f,   1630.0f,  2080.0f,  2530.0f,
          2080.0f,  2655.0f,  3230.0f,  2530.0f,  3230.0f,  3930.0f,
          5455.0f,  6280.0f,  7105.0f,  6280.0f,  7230.0f,  8180.0f,
          7105.0f,  8180.0f,  9255.0f,  11530.0f, 12730.0f, 13930.0f,
          12730.0f, 14055.0f, 15380.0f, 13930.0f, 15380.0f, 16830.0f}));
}

TEST_CASE("transpose", "[operators]") {
  Tensor t{safetensors::kFLOAT32,
           {2, 3},
           std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor ans = transpose<float>(t, 0, 1);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{1, 4, 2, 5, 3, 6}));
}

TEST_CASE("cumsum_rvv", "[operators]") {
  Tensor t{safetensors::kFLOAT32,
           {2, 3},
           std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor ans = cumsum_rvv<float, float>(t);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{1, 3, 6, 4, 9, 15}));
  Tensor t2{safetensors::kFLOAT32, {3}, std::vector<float>{1.0, 2.0, 3.0}};
  Tensor ans2 = cumsum_rvv<float, float>(t2);
  REQUIRE_THAT(ans2.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{1, 3, 6}));
}

TEST_CASE("sample_rvv", "[operators]") {
  Tensor t{safetensors::kFLOAT32,
           {2, 3},
           std::vector<float>{0.0, 0.7, 1.0, 0.5, 0.5, 1.0}};
  Tensor ans = sample_rvv<float, uint32_t>(t, 2, 1337);
  REQUIRE_THAT(ans.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{1, 1, 0, 2}));
  Tensor t2{safetensors::kFLOAT32,
            {2, 3},
            std::vector<float>{0.0, 0.0, 1.0, 1.0, 1.0, 1.0}};
  Tensor ans2 = sample_rvv<float, uint32_t>(t2, 2, 1337);
  REQUIRE_THAT(ans2.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{2, 2, 0, 0}));
  Tensor t3{safetensors::kFLOAT32, {3}, std::vector<float>{0.3, 0.6, 1.0}};
  Tensor ans3 = cumsum_rvv<float, uint32_t>(t3);
  REQUIRE_THAT(ans3.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{0, 0, 1}));
}

TEST_CASE("unary", "[operators]") {
  Tensor t{safetensors::kFLOAT32,
           {2, 3},
           std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  elementwise_unary<float>(t, [](const float &v) { return -v; }, "Neg");
  REQUIRE_THAT(t.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                     -1.0, -2.0, -3.0, -4.0, -5.0, -6.0}));

  Tensor res = elementwise_unary<float, float>(
      t, [](const float &v) { return v + 1.0; }, "dummy");
  REQUIRE_THAT(t.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                     -1.0, -2.0, -3.0, -4.0, -5.0, -6.0}));
  REQUIRE_THAT(res.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       0.0, -1.0, -2.0, -3.0, -4.0, -5.0}));
  TensorIndex idx{{2, 3}}; // 3 rows 2 cols
  idx.print_tensor(res.vdata<float>(), std::cerr);
}

TEST_CASE("binary", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 3}, // 2 rows 3 cols
            std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor ans = elementwise_binary<float, float>(
      t1, t1, [&](const float &v1, const float &v2) { return v1 + v2; }, "Add");
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       2.0, 4.0, 6.0, 8.0, 10.0, 12.0}));
  Tensor t2{safetensors::kFLOAT32, {1, 3}, std::vector<float>{1.0, 2.0, 3.0}};
  Tensor ans2 = elementwise_binary<float, float>(
      t1, t2, [](const float &v1, const float &v2) { return v1 + v2; }, "Add");
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        2.0, 4.0, 6.0, 5.0, 7.0, 9.0}));
  Tensor t3{safetensors::kFLOAT32, {2, 1}, std::vector<float>{1.0, 2.0}};
  Tensor ans3 = elementwise_binary<float, float>(
      t1, t3, [](const float &v1, const float &v2) { return v1 + v2; }, "Add");
  REQUIRE_THAT(ans3.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        2.0, 3.0, 4.0, 6.0, 7.0, 8.0}));
  Tensor ans4 = elementwise_binary<float>(
      t1, t1, [&](const float &v1, const float &v2) { return v1 + v2; }, "Add");
  REQUIRE_THAT(ans4.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        2.0, 4.0, 6.0, 8.0, 10.0, 12.0}));
  REQUIRE_THAT(t1.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                      2.0, 4.0, 6.0, 8.0, 10.0, 12.0}));
}

TEST_CASE("gather", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {4, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor idx{safetensors::kUINT32, {2}, std::vector<uint32_t>{0, 3}};
  Tensor ans = gather<float, uint32_t>(t1, idx);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       1, 2, 3, 10, 11, 12}));
  Tensor t2{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor idx2{safetensors::kUINT32, {1}, std::vector<uint32_t>{1}};
  Tensor ans2 = gather<float, uint32_t>(t2, idx2);
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        7, 8, 9, 10, 11, 12}));
  Tensor idx3{safetensors::kUINT32, {1, 2}, std::vector<uint32_t>{1, 1}};
  Tensor ans3 = gather<float, uint32_t>(t2, idx3);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{10, 11, 12}));
}

TEST_CASE("embeddings_gather", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {4, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor idx{safetensors::kUINT32, {2}, std::vector<uint32_t>{0, 3}};
  Tensor ans = embeddings_gather<float, uint32_t>(t1, idx);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       1, 2, 3, 10, 11, 12}));
  Tensor idx2{safetensors::kUINT32, {2, 1}, std::vector<uint32_t>{0, 3}};
  Tensor ans2 = embeddings_gather<float, uint32_t>(t1, idx2);
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        1, 2, 3, 10, 11, 12}));
  Tensor idx3{safetensors::kUINT32, {2, 2}, std::vector<uint32_t>{3, 2, 1, 0}};
  Tensor ans3 = embeddings_gather<float, uint32_t>(t1, idx3);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Equals(
                   std::vector<float>{10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3}));
}

TEST_CASE("sum", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = reduce_sum<float, float, float>(t1, 0);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       8.0, 10.0, 12.0, 14.0, 16.0, 18.0}));
  Tensor ans2 = reduce_sum<float, float, float>(t1, 1);
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        5.0, 7.0, 9.0, 17.0, 19.0, 21.0}));
  Tensor ans3 = reduce_sum<float, float, float>(t1, 2);
  REQUIRE_THAT(ans3.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        6.0, 15.0, 24.0, 33.0}));
}

TEST_CASE("mean", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = reduce_mean<float, float, float>(t1, 0);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       4.0, 5.0, 6.0, 7.0, 8.0, 9.0}));
  Tensor ans2 = reduce_mean<float, float, float>(t1, 1);
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        2.5, 3.5, 4.5, 8.5, 9.5, 10.5}));
  Tensor ans3 = reduce_mean<float, float, float>(t1, 2);
  REQUIRE_THAT(ans3.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        2.0, 5.0, 8.0, 11.0}));
  Tensor t2{safetensors::kFLOAT32,
            {1, 1, 3, 2},
            std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor ans4 = reduce_mean<float, float, float>(t2, 2);
  REQUIRE_THAT(ans4.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{3.0, 4.0}));
}

TEST_CASE("var", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = reduce_var<float, float, float>(t1, 0);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       18.0, 18.0, 18.0, 18.0, 18.0, 18.0}));
  Tensor ans2 = reduce_var<float, float, float>(t1, 1);
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        4.5, 4.5, 4.5, 4.5, 4.5, 4.5}));
  Tensor ans3 = reduce_var<float, float, float>(t1, 2);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Approx(std::vector<float>{1, 1, 1, 1}));
}

TEST_CASE("prod", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = reduce_mul<float, float, float>(t1, 0);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       7.0, 16.0, 27.0, 40.0, 55.0, 72.0}));
  Tensor ans2 = reduce_mul<float, float, float>(t1, 1);
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        4, 10, 18, 70, 88, 108}));
  Tensor ans3 = reduce_mul<float, float, float>(t1, 2);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{6, 120, 504, 1320}));
}

TEST_CASE("max", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = reduce_max<float, float, float>(t1, 0);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       7.0, 8.0, 9.0, 10.0, 11.0, 12.0}));
  Tensor ans2 = reduce_max<float, float, float>(t1, 1);
  REQUIRE_THAT(ans2.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                        4, 5, 6, 10, 11, 12}));
  Tensor ans3 = reduce_max<float, float, float>(t1, 2);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{3, 6, 9, 12}));
}

TEST_CASE("min", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = reduce_min<float, float, float>(t1, 0);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Equals(std::vector<float>{
                                       1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
  Tensor ans2 = reduce_min<float, float, float>(t1, 1);
  REQUIRE_THAT(ans2.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{1, 2, 3, 7, 8, 9}));
  Tensor ans3 = reduce_min<float, float, float>(t1, 2);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Equals(std::vector<float>{1, 4, 7, 10}));
}

TEST_CASE("argmax", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = argmax<float, uint32_t>(t1, 0);
  REQUIRE_THAT(
      ans.vdata<uint32_t>(),
      Catch::Matchers::Equals(std::vector<uint32_t>{1, 1, 1, 1, 1, 1}));
  Tensor ans2 = argmax<float, uint32_t>(t1, 1);
  REQUIRE_THAT(
      ans2.vdata<uint32_t>(),
      Catch::Matchers::Equals(std::vector<uint32_t>{1, 1, 1, 1, 1, 1}));
  Tensor ans3 = argmax<float, uint32_t>(t1, 2);
  REQUIRE_THAT(ans3.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{2, 2, 2, 2}));
}

TEST_CASE("argmin", "[operators]") {
  Tensor t1{safetensors::kFLOAT32,
            {2, 2, 3},
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  Tensor ans = argmin<float, uint32_t>(t1, 0);
  REQUIRE_THAT(
      ans.vdata<uint32_t>(),
      Catch::Matchers::Equals(std::vector<uint32_t>{0, 0, 0, 0, 0, 0}));
  Tensor ans2 = argmin<float, uint32_t>(t1, 1);
  REQUIRE_THAT(
      ans2.vdata<uint32_t>(),
      Catch::Matchers::Equals(std::vector<uint32_t>{0, 0, 0, 0, 0, 0}));
  Tensor ans3 = argmin<float, uint32_t>(t1, 2);
  REQUIRE_THAT(ans3.vdata<uint32_t>(),
               Catch::Matchers::Equals(std::vector<uint32_t>{0, 0, 0, 0}));
}

TEST_CASE("layer_norm", "[functional]") {
  Tensor t{safetensors::kFLOAT32,
           {2, 3},
           std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor t2{safetensors::kFLOAT32, {3}, std::vector<float>{1.0, 2.0, 3.0}};
  Tensor ans = layer_norm(t, t2, t2);
  REQUIRE_THAT(ans.vdata<float>(), Catch::Matchers::Approx(std::vector<float>{
                                       -0.224736, 2.000000, 6.674207, -0.224736,
                                       2.000000, 6.674207}));
}

TEST_CASE("softmax", "[functional]") {
  Tensor t{safetensors::kFLOAT32,
           {2, 3},
           std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  Tensor ans = softmax(t, 1);
  REQUIRE_THAT(ans.vdata<float>(),
               Catch::Matchers::Approx(std::vector<float>{
                   0.090030573f, 0.244728476f, 0.665240943f, 0.090030573f,
                   0.244728476f, 0.665240943f}));
  Tensor ans2 = softmax(t, 0);
  REQUIRE_THAT(ans2.vdata<float>(),
               Catch::Matchers::Approx(std::vector<float>{
                   0.047425874f, 0.047425874f, 0.047425874f, 0.952574134f,
                   0.952574134f, 0.952574134f}));
}

TEST_CASE("scaled_dot_product_attention", "[functional]") {
  size_t B = 1, T = 3, C = 6;
  std::vector<float> data(B * T * C * 3);
  std::iota(data.begin(), data.end(), 1);
  Tensor t{safetensors::kFLOAT32, {B, T, 3 * C}, data};
  Tensor ans = scaled_dot_product_attention(t, C, 2, nullptr, nullptr, nullptr);
  REQUIRE_THAT(
      ans.vdata<float>(),
      Catch::Matchers::Approx(std::vector<float>{
          49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f, 49.0f, 50.0f, 51.0f, 52.0f,
          53.0f, 54.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f}));
  Tensor mask{safetensors::kFLOAT32, {T, T}};
  float *out = mask.data<float>();
  for (size_t i = 0; i < T; ++i) {
    for (size_t j = 0; j < T; ++j) {
      out[i * T + j] = i < j ? -std::numeric_limits<float>::infinity() : 0.0;
    }
  }
  Tensor ans2 = scaled_dot_product_attention(t, C, 2, &mask, nullptr, nullptr);
  REQUIRE_THAT(
      ans2.vdata<float>(),
      Catch::Matchers::Approx(std::vector<float>{
          13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 31.0f, 32.0f, 33.0f, 34.0f,
          35.0f, 36.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f}));
  B = 1;
  T = 4;
  C = 6;
  std::vector<float> data2(B * T * C * 3);
  std::iota(data2.begin(), data2.end(), 1);
  Tensor t2{safetensors::kFLOAT32, {B, T, 3 * C}, data2};
  Tensor mask2{safetensors::kFLOAT32, {T, T}};
  float *out2 = mask2.data<float>();
  for (size_t i = 0; i < T; ++i) {
    for (size_t j = 0; j < T; ++j) {
      out2[i * T + j] = i < j ? -std::numeric_limits<float>::infinity() : 0.0;
    }
  }
  Tensor ans3 =
      scaled_dot_product_attention(t2, C, 2, &mask2, nullptr, nullptr);
  REQUIRE_THAT(ans3.vdata<float>(),
               Catch::Matchers::Approx(std::vector<float>{
                   13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 31.0f, 32.0f,
                   33.0f, 34.0f, 35.0f, 36.0f, 49.0f, 50.0f, 51.0f, 52.0f,
                   53.0f, 54.0f, 67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f}));
}

TEST_CASE("cached_scaled_dot_product_attention", "[functional]") {
  size_t B = 1, T = 3, C = 6;
  std::vector<float> data(B * T * C * 3);
  std::iota(data.begin(), data.end(), 1);
  Tensor t{safetensors::kFLOAT32, {B, T, 3 * C}, data};
  Tensor k_cache{safetensors::kFLOAT32, {B, 2, T + 1, 3}};
  Tensor v_cache{safetensors::kFLOAT32, {B, 2, T + 1, 3}};
  Tensor mask{safetensors::kFLOAT32, {T, T}};
  float *out = mask.data<float>();
  for (size_t i = 0; i < T; ++i) {
    for (size_t j = 0; j < T; ++j) {
      out[i * T + j] = i < j ? -std::numeric_limits<float>::infinity() : 0.0;
    }
  }
  Tensor ans =
      cached_scaled_dot_product_attention(t, C, 2, &mask, k_cache, v_cache, 0);
  REQUIRE_THAT(
      ans.vdata<float>(),
      Catch::Matchers::Approx(std::vector<float>{
          13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 31.0f, 32.0f, 33.0f, 34.0f,
          35.0f, 36.0f, 49.0f, 50.0f, 51.0f, 52.0f, 53.0f, 54.0f}));
  std::vector<float> data2(B * C * 3);
  std::iota(data2.begin(), data2.end(), data.size() + 1);
  Tensor t2{safetensors::kFLOAT32, {B, 1, 3 * C}, data2};
  Tensor mask2{safetensors::kFLOAT32, {T + 1, T + 1}};
  out = mask2.data<float>();
  for (size_t i = 0; i < T + 1; ++i) {
    for (size_t j = 0; j < T + 1; ++j) {
      out[i * T + j] = i < j ? -std::numeric_limits<float>::infinity() : 0.0;
    }
  }
  Tensor ans2 = cached_scaled_dot_product_attention(t2, C, 2, &mask2, k_cache,
                                                    v_cache, T);
  REQUIRE_THAT(ans2.vdata<float>(),
               Catch::Matchers::Approx(std::vector<float>{
                   67.0f, 68.0f, 69.0f, 70.0f, 71.0f, 72.0f}));
}
