#pragma once

#include <algorithm>
#include <array>
#include <ciso646>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

#ifdef IMGFUN_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace imgfun {

#ifdef _DEBUG
#define IMGFUN_ASSERT(what) assert(what)
#else
#define IMGFUN_ASSERT(what)
#endif

template <typename T>
static constexpr int div_round(T const a, T const b) {
  return static_cast<int>((a + b / 2) / b);
}

constexpr int roundp(float const f) { return static_cast<int>(f + .5f); }

using Pixel = uint8_t;

constexpr Pixel clamp(int const i) {
  return static_cast<Pixel>(std::clamp(i, 0, 255));
}

template <typename T>
constexpr int sign(T const i) {
  return i < 0 ? -1 : (i > 0 ? 1 : 0);
}

struct Coord {
  int x = 0;
  int y = 0;
};

#ifdef OPENCV_CORE_TYPES_HPP
using Size = cv::Size;
#else
struct Size {
  int width = 0;
  int height = 0;
  int area() const { return width * height; }
};
#endif

struct Offset {
  Coord position{};
  int scale = 1;
};

#define IMGFUN_FOREACH_ROW(buf, row)              \
  for (auto row = 0; row < size_.height; ++row) { \
    if constexpr (HasRowFun) {                    \
      row_fun(row);                               \
    }                                             \
    auto* buf = row_begin(row);
#define IMGFUN_FOREACH_COL(col) /* loop over all columns */ \
  for (auto col = 0; col < size_.width; ++col) {
#define IMGFUN_FOREACH(buf, row, col) \
  IMGFUN_FOREACH_ROW(buf, row)        \
  IMGFUN_FOREACH_COL(col)
#define IMGFUN_FOREACH_END \
  }                        \
  }

/// Deleter is used to cleanup the image data on the Image object deletion.
/// The default behavior is to delete the data as an array. An image loaded with
/// the stb_image library will be freed using stbi_image_free.
class Deleter {
 public:
  /// Creates the deleter.
  /// @param deleter should handle the cleanup of data. Will be called on
  /// destruction.
  /// @param data the data to be cleaned up.
  Deleter(std::function<void(Pixel*)>&& deleter, Pixel* data)
      : deleter_(deleter), data_(data) {}
  /// Simplified deleter that just deletes the data array.
  /// @param data pointer to the data to be deleted on destruction.
  Deleter(Pixel* data) : Deleter([](Pixel* data) { delete[] data; }, data) {}
  ~Deleter() { deleter_(data_); }

 protected:
  /// Standard deleter that does nothing but may be derived.
  Deleter() : Deleter([](Pixel*) {}, nullptr) {}

 private:
  std::function<void(Pixel*)> deleter_;
  Pixel* data_;
};

class View {
 public:
  View(Pixel const& current, int stride) : p_(&current), stride_(stride) {}
  View(Pixel const& current, int stride, Coord const& offset)
      : p_(&current + stride * offset.y + offset.x), stride_(stride) {}
  Pixel operator*() const { return *p_; }
  Pixel operator()(int const x_off, int const y_off) const {
    return p_[stride_ * y_off + x_off];
  }
  Pixel operator()(Coord const& off) const {
    return p_[stride_ * off.y + off.x];
  }

 private:
  Pixel const* p_;
  int stride_;
};

using Histogram = std::array<int, 256>;

/// Base class for Image data following the CRTP pattern.
/// The reason for using CRTP is that the `load`, `create` and `map` methods
/// will return the derived class that could contain your own image algorithms.
/// No additional upcasting required.
/// Img is a shared resource and copying is shallow. For a deep copy use the
/// `clone` method.
/// @tparam Img class deriving from ImgBase. The class must define the same
/// constructor as ImgBase - should be defined with `using ImgBase::ImgBase`.
///
/// usage:
/** @code{cpp}
// Example implementing an own filter
class MyImage : public imgfun::ImgBase<MyImage> // CRTP - Img is MyImage now
{
  // use the same constructors - leave them private
  using imgfun::ImgBase::ImgBase;
public:
  // example calling the map function
  // multiply each pixel with the given factor
  MyImage multiply(float const factor) const {
    return map(
        [factor](imgfun::Pixel const& p) { return imgfun::clamp(p * factor); });
  }
};

void main()
{
  auto img = MyImage::load("foo.png");
  auto filtered = img.my_filter();
  filtered.save("foo_filtered.png");
}
@endcode
*/
template <typename Img>
class ImgBase {
 protected:
  ImgBase(Pixel* data, Size size, int channels, Offset const offset,
          std::shared_ptr<Deleter> deleter, int const data_offset = 0,
          int const stride = 0)
      : data_(data),
        size_(size),
        channels_(channels),
        offset_(offset),
        data_offset_(data_offset),
        stride_(stride),
        deleter_(std::move(deleter)) {
    if (stride_ == 0) {
      stride_ = channels_ * size_.width;
    }
  }

  static void empty_row_fun(int){};

 public:
  static Img create(Pixel* img_data, Size const& size, int const channels = 1,
                    Offset const offset = {}) noexcept {
    return Img(img_data, size, channels, offset, {});
  }
  static Img create(Size const& size, int const channels = 1,
                    Offset const offset = {}) noexcept {
    auto const num_bytes = size.area() * channels;
    auto* data = new Pixel[num_bytes];
    return Img(data, size, channels, offset, std::make_shared<Deleter>(data));
  }

  static Img load(std::string const& filename);
  bool save(std::string const& filename) const;

  /// Creates a deep copy of the current image with contiguous memory layout.
  /// Consider using `contiguous` if you do not write on that image and just
  /// need a contiguous memory layout.
  Img clone() const;

  /// Ensure contiguous layout of the image. If the current image is not
  /// contiguous a clone is returned otherwise a share of this image is
  /// returned.
  /// Consider using clone if you need to write on that image and still hold
  /// other references to that image.
  Img contiguous() const;

  Img grayscale(int scale_down = 1) const;

  size_t num_bytes() const;
  Pixel* row_begin(int row);
  Pixel const* row_begin(int row) const;

  View view_at(Pixel const& p) const;
  View view_at(Coord const& pos) const;

  using RowFun = std::function<void(int)>;
  /// @{
  /// @brief reduces the pixels to one result.
  /// @tparam T the accumulator type. Is expected to be constructible and
  /// moveable.
  /// @param akku the accumulator
  /// @param fun function working on the accumulator with 3 different
  /// signatures:
  /// - void(T&, Pixel const&) just taking the pixels
  /// - void(T&, Pixel const&, Coord const&) taking the pixels and their current
  /// coordinate in the picture
  /// @return the accumulator
  template <typename T>
  T reduce(T&& akku, std::function<void(T&, Pixel const&)>&& fun) const;
  template <typename T>
  T reduce(T&& akku,
           std::function<void(T&, Pixel const&, Coord const&)>&& fun) const;
  /// @}
  /// @brief variant of reduce that calls row_fun for each row.
  /// @tparam T the accumulator type. Is expected to be constructible and
  /// moveable.
  /// @param akku the accumulator
  /// @param row_fun called for each row.
  /// @param fun function working on the accumulator with signatur:
  /// - void(T&, Pixel const&, int) - taking the pixels and their current column
  /// in the picture. The current row is already given in row_fun.
  /// @return the accumulator
  template <typename T>
  T reduce(T&& akku, RowFun&& row_fun,
           std::function<void(T&, Pixel const&, int)>&& fun) const;

  /// @brief mixes reduce with all - checking that all elements are satisfied by
  /// a check perform in `fun`.
  /// @tparam T the type of the reduced akkumulator.
  /// @param akku the accumulator that collects the result.
  /// @param row_fun the row function called for each row.
  /// @param fun the accumulator functiont that here returns true, if the
  /// performed check is satisfied or false if not.
  /// @return the optional accumalator which is none in case not all elements
  /// satisfied the function result.
  template <typename T>
  std::optional<T> reduce_all(
      T&& akku, RowFun&& row_fun,
      std::function<bool(T&, Pixel const&, int)>&& fun) const;

  void apply(std::function<Pixel(Pixel const&)>&& fun);
  void apply(std::function<Pixel(Pixel const&, Coord const&)>&& fun);
  void apply(RowFun&& row_fun, std::function<Pixel(Pixel const&, int)>&& fun);

  Img map(std::function<Pixel(Pixel const&)>&& fun) const;
  Img map(std::function<Pixel(Pixel const&, Coord const&)>&& fun) const;
  Img map(RowFun&& row_fun,
          std::function<Pixel(Pixel const&, int)>&& fun) const;

  std::optional<Coord> find(std::function<bool(Pixel const&)>&& fun) const;
  std::optional<Coord> find(RowFun&& row_fun,
                            std::function<bool(Pixel const&, int)>&& fun) const;
  std::optional<Coord> find(
      std::function<bool(Pixel const&, Coord const&)>&& fun) const;

  Pixel* data();
  Pixel const* data() const;
  Pixel& at(Coord const& pos);
  Pixel const& at(Coord const& pos) const;
  Img at(Coord const& upper_left, Size const& size) const;
  Img crop(int crop_width = 1) const;
  Img crop(int crop_width, int crop_height) const;

  Histogram histogram() const;

  std::tuple<Pixel, Pixel> darkest_brightest(int lower_cut = 0,
                                             int upper_cut = 0) const;
  Img auto_contrast(int cut_lower = 0, int cut_upper = 0) const;

  Img denoise() const;
  template <int Add = 0, int Fact = 1>
  Img laplace() const;
  template <int Outer = 1, int Inner = 2 * Outer, int Add = 0>
  Img sobel(bool vertical = true) const;
  Img fine_rotate(float angle_grad) const;

  template <                   // matrix
      int UL, int UC, int UR,  // first row
      int CL, int CC, int CR,  // center row
      int LL, int LC, int LR,  // lower row
      int Add = 0, int Div = UL + UC + UR + CL + CC + CR + LL + LC + LR,
      int Fact = 1>
  Img filter() const;

  template <int Diag, int Orth, int Center, int Add = 0,
            int Div = Diag * 4 + Orth * 4 + Center, int Fact = 1>
  Img filter_sym() const;
  Size size() const { return size_; }
  int channels() const { return channels_; }

 protected:
  Coord _rotate_offset(float const angle_rad) const;

  template <typename T, typename Res, bool HasRowFun, bool HasCoord,
            typename RFun, typename Fun>
  Res _reduce(T&& akku, RFun&& row_fun, Fun&& fun) const;

  template <bool HasRowFun, bool HasCoord, typename RFun, typename Fun>
  std::optional<Coord> _find(RFun&& row_fun, Fun&& fun) const;

  template <bool HasRowFun, bool HasCoord, typename RFun, typename Fun>
  void _apply(RFun&& row_fun, Fun&& fun, Img& out) const;

  template <bool HasRowFun, bool HasCoord, typename RFun, typename Fun>
  Img _map(RFun&& row_fun, Fun&& fun) const;

  Img _shallow_copy() const { return at({}, size_); }

  Img const* _self() const { return reinterpret_cast<Img const*>(this); }
  Img* _self() { return reinterpret_cast<Img*>(this); }
  Pixel* data_;
  Size size_;
  int channels_;
  Offset offset_;
  int data_offset_;
  int stride_;
  /// The deleter is simply shared. When deleter is destructed the associated
  /// callback will free the image data.
  std::shared_ptr<Deleter> deleter_;
};

inline Coord operator+(Coord const& a, Coord const& b) {
  return Coord{a.x + b.x, a.y + b.y};
}

template <typename Img>
Img ImgBase<Img>::load(std::string const& filename) {
  Size size{};
  auto channels = 0;
  auto* image_data =
      stbi_load(filename.c_str(), &size.width, &size.height, &channels, 0);
  return Img(image_data, size, channels, {},
             std::make_shared<Deleter>(
                 [](Pixel* data) { stbi_image_free(data); }, image_data));
}

template <typename Img>
bool ImgBase<Img>::save(std::string const& filename) const {
  auto const res = stbi_write_png(filename.c_str(), size_.width, size_.height,
                                  channels_, &data_[data_offset_], stride_);
  return res == 1;
}

template <typename Img>
Img ImgBase<Img>::clone() const {
  return self_()->map([](auto const& p) { return p; });
}

template <typename Img>
Img ImgBase<Img>::contiguous() const {
  if (stride_ == size_.width * channels_) {
    return self_()->_shallow_copy();
  }
  return self_()->clone();
}

template <typename Img>
Img ImgBase<Img>::grayscale(int const scale_down) const {
  auto const cols = size_.width / scale_down;
  auto const rows = size_.height / scale_down;
  auto const SF = scale_down * channels_;
  auto const SF2 = scale_down * SF;
  auto const row_offset = (size_.width - cols * scale_down) * channels_;
  auto* const out_begin = new Pixel[cols * rows];
  auto* out = out_begin;
  auto const* buf = &data_[0];

  for (int row = 0; row < rows; ++row) {
    auto block_sums = std::vector<int>(cols);
    // cache-friendly optimization taking the whole rows
    for (int y = 0; y < scale_down; ++y) {
      for (auto& sum : block_sums) {
        auto* end = &buf[SF];
        sum += std::accumulate(buf, end, 0);
        buf = end;
      }
      buf += row_offset;
    }
    auto const average = [=](auto const& sum) { return div_round(sum, SF2); };
    std::transform(block_sums.begin(), block_sums.end(), out, average);
    out += cols;
  }

  return Img(out_begin, Size{cols, rows}, 1, Offset{{}, scale_down},
             std::make_shared<Deleter>(out_begin));
}

template <typename Img>
size_t ImgBase<Img>::num_bytes() const {
  return size_.area() * channels_;
}

template <typename Img>
Pixel* ImgBase<Img>::row_begin(int const row) {
  return &data_[data_offset_ + row * stride_];
}
template <typename Img>
Pixel const* ImgBase<Img>::row_begin(int const row) const {
  return &data_[data_offset_ + row * stride_];
}

template <typename Img>
View ImgBase<Img>::view_at(Pixel const& p) const {
  return View(p, stride_);
}

template <typename Img>
View ImgBase<Img>::view_at(Coord const& pos) const {
  return view_at(data_[data_offset_ + pos.x + pos.y * stride_]);
}

template <typename Img>
template <typename T, typename Res, bool HasRowFun, bool HasCoord,
          typename RFun, typename Fun>
Res ImgBase<Img>::_reduce(T&& init_akku, RFun&& row_fun, Fun&& fun) const {
  static auto constexpr bool_result =
      std::is_same<std::invoke_result_t<Fun, T&, Pixel const&, int>,
                   bool>::value;

  T akku = std::move(init_akku);
  IMGFUN_FOREACH(buf, row, col)
  if constexpr (HasRowFun) {
    if constexpr (bool_result) {
      if (not fun(akku, buf[col], col)) {
        return {};
      }
    } else {
      fun(akku, buf[col], col);
    }
  } else if constexpr (HasCoord) {
    fun(akku, buf[col], Coord{col, row});
  } else {
    fun(akku, buf[col]);
  }
  IMGFUN_FOREACH_END
  return akku;
}

template <typename Img>
template <bool HasRowFun, bool HasCoord, typename RFun, typename Fun>
void ImgBase<Img>::_apply(RFun&& row_fun, Fun&& fun, Img& out) const {
  IMGFUN_FOREACH_ROW(buf, row)
  auto* out_buf = out.row_begin(row);
  IMGFUN_FOREACH_COL(col)
  if constexpr (HasRowFun) {
    out_buf[col] = fun(buf[col], col);
  } else if constexpr (HasCoord) {
    out_buf[col] = fun(buf[col], Coord{col, row});
  } else {
    out_buf[col] = fun(buf[col]);
  }
  IMGFUN_FOREACH_END
}

template <typename Img>
template <bool HasRowFun, bool HasCoord, typename RFun, typename Fun>
Img ImgBase<Img>::_map(RFun&& row_fun, Fun&& fun) const {
  auto result = create(size_, channels_);
  _apply<HasRowFun, HasCoord>(std::forward<RFun>(row_fun),
                              std::forward<Fun>(fun), result);
  return result;
}

template <typename Img>
template <typename T>
T ImgBase<Img>::reduce(
    T&& akku, std::function<void(T&, Pixel const&, Coord const&)>&& fun) const {
  return _reduce<T, T, false, true>(std::move(akku), empty_row_fun,
                                    std::forward<decltype(fun)>(fun));
}

template <typename Img>
template <typename T>
T ImgBase<Img>::reduce(T&& akku,
                       std::function<void(T&, Pixel const&)>&& fun) const {
  return _reduce<T, T, false, false>(std::move(akku), empty_row_fun,
                                     std::forward<decltype(fun)>(fun));
}

template <typename Img>
template <typename T>
T ImgBase<Img>::reduce(T&& akku, RowFun&& row_fun,
                       std::function<void(T&, Pixel const&, int)>&& fun) const {
  return _reduce<T, T, true, false>(std::move(akku),
                                    std::forward<RowFun>(row_fun),
                                    std::forward<decltype(fun)>(fun));
}

template <typename Img>
template <typename T>
std::optional<T> ImgBase<Img>::reduce_all(
    T&& akku, RowFun&& row_fun,
    std::function<bool(T&, Pixel const&, int)>&& fun) const {
  return _reduce<T, std::optional<T>, true, false>(
      std::move(akku), std::forward<RowFun>(row_fun),
      std::forward<decltype(fun)>(fun));
}

template <typename Img>
template <bool HasRowFun, bool HasCoord, typename RFun, typename Fun>
std::optional<Coord> ImgBase<Img>::_find(RFun&& row_fun, Fun&& fun) const {
  IMGFUN_FOREACH(buf, row, col)
  if constexpr (HasRowFun) {
    if (fun(buf[col], col)) {
      return Coord{col, row};
    }
  } else if constexpr (HasCoord) {
    if (fun(buf[col], Coord{col, row})) {
      return Coord{col, row};
    }
  } else {
    if (fun(buf[col])) {
      return Coord{col, row};
    }
  }
  IMGFUN_FOREACH_END

  return {};
}

template <typename Img>
void ImgBase<Img>::apply(std::function<Pixel(Pixel const&)>&& fun) {
  _apply<false, false>(empty_row_fun, std::forward<decltype(fun)>(fun),
                       *static_cast<Img*>(this));
}

template <typename Img>
void ImgBase<Img>::apply(
    std::function<Pixel(Pixel const&, Coord const&)>&& fun) {
  _apply<false, true>(empty_row_fun, std::forward<decltype(fun)>(fun),
                      *static_cast<Img*>(this));
}

template <typename Img>
void ImgBase<Img>::apply(RowFun&& row_fun,
                         std::function<Pixel(Pixel const&, int)>&& fun) {
  _apply<true, false>(std::forward<RowFun>(row_fun),
                      std::forward<decltype(fun)>(fun),
                      *static_cast<Img*>(this));
}

template <typename Img>
Img ImgBase<Img>::map(std::function<Pixel(Pixel const&)>&& fun) const {
  return _map<false, false>(empty_row_fun, std::forward<decltype(fun)>(fun));
}

template <typename Img>
Img ImgBase<Img>::map(
    std::function<Pixel(Pixel const&, Coord const&)>&& fun) const {
  return _map<false, true>(empty_row_fun, std::forward<decltype(fun)>(fun));
}

template <typename Img>
Img ImgBase<Img>::map(RowFun&& row_fun,
                      std::function<Pixel(Pixel const&, int)>&& fun) const {
  return _map<true, false>(std::forward<RowFun>(row_fun),
                           std::forward<decltype(fun)>(fun));
}

template <typename Img>
std::optional<Coord> ImgBase<Img>::find(
    std::function<bool(Pixel const&)>&& fun) const {
  return _find<false, false>(empty_row_fun, std::forward<decltype(fun)>(fun));
}

template <typename Img>
std::optional<Coord> ImgBase<Img>::find(
    RowFun&& row_fun, std::function<bool(Pixel const&, int)>&& fun) const {
  return _find<true, false>(std::forward<RowFun>(row_fun),
                            std::forward<decltype(fun)>(fun));
}

template <typename Img>
std::optional<Coord> ImgBase<Img>::find(
    std::function<bool(Pixel const&, Coord const&)>&& fun) const {
  return _find<false, true>(empty_row_fun, std::forward<decltype(fun)>(fun));
}

template <typename Img>
Pixel* ImgBase<Img>::data() {
  return data_;
}

template <typename Img>
Pixel const* ImgBase<Img>::data() const {
  return data_;
}

template <typename Img>
Pixel const& ImgBase<Img>::at(Coord const& pos) const {
  return data_[data_offset_ + pos.x + pos.y * stride_];
}

template <typename Img>
Pixel& ImgBase<Img>::at(Coord const& pos) {
  return data_[data_offset_ + pos.x + pos.y * stride_];
}

template <typename Img>
Img ImgBase<Img>::at(Coord const& upper_left, Size const& size) const {
  return Img(data_, size, channels_,
             Offset{offset_.position + upper_left, offset_.scale}, deleter_,
             data_offset_ + upper_left.x + upper_left.y * stride_, stride_);
}

template <typename Img>
Img ImgBase<Img>::crop(int const crop_width, int const crop_height) const {
  return at(Coord{crop_width, crop_height},
            Size{size_.width - crop_width * 2, size_.height - crop_height * 2});
}

template <typename Img>
Img ImgBase<Img>::crop(int const crop_width) const {
  return crop(crop_width, crop_width);
}

template <typename Img>
Histogram ImgBase<Img>::histogram() const {
  return reduce<Histogram>(
      {}, [](auto& histogram, Pixel const& p) { ++histogram[p]; });
}

template <typename Img>
std::tuple<Pixel, Pixel> ImgBase<Img>::darkest_brightest(
    int const lower_cut, int const upper_cut) const {
  auto const hist = histogram();
  auto const find_cut = [this](auto const begin, auto const end,
                               int const cut) -> int {
    auto const it = std::find_if(begin, end,
                                 [num_pixels_cut = size_.area() * cut / 100,
                                  pixel_count = 0](int const h) mutable {
                                   pixel_count += h;
                                   return pixel_count > num_pixels_cut;
                                 });
    return static_cast<int>(std::distance(begin, it));
  };

  auto const pixel_low = find_cut(hist.cbegin(), hist.cend(), lower_cut);
  auto const pixel_high =
      255 - find_cut(hist.crbegin(), hist.crend(), upper_cut);

  return std::make_tuple(pixel_low, pixel_high);
}

template <typename Img>
Img ImgBase<Img>::auto_contrast(int const cut_lower,
                                int const cut_upper) const {
  auto const [pixel_low, pixel_high] = darkest_brightest(cut_lower, cut_upper);

  if (pixel_low >= pixel_high) {
    return _shallow_copy();
  }

  // solve a * low + b = 0 (black) and a * high + b = 255 (white)
  auto const p_div = static_cast<float>(pixel_low) / pixel_high;
  auto const b = 255.5f / (p_div - 1) * p_div;
  auto const a = (255.5f - b) / pixel_high;

  return map(
      [a, b](Pixel const& p) -> Pixel { return clamp(roundp(p * a + b)); });
}

template <typename Img>
Img ImgBase<Img>::denoise() const {
  return _self()->filter_sym<2, 3, 6>();
}

template <typename Img>
template <                   // matrix
    int UL, int UC, int UR,  // first row
    int CL, int CC, int CR,  // center row
    int LL, int LC, int LR,  // lower row
    int Add,                 // add to center pixel
    int Div,                 // divider
    int Fact                 // Factor
    >
Img ImgBase<Img>::filter() const {
  return crop().map([this](Pixel const& p) -> Pixel {
    constexpr static auto div = Div ? Div : 1;
    auto const v = view_at(p);

    return clamp(
        (UL * v(-1, -1) + UC * v(0, -1) + UR * v(1, -1) +  // first row
         CL * v(-1, 0) + CC * p + CR * v(1, 0) +           // center row
         LL * v(-1, 1) + LC * v(0, 1) + LR * v(1, 1) +     // lower row
         Add)  //   Add -> reduce noise (if < 0 and when using diff filter)
        / div * Fact);
  });
}

template <typename Img>
template <int Diag, int Orth, int Center, int Add, int Div, int Fact>
Img ImgBase<Img>::filter_sym() const {
  return _self()
      ->filter<Diag, Orth, Diag, Orth, Center, Orth, Diag, Orth, Diag, Add, Div,
               Fact>();
}

template <typename Img>
template <int Add, int Fact>
Img ImgBase<Img>::laplace() const {
  return _self()->filter_sym<2, 3, -20, 1, Add, Fact>();
}

template <typename Img>
template <int Outer, int Inner, int Add>
Img ImgBase<Img>::sobel(bool const vertical) const {
  return vertical ? _self()
                        ->filter<Outer, 0, -Outer, Inner, 0, -Inner, Outer, 0,
                                 -Outer, Add>()
                  : _self()
                        ->filter<Outer, Inner, Outer, 0, 0, 0, -Outer, -Inner,
                                 -Outer, Add>();
}

float calc_rad(float const angle_grad) {
  static auto const pi = acosf(-1);
  return angle_grad / 180.f * pi;
}

template <typename Img>
Coord ImgBase<Img>::_rotate_offset(float const angle_rad) const {
  auto const sin_angle = sinf(angle_rad);

  // the resulting image is cropped
  auto const x_gap = static_cast<int>(std::abs(sin_angle * size_.width)) + 1;
  auto const y_gap = static_cast<int>(std::abs(sin_angle * size_.height)) + 1;

  return {x_gap, y_gap};
}

template <typename Img>
Img ImgBase<Img>::fine_rotate(float const angle_grad) const {
  // The general idea of the algorithm is to use 4 adjacent source pixels to get
  // the destination pixel. Each source pixel is weighted depending on the
  // actual calculated (non-discrete) position. see also:
  // http://www.leptonica.org/rotation.html#ROTATION-BY-AREA-MAPPING
  auto const angle = calc_rad(angle_grad);
  auto const [x_gap, y_gap] = _rotate_offset(angle);

  return crop(x_gap, y_gap)
      .map(
          // mid (as coord) is relative to upper_left
          [mid = Coord{div_round(size_.width - 1, 2),
                       div_round(size_.height - 1, 2)},
           cos_angle_1 = cosf(angle) - 1, sin_angle = sinf(angle),
           this](Pixel const& p, Coord const& coord) -> Pixel {
            auto const v = view_at(p);

            // rotate around the mid point
            auto const dx = coord.x - mid.x;
            auto const dy = coord.y - mid.y;

            // determine the discrete destination position for x:
            // x_le = left, x_ri = right position of the floating point
            // destination position. dest_x / dest_y is _relative_ to current
            // coord where the view v() is located.
            auto const dest_x = cos_angle_1 * dx + sin_angle * dy;
            auto const x_le = static_cast<int>(dest_x);
            auto const x_ri = x_le + sign(dest_x);
            auto const x_part_ri = std::abs(dest_x - x_le);

            // same for y: y_up = upper and y_lo = lower neighbouring position
            auto const dest_y = cos_angle_1 * dy - sin_angle * dx;
            auto const y_up = static_cast<int>(dest_y);
            auto const y_lo = y_up + sign(dest_y);

            // first interpolate on x-axis
            auto const x_part_le = 1.f - x_part_ri;
            auto const upper =
                v(x_le, y_up) * x_part_le + v(x_ri, y_up) * x_part_ri;
            auto const lower =
                v(x_le, y_lo) * x_part_le + v(x_ri, y_lo) * x_part_ri;

            // then interpolate on y-axis
            auto const y_part = std::abs(dest_y - y_up);
            return clamp(roundp(upper * (1.f - y_part) + lower * y_part));
          });
}

#ifndef USE_OPENCV
class Image : public ImgBase<Image> {
  using ImgBase::ImgBase;
};
#endif
};  // namespace imgfun

#ifdef USE_OPENCV
#include "ImgFunCv.h"
namespace imgfun {
using Image = ImageCv;
}
#endif
