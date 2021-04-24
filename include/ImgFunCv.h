#pragma once

#include <opencv2/opencv.hpp>
#include "ImgFun.h"

namespace imgfun {

template <typename Img>
class ImgBaseCv : public imgfun::ImgBase<Img> {
 public:
  // use the same constructors - leave them private
  using ImgBase::ImgBase;

 public:
  /// Moves the matrix to its image representation.
  /// @param matrix the opencv matrix to be moved.
  /// @return the image representation which also contains the moved matrix.
  /// When destroyed the matrix gets also destroyed.
  static Img from_matrix(cv::Mat&& matrix);

  /// Creates a cv matrix from this by referencing the content of this image.
  /// However the data is _not_ moved over to the new matrix. When this gets
  /// destroyed the matrix content will also be destroyed!
  /// Use `as_matrix().clone()` to get an independent copy.
  /// @return the opencv matrix.
  cv::Mat as_matrix() const;

  template <                   // matrix
      int UL, int UC, int UR,  // first row
      int CL, int CC, int CR,  // center row
      int LL, int LC, int LR,  // lower row
      int Add = 0, int Div = UL + UC + UR + CL + CC + CR + LL + LC + LR,
      int Fact = 1>
  Img filter() const;

  Img filter2D(cv::Mat const& kernel, cv::Point anchor = Point(-1, -1),
               double delta = 0, int borderType = cv::BORDER_DEFAULT) const;

  Img fine_rotate(float const angle_grad) const;
};

template <typename Img>
Img ImgBaseCv<Img>::from_matrix(cv::Mat&& matrix) {
  struct MatDeleter : Deleter {
    MatDeleter(cv::Mat&& matrix) : Deleter(), matrix_(matrix) {}
    cv::Mat matrix_;
  };
  // TODO support more channels / matrix data types
  auto* image_data = matrix.ptr(0);
  auto const size = Size{matrix.cols, matrix.rows};
  return Img(image_data, size, 1, {},
             std::make_shared<MatDeleter>(std::move(matrix)));
}

template <typename Img>
cv::Mat ImgBaseCv<Img>::as_matrix() const {
  // TODO support number of channels != 1
  IMGFUN_ASSERT(channels_ == 1);
  return cv::Mat(size_.height, size_.width, CV_8UC1, &data_[data_offset_],
                 stride_);
}

template <typename Img>
Img ImgBaseCv<Img>::filter2D(cv::Mat const& kernel, cv::Point anchor,
                             double delta, int borderType) const {
  cv::Mat mat;
  cv::filter2D(as_matrix(), mat, 0, kernel, anchor, delta, borderType);

  return from_matrix(std::move(mat));
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
Img ImgBaseCv<Img>::filter() const {
  constexpr static auto div = Div ? Div : 1;
  constexpr static auto M = [](int const x) -> float {
    return (static_cast<float>(x) * Fact) / div;
  };
  auto const kernel = cv::Mat{
      {3, 3}, {M(UL), M(UC), M(UR), M(CL), M(CC), M(CR), M(LL), M(LC), M(LR)}};
  return filter2D(kernel, {-1, -1}, Add).crop();
}

template <typename Img>
Img ImgBaseCv<Img>::fine_rotate(float const angle_grad) const {
  auto const mid = cv::Point2d((size_.width - 1) / 2.0, (size_.height - 1) / 2.0);
  auto rot = cv::getRotationMatrix2D(mid, -angle_grad, 1);
  cv::Mat dst;
  
  cv::warpAffine(as_matrix(), dst, rot, {size_.width, size_.height},
                 cv::INTER_LANCZOS4);
  auto const [x_gap, y_gap] = _rotate_offset(calc_rad(angle_grad));
  return from_matrix(std::move(dst))
      .at(Coord{x_gap + 1, y_gap - 1},
          Size{size_.width - x_gap * 2, size_.height - y_gap * 2});
}

class ImageCv : public ImgBaseCv<ImageCv> {
  using ImgBaseCv::ImgBaseCv;
};
};  // namespace imgfun
