#include "kimera_multi_lcd/orbslam3_sim3_estimator.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Random.h"
#include "Sim3Solver.h"
#include "kimera_multi_lcd/sim3_utils.h"

namespace kimera_multi_lcd {
namespace {

constexpr double kChiSquare2D99 = 9.210;

bool isValidCameraPoint(const gtsam::Point3& point) {
  if (!point.allFinite() || point.z() <= 1e-3 || point.norm() <= 1e-3) {
    return false;
  }
  return point.cwiseAbs().maxCoeff() <=
         static_cast<double>(std::numeric_limits<float>::max());
}

bool validParams(const Orbslam3Sim3Params& params) {
  return std::isfinite(params.reprojection_threshold_px) &&
         params.reprojection_threshold_px > 0.0 &&
         std::isfinite(params.average_focal_length_px) &&
         params.average_focal_length_px > 0.0 && std::isfinite(params.min_scale) &&
         params.min_scale > 0.0 && std::isfinite(params.max_scale) &&
         params.max_scale >= params.min_scale &&
         std::isfinite(params.min_inlier_percentage) &&
         params.min_inlier_percentage >= 0.0 && params.min_inlier_percentage <= 1.0 &&
         params.min_inlier_count <=
             static_cast<size_t>(std::numeric_limits<int>::max()) &&
         params.max_ransac_iterations > 0 && std::isfinite(params.ransac_probability) &&
         params.ransac_probability > 0.0 && params.ransac_probability < 1.0;
}

}  // namespace

Orbslam3Sim3Estimate estimateOrbslam3Sim3(
    const std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>>&
        source_points,
    const std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>>&
        destination_points,
    const std::vector<std::pair<unsigned int, unsigned int>>& descriptor_pairs,
    const Orbslam3Sim3Params& params) {
  Orbslam3Sim3Estimate result;
  if (!validParams(params) || source_points.size() != destination_points.size() ||
      source_points.size() != descriptor_pairs.size()) {
    return result;
  }

  std::vector<size_t> valid_to_input;
  valid_to_input.reserve(source_points.size());
  for (size_t index = 0; index < source_points.size(); ++index) {
    if (isValidCameraPoint(source_points[index]) &&
        isValidCameraPoint(destination_points[index])) {
      valid_to_input.push_back(index);
    }
  }
  result.valid_pair_count = valid_to_input.size();
  const size_t required_inliers = std::max<size_t>(3, params.min_inlier_count);
  if (result.valid_pair_count < required_inliers) {
    return result;
  }

  utils::Sim3SolverInput2 input;
  input.mvX3Dc1.reserve(result.valid_pair_count);
  input.mvX3Dc2.reserve(result.valid_pair_count);
  input.mvSigmaSquare1.reserve(result.valid_pair_count);
  input.mvSigmaSquare2.reserve(result.valid_pair_count);
  const float sigma_square =
      static_cast<float>(params.reprojection_threshold_px *
                         params.reprojection_threshold_px / kChiSquare2D99);
  for (const size_t input_index : valid_to_input) {
    // The upstream solver estimates T12, which maps camera 2 into camera 1.
    // Camera 1 is therefore the query/destination frame.
    input.mvX3Dc1.push_back(destination_points[input_index].cast<float>());
    input.mvX3Dc2.push_back(source_points[input_index].cast<float>());
    input.mvSigmaSquare1.push_back(sigma_square);
    input.mvSigmaSquare2.push_back(sigma_square);
  }
  input.K1 = Eigen::Matrix3f::Identity();
  input.K2 = Eigen::Matrix3f::Identity();
  input.K1(0, 0) = static_cast<float>(params.average_focal_length_px);
  input.K1(1, 1) = static_cast<float>(params.average_focal_length_px);
  input.K2(0, 0) = static_cast<float>(params.average_focal_length_px);
  input.K2(1, 1) = static_cast<float>(params.average_focal_length_px);
  input.bFixScale = false;

  // Keep sampling repeatable across offline replay and focused tests.
  utils::Random::SeedRand(0);
  utils::Sim3Solver solver(input);
  solver.SetRansacParameters(params.ransac_probability,
                             static_cast<int>(required_inliers),
                             params.max_ransac_iterations);
  std::vector<uint8_t> inlier_mask;
  int inlier_count = 0;
  bool converged = false;
  solver.find(inlier_mask, inlier_count, converged);
  if (!converged || inlier_count < static_cast<int>(required_inliers) ||
      inlier_mask.size() != result.valid_pair_count ||
      static_cast<double>(inlier_count) / static_cast<double>(result.valid_pair_count) <
          params.min_inlier_percentage) {
    return result;
  }

  const double scale = solver.GetEstimatedScale();
  const Eigen::Matrix3d rotation = solver.GetEstimatedRotation().cast<double>();
  const gtsam::Point3 translation = solver.GetEstimatedTranslation().cast<double>();
  if (!std::isfinite(scale) || scale <= 0.0 || scale < params.min_scale ||
      scale > params.max_scale || !rotation.allFinite() || !translation.allFinite() ||
      std::abs(rotation.determinant() - 1.0) > 1e-3) {
    return result;
  }

  result.descriptor_inlier_pairs.reserve(static_cast<size_t>(inlier_count));
  for (size_t valid_index = 0; valid_index < inlier_mask.size(); ++valid_index) {
    if (inlier_mask[valid_index]) {
      result.descriptor_inlier_pairs.push_back(
          descriptor_pairs[valid_to_input[valid_index]]);
    }
  }
  if (result.descriptor_inlier_pairs.size() != static_cast<size_t>(inlier_count)) {
    return Orbslam3Sim3Estimate{};
  }
  result.destination_T_source =
      similarityFromPhysical(gtsam::Rot3(rotation), translation, scale);
  result.success = true;
  return result;
}

}  // namespace kimera_multi_lcd
