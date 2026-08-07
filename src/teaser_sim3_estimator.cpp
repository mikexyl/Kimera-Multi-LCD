#include "kimera_multi_lcd/teaser_sim3_estimator.h"

#include <teaser/registration.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "kimera_multi_lcd/sim3_utils.h"

namespace kimera_multi_lcd {
namespace {

bool isValidPoint(const gtsam::Point3& point) {
  return point.allFinite() && point.norm() > 1e-3;
}

bool validParams(const TeaserSim3Params& params) {
  return std::isfinite(params.noise_bound_m) && params.noise_bound_m > 0.0 &&
         std::isfinite(params.min_scale) && params.min_scale > 0.0 &&
         std::isfinite(params.max_scale) &&
         params.max_scale >= params.min_scale &&
         std::isfinite(params.min_inlier_percentage) &&
         params.min_inlier_percentage >= 0.0 &&
         params.min_inlier_percentage <= 1.0;
}

}  // namespace

TeaserSim3Estimate estimateTeaserSim3(
    const std::vector<gtsam::Point3,
                      Eigen::aligned_allocator<gtsam::Point3>>& source_points,
    const std::vector<gtsam::Point3,
                      Eigen::aligned_allocator<gtsam::Point3>>& destination_points,
    const std::vector<std::pair<unsigned int, unsigned int>>& descriptor_pairs,
    const TeaserSim3Params& params) {
  TeaserSim3Estimate result;
  if (!validParams(params) || source_points.size() != destination_points.size() ||
      source_points.size() != descriptor_pairs.size()) {
    return result;
  }

  std::vector<size_t> valid_to_input;
  valid_to_input.reserve(source_points.size());
  for (size_t index = 0; index < source_points.size(); ++index) {
    if (isValidPoint(source_points[index]) &&
        isValidPoint(destination_points[index])) {
      valid_to_input.push_back(index);
    }
  }
  result.valid_pair_count = valid_to_input.size();
  if (result.valid_pair_count < 3 ||
      result.valid_pair_count < params.min_inlier_count) {
    return result;
  }

  Eigen::Matrix<double, 3, Eigen::Dynamic> source(3, valid_to_input.size());
  Eigen::Matrix<double, 3, Eigen::Dynamic> destination(3, valid_to_input.size());
  for (size_t column = 0; column < valid_to_input.size(); ++column) {
    source.col(column) = source_points[valid_to_input[column]];
    destination.col(column) = destination_points[valid_to_input[column]];
  }

  teaser::RobustRegistrationSolver::Params solver_params;
  solver_params.noise_bound = params.noise_bound_m;
  solver_params.cbar2 = 1.0;
  solver_params.estimate_scaling = true;
  solver_params.rotation_estimation_algorithm =
      teaser::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  solver_params.rotation_tim_graph =
      teaser::RobustRegistrationSolver::INLIER_GRAPH_FORMULATION::CHAIN;
  solver_params.inlier_selection_mode =
      teaser::RobustRegistrationSolver::INLIER_SELECTION_MODE::PMC_HEU;

  try {
    teaser::RobustRegistrationSolver solver(solver_params);
    const teaser::RegistrationSolution solution = solver.solve(source, destination);
    if (!solution.valid || !std::isfinite(solution.scale) ||
        solution.scale <= 0.0 || solution.scale < params.min_scale ||
        solution.scale > params.max_scale || !solution.rotation.allFinite() ||
        !solution.translation.allFinite()) {
      return result;
    }

    const auto input_inliers = solver.getInputOrderedTranslationInliers();
    if (input_inliers.size() < params.min_inlier_count ||
        static_cast<double>(input_inliers.size()) /
                static_cast<double>(result.valid_pair_count) <
            params.min_inlier_percentage) {
      return result;
    }

    result.descriptor_inlier_pairs.reserve(input_inliers.size());
    for (const int valid_index : input_inliers) {
      if (valid_index < 0 ||
          static_cast<size_t>(valid_index) >= valid_to_input.size()) {
        return TeaserSim3Estimate{};
      }
      result.descriptor_inlier_pairs.push_back(
          descriptor_pairs[valid_to_input[static_cast<size_t>(valid_index)]]);
    }
    result.destination_T_source = similarityFromPhysical(
        gtsam::Rot3(solution.rotation),
        gtsam::Point3(solution.translation),
        solution.scale);
    result.success = true;
    return result;
  } catch (const std::exception&) {
    return result;
  }
}

}  // namespace kimera_multi_lcd
