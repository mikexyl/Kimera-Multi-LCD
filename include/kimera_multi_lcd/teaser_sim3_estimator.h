#pragma once

#include <gtsam/geometry/Similarity3.h>

#include <utility>
#include <vector>

namespace kimera_multi_lcd {

struct TeaserSim3Params {
  double noise_bound_m = 0.10;
  double min_scale = 0.5;
  double max_scale = 2.0;
  size_t min_inlier_count = 10;
  double min_inlier_percentage = 0.12;
};

struct TeaserSim3Estimate {
  bool success = false;
  gtsam::Similarity3 destination_T_source;
  std::vector<std::pair<unsigned int, unsigned int>> descriptor_inlier_pairs;
  size_t valid_pair_count = 0;
};

// Source and destination points are aligned with descriptor_pairs. Invalid
// finite/depth pairs are discarded before solving. The returned transform maps
// source points into the destination frame.
TeaserSim3Estimate estimateTeaserSim3(
    const std::vector<gtsam::Point3,
                      Eigen::aligned_allocator<gtsam::Point3>>& source_points,
    const std::vector<gtsam::Point3,
                      Eigen::aligned_allocator<gtsam::Point3>>& destination_points,
    const std::vector<std::pair<unsigned int, unsigned int>>& descriptor_pairs,
    const TeaserSim3Params& params);

}  // namespace kimera_multi_lcd
