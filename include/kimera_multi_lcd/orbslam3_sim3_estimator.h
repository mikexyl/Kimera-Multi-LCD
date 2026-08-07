#pragma once

#include <gtsam/geometry/Similarity3.h>

#include <utility>
#include <vector>

namespace kimera_multi_lcd {

struct Orbslam3Sim3Params {
  double reprojection_threshold_px = 3.0;
  double average_focal_length_px = 900.0;
  double min_scale = 0.5;
  double max_scale = 2.0;
  size_t min_inlier_count = 10;
  double min_inlier_percentage = 0.12;
  int max_ransac_iterations = 1000;
  double ransac_probability = 0.99;
};

struct Orbslam3Sim3Estimate {
  bool success = false;
  gtsam::Similarity3 destination_T_source;
  std::vector<std::pair<unsigned int, unsigned int>> descriptor_inlier_pairs;
  size_t valid_pair_count = 0;
};

// Uses pySLAM's ORB-SLAM3-style three-point Horn Sim3 RANSAC with symmetric
// reprojection checks. Source and destination points are aligned with
// descriptor_pairs. The returned transform maps source points into the
// destination frame.
Orbslam3Sim3Estimate estimateOrbslam3Sim3(
    const std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>>&
        source_points,
    const std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>>&
        destination_points,
    const std::vector<std::pair<unsigned int, unsigned int>>& descriptor_pairs,
    const Orbslam3Sim3Params& params);

}  // namespace kimera_multi_lcd
