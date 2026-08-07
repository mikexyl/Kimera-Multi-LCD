#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>

#include <cmath>
#include <stdexcept>

namespace kimera_multi_lcd {

// GTSAM stores Similarity3 translation before multiplication by scale. Public
// loop messages and Pose3 projections use the physical translation instead.
inline gtsam::Similarity3 similarityFromPhysical(const gtsam::Rot3& rotation,
                                                  const gtsam::Point3& translation,
                                                  double scale) {
  if (!std::isfinite(scale) || scale <= 0.0 || !translation.allFinite() ||
      !rotation.matrix().allFinite()) {
    throw std::invalid_argument("Invalid physical Sim3 measurement");
  }
  return gtsam::Similarity3(rotation, translation / scale, scale);
}

inline gtsam::Similarity3 similarityFromPose(const gtsam::Pose3& pose) {
  return similarityFromPhysical(pose.rotation(), pose.translation(), 1.0);
}

inline gtsam::Pose3 physicalPose(const gtsam::Similarity3& similarity) {
  return gtsam::Pose3(similarity.rotation(),
                      similarity.scale() * similarity.translation());
}

}  // namespace kimera_multi_lcd
