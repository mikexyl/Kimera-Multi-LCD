#pragma once

#include <gtsam/geometry/Pose3.h>

#include <opencv2/core.hpp>
#include <utility>
#include <vector>

#include "kimera_multi_lcd/types.h"

namespace kimera_multi_lcd {

class VLCFrame;  // Forward declaration

class Visualizer {
 public:
  virtual ~Visualizer() = default;

  /**
   * @brief Visualize matches between versors from two frames
   * @param frame1 The first VLCFrame
   * @param frame2 The second VLCFrame
   * @param matches Vector of pairs (index in frame1, index in frame2)
   */
  virtual void visualizeMatchesVersors(VLCFrame* frame1,
                                       VLCFrame* frame2,
                                       const BearingVectors& versors1,
                                       const BearingVectors& versors2) = 0;

  /**
   * @brief Visualize matches between keypoints from two frames
   * @param frame1 The first VLCFrame
   * @param frame2 The second VLCFrame
   * @param matches Vector of pairs (index in frame1, index in frame2)
   */
  virtual void visualizeMatchesKeypoints(const std::string& entity,
                                         VLCFrame* frame1,
                                         VLCFrame* frame2,
                                         const std::vector<unsigned int>& match1,
                                         const std::vector<unsigned int>& match2) = 0;

  virtual void visualizeCandidates(std::string name,
                                   const RobotPoseId& query_id,
                                   const std::vector<RobotPoseId>& candidate_ids,
                                   const std::vector<float>& candidate_scores) = 0;
};

}  // namespace kimera_multi_lcd
