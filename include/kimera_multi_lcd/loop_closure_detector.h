/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu) Yulun Tian (yulun@mit.edu)
 */

#pragma once

#include <DBoW2/DBoW2.h>
#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <kimera_multi_lcd/lcd_third_party.h>
#include <kimera_multi_lcd/types.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <xfeat-cpp/netvlad_onnx.h>

#include <iostream>
#include <map>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <unordered_map>
#include <vector>

namespace kimera_multi_lcd {

using DMatchVec = std::vector<cv::DMatch>;

class LoopClosureDetectorBase {
 public:
  virtual ~LoopClosureDetectorBase() = default;

  // Load params and initialize
  virtual void loadAndInitialize(const LcdParams& params) { lcd_params_ = params; }

 protected:
  LcdParams lcd_params_;

  // Dictionary of VLC frames
  VLCFrameDict vlc_frames_;
};

typedef DBoW2::TemplatedVocabulary<cv::Mat, DBoW2::FORB> OrbVocabulary;
struct OrbDatabaseWrapper : DBoW2::TemplatedDatabase<cv::Mat, DBoW2::FORB> {
  using Base = DBoW2::TemplatedDatabase<cv::Mat, DBoW2::FORB>;
  using GlobalDesc = DBoW2::BowVector;
  using DescMat = cv::Mat;

  // default constructor
  template <typename... Args>
  OrbDatabaseWrapper(Args&&... args) : Base(std::forward<Args>(args)...) {}
};

template <typename Database, typename FeatureDector, typename FeatureMatcher>
class LoopClosureDetector : public LoopClosureDetectorBase {
 public:
  using GlobalDesc = typename Database::GlobalDesc;

  typedef std::map<PoseId, GlobalDesc> PoseGlobalDesc;

  LoopClosureDetector();
  ~LoopClosureDetector();

  virtual std::unique_ptr<Database> createDatabase() {
    LOG(ERROR) << "createDatabase not implemented!" << std::endl;
    return nullptr;
  }

  virtual void loadAndInitialize(const LcdParams& params) override {
    LoopClosureDetectorBase::loadAndInitialize(params);
    LOG(ERROR) << "loadAndInitialize not implemented!" << std::endl;
  }

  // Add new bow vector to database
  virtual void addGlobalDesc(const RobotPoseId& id, const GlobalDesc& bow_vector) = 0;

  /**
   * @brief Find loop closure against the trajectory of the specified robot
   * @param robot
   * @param vertex_query
   * @param bow_vector_query
   * @param vertex_matches
   * @param scores If not null, also return the corresponding vector of normalized
   * score (BoW score / nss_factor)
   * @return
   */
  virtual bool detectLoopWithRobot(size_t robot,
                                   const RobotPoseId& vertex_query,
                                   const GlobalDesc& bow_vector_query,
                                   std::vector<RobotPoseId>* vertex_matches,
                                   std::vector<double>* scores = nullptr) = 0;

  /**
   * @brief Find loop closure against all robots in the database
   * @param vertex_query
   * @param bow_vector_query
   * @param vertex_matches
   * @param scores If not null, also return the corresponding vector of normalized
   * score (BoW score / nss_factor)
   * @return
   */
  bool detectLoop(const RobotPoseId& vertex_query,
                  const GlobalDesc& bow_vector_query,
                  std::vector<RobotPoseId>* vertex_matches,
                  std::vector<double>* scores = nullptr);

  void computeMatchedIndices(const RobotPoseId& vertex_query,
                             const RobotPoseId& vertex_match,
                             std::vector<unsigned int>* i_query,
                             std::vector<unsigned int>* i_match) const;
  /**
   * @brief Perform monocular RANSAC
   * @param vertex_query
   * @param vertex_match
   * @param inlier_query
   * @param inlier_match
   * @param R_query_match optional output that stores the estimated relative rotation
   * from monocular RANSAC
   * @return
   */
  bool geometricVerificationNister(const RobotPoseId& vertex_query,
                                   const RobotPoseId& vertex_match,
                                   std::vector<unsigned int>* inlier_query,
                                   std::vector<unsigned int>* inlier_match,
                                   gtsam::Rot3* R_query_match = nullptr);
  /**
   * @brief Perform stereo RANSAC
   * @param vertex_query
   * @param vertex_match
   * @param inlier_query
   * @param inlier_match
   * @param T_query_match output 3D transformation from match frame to query frame
   * @param R_query_match_prior prior estimates on the relative rotation, e.g,
   * computed with mono RANSAC. Default to nullptr in which case no prior information
   * is used.
   * @return
   */
  bool recoverPose(const RobotPoseId& vertex_query,
                   const RobotPoseId& vertex_match,
                   std::vector<unsigned int>* inlier_query,
                   std::vector<unsigned int>* inlier_match,
                   gtsam::Pose3* T_query_match,
                   const gtsam::Rot3* R_query_match_prior = nullptr);

  inline void addVLCFrame(const RobotPoseId& id, const VLCFrame& frame) {
    vlc_frames_[id] = frame;
  }

  bool globalDescExists(const RobotPoseId& id) const;

  // Try to find a BoW vector in the database with ID in the
  // range of (id-window, id-1)
  bool findPreviousGlobalDesc(const RobotPoseId& id,
                              int window = 5,
                              GlobalDesc* previous_bow = nullptr);

  int numGlobalDescsForRobot(RobotId robot_id) const;

  // For the input robot, return the latest PoseId where the BoW is stored
  // If no BoW vector is found for this robot, -1 is returned instead
  int latestPoseIdWithGlobalDesc(RobotId robot_id) const;

  GlobalDesc getGlobalDesc(const RobotPoseId& id) const;

  PoseGlobalDesc getGlobalDescs(const RobotId& robot_id) const;

  VLCFrame getVLCFrame(const RobotPoseId& id) const;

  std::map<PoseId, VLCFrame> getVLCFrames(const RobotId& robot_id) const;

  inline bool frameExists(const RobotPoseId& id) const {
    return vlc_frames_.find(id) != vlc_frames_.end();
  }

  inline size_t totalGlobalDescMatches() const { return total_global_desc_matches_; }

  inline size_t getNumGeomVerificationsMono() const {
    return total_geom_verifications_mono_;
  }

  inline size_t getNumGeomVerifications() const {
    return total_geometric_verifications_;
  }

  inline LcdParams getParams() const { return params_; }

  virtual void matchFeatures(const std::vector<cv::Point2f>& query_kpts,
                             const cv::Mat& query_desc,
                             std::vector<cv::Point2f>& train_kpts,
                             const cv::Mat& train_desc,
                             std::vector<DMatchVec>& matches) const = 0;

 protected:
  // Loop closure detection parameters
  LcdParams params_;

  // Track loop closure stats
  size_t total_global_desc_matches_;
  size_t total_geom_verifications_mono_;
  size_t total_geometric_verifications_;

  // Database of BOW vectors from each robot (trajectory)
  std::unordered_map<RobotId, PoseGlobalDesc> global_descs_;
  std::unordered_map<RobotId, std::unique_ptr<Database>> db_;
  // Keep track of latest pose Id with BoW for each robot
  std::unordered_map<RobotId, PoseId> global_desc_latest_pose_id_;
  // Map DBoW2 Entry Id to Pose Id
  std::unordered_map<RobotId, std::unordered_map<size_t, PoseId>> db_EntryId_to_PoseId_;

  // LCD third party wrapper
  std::unique_ptr<LcdThirdPartyWrapper> lcd_tp_wrapper_;

  // ORB extraction and matching members
  cv::Ptr<FeatureMatcher> feature_matcher_;
};

}  // namespace kimera_multi_lcd

#include "kimera_multi_lcd/loop_closure_detector-inl.h"