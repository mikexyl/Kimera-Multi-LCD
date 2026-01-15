#pragma once

/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu) Yulun Tian (yulun@mit.edu)
 */

#include <glog/logging.h>
#include <spdlog/fmt/bundled/core.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <string>

#include "kimera_multi_lcd/loop_closure_detector.h"

using RansacProblem =
    opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
using Adapter = opengv::relative_pose::CentralRelativeAdapter;
using AdapterStereo = opengv::point_cloud::PointCloudAdapter;
using RansacProblemStereo = opengv::sac_problems::point_cloud::PointCloudSacProblem;
using ProblemPnP = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
using AdapterPnp = opengv::absolute_pose::CentralAbsoluteAdapter;

namespace kimera_multi_lcd {

// Template implementations
template <typename Database, typename FeatureDetector, typename FeatureMatcher>
LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::LoopClosureDetector()
    : lcd_tp_wrapper_(nullptr) {
  // Track stats
  total_global_desc_matches_ = 0;
  total_geom_verifications_mono_ = 0;
  total_geometric_verifications_ = 0;
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::~LoopClosureDetector() {
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
bool LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::globalDescExists(
    const kimera_multi_lcd::RobotPoseId& id) const {
  RobotId robot_id = id.first;
  PoseId pose_id = id.second;
  if (global_descs_.find(robot_id) != global_descs_.end() &&
      global_descs_.at(robot_id).find(pose_id) != global_descs_.at(robot_id).end()) {
    return true;
  }
  return false;
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
int LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::
    numGlobalDescsForRobot(RobotId robot_id) const {
  if (global_descs_.find(robot_id) != global_descs_.end()) {
    return global_descs_.at(robot_id).size();
  }
  return 0;
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
int LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::
    latestPoseIdWithGlobalDesc(RobotId robot_id) const {
  if (numGlobalDescsForRobot(robot_id) == 0) {
    return -1;
  }
  return global_desc_latest_pose_id_.at(robot_id);
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
bool LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::
    findPreviousGlobalDesc(const RobotPoseId& id,
                           int window,
                           GlobalDesc* previous_bow) {
  CHECK_GE(window, 1);
  RobotId robot_id = id.first;
  PoseId pose_id = id.second;
  for (size_t i = 1; i <= static_cast<size_t>(window); ++i) {
    if (i > pose_id) break;
    RobotPoseId prev_id(robot_id, pose_id - i);
    if (globalDescExists(prev_id)) {
      if (previous_bow) {
        *previous_bow = getGlobalDesc(prev_id);
      }
      return true;
    }
  }
  return false;
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
typename LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::GlobalDesc
LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::getGlobalDesc(
    const kimera_multi_lcd::RobotPoseId& id) const {
  CHECK(globalDescExists(id));
  RobotId robot_id = id.first;
  PoseId pose_id = id.second;
  return global_descs_.at(robot_id).at(pose_id);
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
typename LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::PoseGlobalDesc
LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::getGlobalDescs(
    const RobotId& robot_id) const {
  if (!global_descs_.count(robot_id)) {
    return PoseGlobalDesc();
  }
  return global_descs_.at(robot_id);
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
VLCFrame LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::getVLCFrame(
    const kimera_multi_lcd::RobotPoseId& id) const {
  CHECK(frameExists(id));
  return vlc_frames_.at(id);
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
std::map<PoseId, VLCFrame>
LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::getVLCFrames(
    const RobotId& robot_id) const {
  std::map<PoseId, VLCFrame> vlc_frames;
  for (const auto& robot_pose_id_vlc : vlc_frames_) {
    if (robot_pose_id_vlc.first.first == robot_id) {
      vlc_frames[robot_pose_id_vlc.first.second] = robot_pose_id_vlc.second;
    }
  }
  return vlc_frames;
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
bool LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::detectLoop(
    const RobotPoseId& vertex_query,
    const GlobalDesc& bow_vector_query,
    std::vector<RobotPoseId>* vertex_matches,
    std::vector<double>* scores) {
  assert(NULL != vertex_matches);
  vertex_matches->clear();
  if (scores) scores->clear();
  // Detect loop with every robot in the database
  for (const auto& db : db_) {
    std::vector<RobotPoseId> vertex_matches_with_robot;
    std::vector<double> scores_with_robot;
    if (detectLoopWithRobot(db.first,
                            vertex_query,
                            bow_vector_query,
                            &vertex_matches_with_robot,
                            &scores_with_robot)) {
      vertex_matches->insert(vertex_matches->end(),
                             vertex_matches_with_robot.begin(),
                             vertex_matches_with_robot.end());
      if (scores)
        scores->insert(
            scores->end(), scores_with_robot.begin(), scores_with_robot.end());
    }
  }
  if (scores) CHECK_EQ(vertex_matches->size(), scores->size());
  if (!vertex_matches->empty()) return true;
  return false;
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
void LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::
    computeMatchedIndices(const RobotPoseId& vertex_query,
                          const RobotPoseId& vertex_match,
                          std::vector<unsigned int>* i_query,
                          std::vector<unsigned int>* i_match) const {
  VLOG(1) << "Computing matched indices between " << vertex_query.first << ":"
          << vertex_query.second << " and " << vertex_match.first << ":"
          << vertex_match.second << ".";
  CHECK_NOTNULL(i_match);
  CHECK_NOTNULL(i_query);
  i_query->clear();
  i_match->clear();

  // Get two best matches between frame descriptors.
  std::vector<DMatchVec> matches;

  CHECK(vlc_frames_.find(vertex_query) != vlc_frames_.end())
      << "VLCFrame for query " << vertex_query.first << ":" << vertex_query.second
      << " does not exist.";
  CHECK(vlc_frames_.find(vertex_match) != vlc_frames_.end())
      << "VLCFrame for match " << vertex_match.first << ":" << vertex_match.second
      << " does not exist.";

  VLCFrame frame_query = vlc_frames_.find(vertex_query)->second;
  VLCFrame frame_match = vlc_frames_.find(vertex_match)->second;

  // check if frames have keypoints and descriptors
  CHECK(!frame_query.keypoints_.empty())
      << "VLCFrame for query " << vertex_query.first << ":" << vertex_query.second
      << " has no keypoints.";
  CHECK(!frame_query.descriptors_mat_.empty())
      << "VLCFrame for query " << vertex_query.first << ":" << vertex_query.second
      << " has no descriptors.";
  CHECK(!frame_match.keypoints_.empty())
      << "VLCFrame for match " << vertex_match.first << ":" << vertex_match.second
      << " has no keypoints.";
  CHECK(!frame_match.descriptors_mat_.empty())
      << "VLCFrame for match " << vertex_match.first << ":" << vertex_match.second
      << " has no descriptors.";

  // check size consistency
  CHECK_EQ(frame_query.keypoints_.size(), frame_query.descriptors_mat_.rows)
      << "VLCFrame for query " << vertex_query.first << ":" << vertex_query.second
      << " has inconsistent keypoints and descriptors size.";
  CHECK_EQ(frame_match.keypoints_.size(), frame_match.descriptors_mat_.rows)
      << "VLCFrame for match " << vertex_match.first << ":" << vertex_match.second
      << " has inconsistent keypoints and descriptors size.";

  matchFeatures(frame_query.keypoints_,
                frame_query.descriptors_mat_,
                frame_match.keypoints_,
                frame_match.descriptors_mat_,
                matches);

  // remove empty matches
  matches.erase(
      std::remove_if(
          matches.begin(), matches.end(), [](const DMatchVec& m) { return m.empty(); }),
      matches.end());

  VLOG(1) << "Found " << matches.size() << " matches.";

  const size_t& n_matches = matches.size();
  for (size_t i = 0; i < n_matches; i++) {
    const DMatchVec& match = matches[i];
    i_query->push_back(match[0].queryIdx);
    i_match->push_back(match[0].trainIdx);
  }

  auto robot_label = fmt::format("{}-{}", vertex_query.first, vertex_match.first);

  // visualizer_->visualizeMatchesKeypoints(
  // robot_label + "-glue", &frame_query, &frame_match, *i_query, *i_match);
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
bool LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::
    geometricVerificationNister(const RobotPoseId& vertex_query,
                                const RobotPoseId& vertex_match,
                                std::vector<unsigned int>* inlier_query,
                                std::vector<unsigned int>* inlier_match,
                                gtsam::Rot3* R_query_match) {
  assert(NULL != inlier_query);
  assert(NULL != inlier_match);

  total_geom_verifications_mono_++;

  std::vector<unsigned int> i_query = *inlier_query;
  std::vector<unsigned int> i_match = *inlier_match;

  BearingVectors query_versors, match_versors;

  query_versors.resize(i_query.size());
  match_versors.resize(i_match.size());
  for (size_t i = 0; i < i_match.size(); i++) {
    query_versors[i] = vlc_frames_[vertex_query].versors_.at(i_query[i]);
    match_versors[i] = vlc_frames_[vertex_match].versors_.at(i_match[i]);
  }

  auto robot_label = fmt::format("{}-{}", vertex_query.first, vertex_match.first);

  VLOG(1) << "Preparing RANSAC with " << query_versors.size() << " correspondences";

  // Check for valid versors
  size_t valid_versors = 0;
  for (size_t i = 0; i < query_versors.size(); i++) {
    if (query_versors[i].norm() > 1e-6 && match_versors[i].norm() > 1e-6) {
      valid_versors++;
    } else {
      LOG(WARNING) << "Invalid versor at index " << i
                   << " query norm: " << query_versors[i].norm()
                   << " match norm: " << match_versors[i].norm();
    }
  }
  VLOG(1) << "Valid versors: " << valid_versors << " / " << query_versors.size();

  if (query_versors.size() < 5) {
    LOG(WARNING) << "Too few correspondences (" << query_versors.size()
                 << ") for RANSAC, need at least 5";
    return false;
  }

  Adapter adapter(query_versors, match_versors);

  // Use RANSAC to solve the central-relative-pose problem.
  opengv::sac::Ransac<RansacProblem> ransac;

  ransac.sac_model_ =
      std::make_shared<RansacProblem>(adapter, RansacProblem::Algorithm::NISTER, true);
  ransac.max_iterations_ = params_.max_ransac_iterations_mono_;
  ransac.threshold_ = params_.ransac_threshold_mono_;

  // Compute transformation via RANSAC.
  VLOG(1) << "Starting Monocular RANSAC for geometric verification.";
  VLOG(1) << "RANSAC params - max_iterations: " << params_.max_ransac_iterations_mono_
          << " threshold: " << params_.ransac_threshold_mono_;
  auto time_ransac_start = std::chrono::high_resolution_clock::now();
  bool ransac_success = ransac.computeModel();
  auto time_ransac_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_ransac = time_ransac_end - time_ransac_start;
  VLOG(1) << "Monocular RANSAC took " << elapsed_ransac.count() * 1000 << " ms.";
  VLOG(1) << "RANSAC result - success: " << ransac_success
          << " iterations performed: " << ransac.iterations_
          << " inliers found: " << ransac.inliers_.size();
  inlier_query->clear();
  inlier_match->clear();
  if (ransac_success) {
    double inlier_percentage =
        static_cast<double>(ransac.inliers_.size()) / query_versors.size();

    VLOG(1) << "Monocular RANSAC found " << ransac.inliers_.size()
            << " inliers, with inlier percentage " << inlier_percentage << ".";

    if (inlier_percentage >= params_.ransac_inlier_percentage_mono_) {
      if (R_query_match) {
        opengv::transformation_t monoT_query_match = ransac.model_coefficients_;
        *R_query_match = gtsam::Rot3(monoT_query_match.block<3, 3>(0, 0));
      }

      for (auto idx : ransac.inliers_) {
        inlier_query->push_back(i_query[idx]);
        inlier_match->push_back(i_match[idx]);
      }
      return true;
    }
  }
  // visualizer_->visualizeMatchesKeypoints(robot_label + "-mono",
  //                                        &vlc_frames_[vertex_query],
  //                                        &vlc_frames_[vertex_match],
  //                                        *inlier_query,
  //                                        *inlier_match);
  return false;
}

template <typename Database, typename FeatureDetector, typename FeatureMatcher>
bool LoopClosureDetector<Database, FeatureDetector, FeatureMatcher>::recoverPose(
    const RobotPoseId& vertex_query,
    const RobotPoseId& vertex_match,
    std::vector<unsigned int>* inlier_query,
    std::vector<unsigned int>* inlier_match,
    gtsam::Pose3* T_query_match,
    const gtsam::Rot3* R_query_match_prior) {
  CHECK_NOTNULL(inlier_query);
  CHECK_NOTNULL(inlier_match);
  total_geometric_verifications_++;
  std::vector<unsigned int> i_query;  // input indices to stereo ransac
  std::vector<unsigned int> i_match;

  std::vector<gtsam::Vector3, Eigen::aligned_allocator<gtsam::Vector3>>
      bearing_vectors_query, bearing_vectors_match;
  std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>> points_query,
      points_match;

  for (size_t i = 0; i < inlier_match->size(); i++) {
    gtsam::Vector3 point_query =
        vlc_frames_[vertex_query].landmarks_.at(inlier_query->at(i));
    gtsam::Vector3 point_match =
        vlc_frames_[vertex_match].landmarks_.at(inlier_match->at(i));
    auto bearing_vector_query =
        vlc_frames_[vertex_query].versors_.at(inlier_query->at(i));
    auto bearing_vector_match =
        vlc_frames_[vertex_match].versors_.at(inlier_match->at(i));
    if (point_query.norm() > 1e-3) {
      points_query.push_back(point_query);
      bearing_vectors_match.push_back(bearing_vector_match);
    }
    if (point_match.norm() > 1e-3) {
      points_match.push_back(point_match);
      bearing_vectors_query.push_back(bearing_vector_query);
    }
  }

  // if (f_query.size() < 3) {
  //   LOG(WARNING) << "Too few 3D-3D correspondences (" << f_query.size()
  //                << ") for RANSAC, need at least 3.";
  //   return false;
  // }

  AdapterPnp adapter(bearing_vectors_query, points_match);
  if (R_query_match_prior) {
    // Use input rotation estimate as prior
    adapter.setR(R_query_match_prior->inverse().matrix());
  }

  // Should be similar to current klt_eps, but keep it separate.
  const double reprojection_error = params_.ransac_threshold_;
  const double avg_focal_length = params_.avg_focal_length_;
  const double threshold =
      1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error / avg_focal_length));

  // Compute transform using RANSAC 3-point method (Arun).
  // Forward direction: query <- match
  std::shared_ptr<ProblemPnP> ptcloudproblem_ptr(
      new ProblemPnP(adapter, ProblemPnP::EPNP, true));
  opengv::sac::Ransac<ProblemPnP> ransac_forward;
  ransac_forward.sac_model_ = ptcloudproblem_ptr;
  ransac_forward.max_iterations_ = params_.max_ransac_iterations_;
  ransac_forward.threshold_ = threshold;
  ransac_forward.probability_ = 0.99;

  // log input sizes of RANSAC
  VLOG(1) << "RANSAC input - total correspondences: " << bearing_vectors_match.size()
          << " x " << bearing_vectors_query.size();

  // log details and timing of ransac
  VLOG(1) << "Starting Stereo RANSAC (forward) for geometric verification.";
  VLOG(1) << "RANSAC params - max_iterations: " << params_.max_ransac_iterations_
          << " threshold: " << params_.ransac_threshold_;
  auto time_ransac_start = std::chrono::high_resolution_clock::now();

  // Compute transformation via RANSAC (forward).
  bool ransac_forward_success = ransac_forward.computeModel();
  auto time_ransac_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_ransac = time_ransac_end - time_ransac_start;
  VLOG(1) << "Stereo RANSAC (forward) took " << elapsed_ransac.count() * 1000 << " ms.";
  VLOG(1) << "RANSAC result (forward) - success: " << ransac_forward_success
          << " iterations performed: " << ransac_forward.iterations_
          << " inliers found: " << ransac_forward.inliers_.size();

  gtsam::Pose3 T_mcam_qcam;

  if (ransac_forward_success) {
    if (ransac_forward.inliers_.size() <
        params_.geometric_verification_min_inlier_count_) {
      VLOG(1) << "Number of inlier correspondences after RANSAC (forward) "
              << ransac_forward.inliers_.size() << " is too low.";
      return false;
    }

    double inlier_percentage_forward =
        static_cast<double>(ransac_forward.inliers_.size()) /
        bearing_vectors_match.size();
    if (inlier_percentage_forward <
        params_.geometric_verification_min_inlier_percentage_) {
      VLOG(1) << "Percentage of inlier correspondences after RANSAC (forward) "
              << inlier_percentage_forward << " is too low.";
      return false;
    }

    opengv::transformation_t T_forward;
    ptcloudproblem_ptr->optimizeModelCoefficients(
        ransac_forward.inliers_, ransac_forward.model_coefficients_, T_forward);
    gtsam::Matrix T_mat = gtsam::Matrix::Identity(4, 4);
    T_mat.block<3, 4>(0, 0) = T_forward;
    T_mcam_qcam = gtsam::Pose3(T_mat);
  } else {
    // run backward RANSAC if forward fails
    VLOG(1) << "Starting Stereo RANSAC (backward)";
    AdapterPnp adapter_backward(bearing_vectors_match, points_query);
    if (R_query_match_prior) {
      // Use input rotation estimate as prior
      adapter_backward.setR(R_query_match_prior->matrix());
    }
    auto pnp_backward_problem_ptr =
        std::make_shared<ProblemPnP>(adapter_backward, ProblemPnP::EPNP, true);
    opengv::sac::Ransac<ProblemPnP> ransac_backward;
    ransac_backward.sac_model_ = pnp_backward_problem_ptr;
    ransac_backward.max_iterations_ = params_.max_ransac_iterations_;
    ransac_backward.threshold_ = threshold;
    ransac_backward.probability_ = 0.99;
    bool ransac_backward_success = ransac_backward.computeModel();
    VLOG(1) << "RANSAC result (backward) - success: " << ransac_backward_success
            << " iterations performed: " << ransac_backward.iterations_
            << " inliers found: " << ransac_backward.inliers_.size();
    if (not ransac_backward_success) {
      return false;
    }
    if (ransac_backward.inliers_.size() <
        params_.geometric_verification_min_inlier_count_) {
      VLOG(1) << "Number of inlier correspondences after RANSAC (backward) "
              << ransac_backward.inliers_.size() << " is too low.";
      return false;
    }
    if (static_cast<double>(ransac_backward.inliers_.size()) /
            bearing_vectors_query.size() <
        params_.geometric_verification_min_inlier_percentage_) {
      VLOG(1) << "Percentage of inlier correspondences after RANSAC (backward) "
              << static_cast<double>(ransac_backward.inliers_.size()) /
                     bearing_vectors_query.size()
              << " is too low.";
      return false;
    }

    opengv::transformation_t T_backward;
    pnp_backward_problem_ptr->optimizeModelCoefficients(
        ransac_backward.inliers_, ransac_backward.model_coefficients_, T_backward);
    gtsam::Matrix T_mat = gtsam::Matrix::Identity(4, 4);
    T_mat.block<3, 4>(0, 0) = T_backward;
    gtsam::Pose3 T_qcam_mcam = gtsam::Pose3(T_mat);
    T_mcam_qcam = T_qcam_mcam.inverse();
  }

  gtsam::Pose3 T_qbody_qcam = vlc_frames_[vertex_query].T_base_cam_;
  gtsam::Pose3 T_mbody_mcam = vlc_frames_[vertex_match].T_base_cam_;

  // check if T_base_cam_ is identity
  if (T_qbody_qcam.equals(gtsam::Pose3(), 1e-6)) {
    LOG(FATAL) << "T_base_cam_ for query frame " << vertex_query.first << ":"
               << vertex_query.second << " is identity.";
  }
  if (T_mbody_mcam.equals(gtsam::Pose3(), 1e-6)) {
    LOG(FATAL) << "T_base_cam_ for match frame " << vertex_match.first << ":"
               << vertex_match.second << " is identity.";
  }

  gtsam::Pose3 T_qbody_mbody =
      T_qbody_qcam * T_mcam_qcam.inverse() * T_mbody_mcam.inverse();

  // Output is the averaged 3D transformation from the match frame to the query frame
  *T_query_match = T_qbody_mbody;

  // Populate inlier indices (use forward direction inliers)
  inlier_query->clear();
  inlier_match->clear();

  // TODO: inliers doesn't seem needed but only the count
  inlier_query->resize(ransac_forward.inliers_.size());

  return true;
}

}  // namespace kimera_multi_lcd