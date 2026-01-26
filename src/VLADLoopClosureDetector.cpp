#include <glog/logging.h>

#include "kimera_multi_lcd/vlad_loop_closure_detector.h"

namespace kimera_multi_lcd {

DEFINE_double(max_nss_vlad_distance,
              0.999,  // turn this off for now
              "Maximum NSS distance for VLAD loop closure detection.");

bool VLADLoopClosureDetector::detectLoopWithRobot(
    size_t robot,
    const RobotPoseId& vertex_query,
    const VLADLoopClosureDetector::GlobalDesc& bow_vec,
    std::vector<RobotPoseId>* vertex_matches,
    std::vector<double>* scores) {
  LOG(INFO) << "VLADLoopClosureDetector: detectLoopWithRobot called for robot " << robot
            << ", vertex_query=(" << vertex_query.first << "," << vertex_query.second
            << ").";
  auto query_frame_outside_local_window =
      this->findFirstRobotPoseIdOutsideLocalWindow(vertex_query);
  if (query_frame_outside_local_window and
      this->detectLoopOutsideLocalWindow(
          robot, *query_frame_outside_local_window, bow_vec, vertex_matches, scores)) {
    LOG(INFO) << "VLADLoopClosureDetector: Loop closures detected with robot " << robot
              << " for vertex_query=(" << vertex_query.first << ","
              << vertex_query.second << ").";
    return true;
  } else {
    LOG(INFO) << "VLADLoopClosureDetector: No loop closures detected with robot "
              << robot << " for vertex_query=(" << vertex_query.first << ","
              << vertex_query.second << ").";
    // empty return
    vertex_matches->clear();
    if (scores) {
      scores->clear();
    }
    return false;
  }
}

bool VLADLoopClosureDetector::detectLoopOutsideLocalWindow(
    size_t robot,
    const RobotPoseId& robot_pose_id,
    const Database::GlobalDesc&
        global_desc,  // not used, leave it here for compatibility
    std::vector<RobotPoseId>* vertex_matches,
    std::vector<double>* scores) {
  CHECK_NOTNULL(vertex_matches);
  if (db_.find(robot) == db_.end()) {
    LOG(WARNING) << "VLADLoopClosureDetector: No database for robot " << robot << ".";
    return false;
  }

  RobotId robot_query = robot_pose_id.first;
  PoseId pose_query = robot_pose_id.second;
  auto robot_db = db_.at(robot).get();

  CHECK(!global_desc.empty()) << "VLADLoopClosureDetector: Global descriptor for pose "
                              << robot_query << ":" << pose_query << " is empty.";

  if (params_.inter_robot_only_ && robot_query == robot) return false;

  if (pose_query <
      static_cast<PoseId>(params_.local_window_size_ + params_.max_db_results_)) {
    VLOG(1) << "VLADLoopClosureDetector: Not enough frames processed yet. "
            << "Skipping loop closure detection.";
    return false;
  }

  int top_k = params_.max_db_results_;

  Database::Database::QueryResults query_result(top_k, -1);
  Database::Database::QueryDistances query_distance(top_k,
                                                    std::numeric_limits<float>::max());

  int max_search_id = -1;
  size_t num_entries = robot_db->nTotal();
  if (robot_query == robot) {
    if (num_entries >
        static_cast<PoseId>(params_.dist_local_ + params_.local_window_size_)) {
      max_search_id = num_entries - static_cast<PoseId>(params_.dist_local_) -
                      params_.local_window_size_;
    } else {
      return false;
    }
  }

  robot_db->search(global_desc, top_k, query_result, query_distance, max_search_id);

  for (size_t i = 0; i < query_result.size(); ++i) {
    if (query_result[i] == -1) {
      query_result.erase(query_result.begin() + i);
      query_distance.erase(query_distance.begin() + i);
      --i;  // Adjust index after erasure.
    }
  }

  // check if any result's entry id is larger than max_search_id
  for (const auto& id : query_result) {
    if (max_search_id > 0) {
      CHECK_LT(id, max_search_id);
    }
  }

  std::vector<RobotPoseId> query_result_ids;
  for (const auto& id : query_result) {
    query_result_ids.push_back(std::make_pair(robot, db_EntryId_to_PoseId_[robot][id]));
  }

  if (visualizer_) {
    visualizer_->visualizeCandidates(
        "lcd/raw_vlad", robot_pose_id, query_result_ids, query_distance);
  }

  if (query_result.empty()) {
    VLOG(1) << "VLADLoopClosureDetector: No matches found.";
    return false;
  }

  auto faiss_to_dbow_queryresults =
      [&](Database::Database::QueryResults& query_result,
          Database::Database::QueryDistances& query_distance) -> DBoW2::QueryResults {
    DBoW2::QueryResults dbow_query_result;
    for (size_t i = 0; i < query_result.size(); ++i) {
      float score = query_distance[i];
      DBoW2::Result result;
      result.Id = query_result[i];
      result.Score = score;
      dbow_query_result.push_back(result);
    }
    return dbow_query_result;
  };

  VLOG_IF(1, query_result.empty())
      << "VLADLoopClosureDetector: No matches found after applying nss threshold.";

  auto dbow_query_result = faiss_to_dbow_queryresults(query_result, query_distance);

  if (!dbow_query_result.empty()) {
    // Select best result from the raw query results.
    DBoW2::Result best_result = dbow_query_result[0];
    if (best_result.Score < params_.min_sim_vlad) {
      VLOG(1) << "VLADLoopClosureDetector: Best VLAD match below min_sim_vlad.";
      return false;
    }

    const PoseId best_match_pose_id = db_EntryId_to_PoseId_[robot][best_result.Id];

    if (robot != robot_query) {
      LOG(INFO) << "Inter-robot loop closure detected: " << robot_query << ":"
                << pose_query << " <-> " << robot << ":" << best_match_pose_id;
      vertex_matches->push_back(std::make_pair(robot, best_match_pose_id));
      if (scores) scores->push_back(best_result.Score);
    } else {
      // LOG(FATAL) << "intra-robot loop disabled for now";
      // Check dist_local param
      int pose_query_int = (int)pose_query;
      int pose_match_int = (int)best_match_pose_id;
      int pose_distance = std::abs(pose_query_int - pose_match_int);

      LOG(INFO) << "Same-robot match: checking temporal distance=" << pose_distance
                << " vs dist_local=" << params_.dist_local_;

      if (pose_distance < params_.dist_local_) {
        LOG(INFO) << "CONDITION FAILED: Temporal distance too small (" << pose_distance
                  << " < " << params_.dist_local_ << ")";
        return false;
      }

      LOG(INFO) << "Same-robot loop closure detected: " << robot_query << ":"
                << pose_query << " <-> " << robot << ":" << best_match_pose_id;
      vertex_matches->push_back(std::make_pair(robot, best_match_pose_id));
      if (scores) scores->push_back(best_result.Score);
    }
  } else {
    LOG(INFO) << "No valid DBoW query results after conversion.";
  }

  if (scores) CHECK_EQ(vertex_matches->size(), scores->size());

  if (!vertex_matches->empty()) {
    total_global_desc_matches_ += vertex_matches->size();
    return true;
  }

  return false;
}

}  // namespace kimera_multi_lcd