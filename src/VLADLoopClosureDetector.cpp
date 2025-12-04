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
  auto query_frame_outside_local_window =
      this->findFirstRobotPoseIdOutsideLocalWindow(vertex_query);
  if (query_frame_outside_local_window and
      this->detectLoopOutsideLocalWindow(
          robot, *query_frame_outside_local_window, bow_vec, vertex_matches, scores)) {
    return true;
  } else {
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
  CHECK(db_.find(robot_pose_id.first) != db_.end())
      << "VLADLoopClosureDetector: Robot " << robot_pose_id.first
      << " not found in database.";

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

  // int max_possible_match_id =
  //     pose_query - params_.local_window_size_ - params_.max_db_results_;

  // if (max_possible_match_id < 0) {
  //   max_possible_match_id = 0;
  // }
  // if (robot_query != robot) {
  //   max_possible_match_id = -1;
  // }

  int top_k = params_.max_db_results_ + params_.local_window_size_;

  Database::Database::QueryResults query_result(top_k, -1);
  Database::Database::QueryDistances query_distance(top_k,
                                                    std::numeric_limits<float>::max());

  robot_db->search(global_desc, top_k, query_result, query_distance);

  // remove -1 from query_result
  size_t removed_invalid = 0;
  size_t removed_recent = 0;
  for (size_t i = 0; i < query_result.size(); ++i) {
    if (query_result[i] == -1) {
      removed_invalid++;
      query_result.erase(query_result.begin() + i);
      query_distance.erase(query_distance.begin() + i);
      --i;  // Adjust index after erasure.
    }
    // else if (query_result[i] >= max_possible_match_id) {
    //   removed_recent++;
    //   query_result.erase(query_result.begin() + i);
    //   query_distance.erase(query_distance.begin() + i);
    //   --i;  // Adjust index after erasure.
    // }
  }

  std::vector<RobotPoseId> query_result_ids;
  for (const auto& id : query_result) {
    query_result_ids.push_back(std::make_pair(robot, id));
  }

  if (visualizer_) {
    visualizer_->visualizeCandidates(
        "lcd/raw_vlad", robot_pose_id, query_result_ids, query_distance);
  }

  // if the query result has recent frames, throw error
  // for (const auto& id : query_result) {
  //   if (id >= max_possible_match_id) {
  //     throw std::runtime_error(
  //         "VLADLoopClosureDetector: Query result contains recent frames. "
  //         "This should not happen.");
  //   }
  // }

  if (query_result.empty()) {
    VLOG(1) << "VLADLoopClosureDetector: No matches found.";
    return false;
  }

  GlobalDesc prev_global_desc;
  bool found_prev_global_desc =
      findPreviousGlobalDesc(robot_pose_id, 5, &prev_global_desc);
  CHECK(found_prev_global_desc and not prev_global_desc.empty())
      << "VLADLoopClosureDetector: Previous global descriptor for frame " << robot_query
      << ":" << (pose_query - 1) << " is empty.";

  double nss_distance = 0.0;
  if (FLAGS_max_nss_vlad_distance > 0.0) {
    nss_distance = robot_db->distance(global_desc, prev_global_desc);
  } else {
    LOG_IF(ERROR, FLAGS_max_nss_vlad_distance < 0.0)
        << "Setting use_nss as false is deprecated.";
  }

  if (FLAGS_max_nss_vlad_distance > 0.0 && nss_distance > FLAGS_max_nss_vlad_distance) {
    LOG(INFO) << "CONDITION FAILED: NSS distance " << nss_distance
              << " exceeds threshold " << FLAGS_max_nss_vlad_distance;
    VLOG(1) << "VLADLoopClosureDetector: NSS distance " << nss_distance
            << " exceeds threshold " << FLAGS_max_nss_vlad_distance
            << ". No loop closure.";
    return false;
  }

  static constexpr double kL2DistanceToScoreFactor = 10.0;
  float nss_factor = std::exp(-kL2DistanceToScoreFactor * nss_distance);

  auto faiss_to_dbow_queryresults =
      [&](Database::Database::QueryResults& query_result,
          Database::Database::QueryDistances& query_distance) -> DBoW2::QueryResults {
    DBoW2::QueryResults dbow_query_result;
    for (size_t i = 0; i < query_result.size(); ++i) {
      float score = std::exp(-kL2DistanceToScoreFactor * query_distance[i]);
      DBoW2::Result result;
      result.Id = query_result[i];
      result.Score = score;
      dbow_query_result.push_back(result);
    }
    return dbow_query_result;
  };

  // Remove high distances from the QueryResults based on nss.
  double nss_threshold = nss_distance / params_.alpha_;

  size_t removed_by_nss = 0;
  for (size_t i = 0; i < query_result.size(); ++i) {
    if (query_distance[i] > nss_threshold) {
      removed_by_nss++;
      query_result.erase(query_result.begin() + i);
      query_distance.erase(query_distance.begin() + i);
      --i;  // Adjust index after erasure.
    }
  }

  std::vector<RobotPoseId> nss_result_ids;
  std::vector<float> nss_result_distances;
  for (size_t i = 0; i < query_result.size(); ++i) {
    nss_result_ids.push_back(std::make_pair(robot, query_result[i]));
    nss_result_distances.push_back(query_distance[i]);
  }
  if (visualizer_) {
    visualizer_->visualizeCandidates(
        "lcd/nss_filtered_vlad", robot_pose_id, nss_result_ids, nss_result_distances);
  }

  VLOG_IF(1, query_result.empty())
      << "VLADLoopClosureDetector: No matches found after applying nss threshold.";

  auto dbow_query_result = faiss_to_dbow_queryresults(query_result, query_distance);

  if (!dbow_query_result.empty()) {
    // By default use the raw DBow query results. For same-robot matches we will
    // aggregate nearby frames' scores (±kNeighborhood) and use the aggregated
    // results for island computation and best-match selection.
    DBoW2::QueryResults aggregated_dbow_query_result;
    DBoW2::QueryResults* use_results = &dbow_query_result;

    // Aggregate scores in a neighborhood around each matched entry id.
    const int kNeighborhood = 10;  // +/- 10 frames
    const size_t num_entries = db_EntryId_to_PoseId_[robot].size();
    std::vector<float> accum_scores(num_entries, 0.0);
    std::vector<int> neighbor_counts(num_entries, 0);

    for (const auto& res : dbow_query_result) {
      const int id = res.Id;
      const int start = std::max(0, id - kNeighborhood);
      const int end = std::min(static_cast<int>(num_entries) - 1, id + kNeighborhood);
      for (int nid = start; nid <= end; ++nid) {
        accum_scores[nid] += res.Score;
        neighbor_counts[nid] += 1;
      }
    }

    // Only keep aggregated entries that have at least 3 contributing
    // neighboring matches (including the candidate itself).
    const int kMinNeighborMatches = 3;
    for (size_t i = 0; i < num_entries; ++i) {
      if (accum_scores[i] > 0.0 && neighbor_counts[i] >= kMinNeighborMatches) {
        DBoW2::Result r;
        r.Id = static_cast<int>(i);
        r.Score = accum_scores[i];
        aggregated_dbow_query_result.push_back(r);
      }
    }

    // Sort aggregated results by descending score.
    std::sort(aggregated_dbow_query_result.begin(),
              aggregated_dbow_query_result.end(),
              [](const DBoW2::Result& a, const DBoW2::Result& b) {
                return a.Score > b.Score;
              });

    VLOG(2) << "Aggregated " << dbow_query_result.size() << " results into "
            << aggregated_dbow_query_result.size() << " entries using +/-"
            << kNeighborhood << " neighborhood.";

    if (!aggregated_dbow_query_result.empty())
      use_results = &aggregated_dbow_query_result;

    std::vector<RobotPoseId> aggr_result_ids;
    for (const auto& res : *use_results) {
      aggr_result_ids.push_back(std::make_pair(robot, res.Id));
    }
    if (visualizer_) {
      visualizer_->visualizeCandidates(
          "lcd/aggregated_vlad", robot_pose_id, aggr_result_ids, accum_scores);
      visualizer_->visualizeCandidates(
          "lcd/best_vlad", robot_pose_id, {aggr_result_ids.front()}, accum_scores);
    }

    // Select best result from the chosen results (aggregated for same-robot,
    // raw for inter-robot).
    DBoW2::Result best_result = (*use_results)[0];
    double normalized_score = best_result.Score / nss_factor;
    const PoseId best_match_pose_id = db_EntryId_to_PoseId_[robot][best_result.Id];

    if (robot != robot_query) {
      LOG(INFO) << "Inter-robot loop closure detected: " << robot_query << ":"
                << pose_query << " <-> " << robot << ":" << best_match_pose_id;
      vertex_matches->push_back(std::make_pair(robot, best_match_pose_id));
      if (scores) scores->push_back(normalized_score);
    } else {
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
      // Compute islands in the matches.
      // An island is a group of matches with close frame_ids.
      std::vector<MatchIsland> islands;
      // Use aggregated results for island computation if available.
      if (use_results == &dbow_query_result) {
        lcd_tp_wrapper_->computeIslands(&dbow_query_result, &islands);
      } else {
        lcd_tp_wrapper_->computeIslands(&aggregated_dbow_query_result, &islands);
      }

      LOG(INFO) << "Computed islands: count=" << islands.size();

      if (!islands.empty()) {
        // Check for temporal constraint if it is an single robot lc
        // Find the best island grouping using MatchIsland sorting.
        const MatchIsland& best_island =
            *std::max_element(islands.begin(), islands.end());

        LOG(INFO) << "Best island: best_score=" << best_island.best_score_
                  << ", island_score=" << best_island.island_score_
                  << ", size=" << best_island.size() << ", range=["
                  << best_island.start_id_ << "," << best_island.end_id_ << "]";

        // Run temporal constraint check on this best island.
        bool pass_temporal_constraint =
            lcd_tp_wrapper_->checkTemporalConstraint(pose_query, best_island);

        LOG(INFO) << "Temporal constraint check: "
                  << (pass_temporal_constraint ? "PASSED" : "FAILED");

        if (pass_temporal_constraint) {
          LOG(INFO) << "Same-robot loop closure detected: " << robot_query << ":"
                    << pose_query << " <-> " << robot << ":" << best_match_pose_id;
          vertex_matches->push_back(std::make_pair(robot, best_match_pose_id));
          if (scores) scores->push_back(normalized_score);
        } else {
          LOG(INFO) << "CONDITION FAILED: Temporal constraint not satisfied.";
        }
      } else {
        LOG(INFO) << "CONDITION FAILED: No islands computed from matches.";
      }
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