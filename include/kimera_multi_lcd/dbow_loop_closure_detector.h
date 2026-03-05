#include "kimera_multi_lcd/loop_closure_detector.h"

namespace kimera_multi_lcd {
class DBoWLoopClosureDetector
    : public LoopClosureDetector<OrbDatabaseWrapper, cv::ORB, cv::DescriptorMatcher> {
 public:
  DBoWLoopClosureDetector()
      : LoopClosureDetector<OrbDatabaseWrapper, cv::ORB, cv::DescriptorMatcher>() {}
  ~DBoWLoopClosureDetector() = default;

  void addGlobalDesc(const RobotPoseId& id, const GlobalDesc& bow_vector) override {
    const size_t robot_id = id.first;
    const size_t pose_id = id.second;
    // Skip if this BoW vector has been added
    if (globalDescExists(id)) return;
    if (db_.find(robot_id) == db_.end()) {
      db_[robot_id] = createDatabase(0);
      global_descs_[robot_id] = PoseGlobalDesc();
      db_EntryId_to_PoseId_[robot_id] = std::unordered_map<size_t, PoseId>();
      global_desc_latest_pose_id_[robot_id] = pose_id;
      ROS_INFO("Initialized BoW for robot %lu.", robot_id);
    }
    // Add Bow vector to the robot's database
    DBoW2::EntryId entry_id = db_[robot_id]->add(bow_vector);
    // Save the raw bow vectors
    global_descs_[robot_id][pose_id] = bow_vector;
    db_EntryId_to_PoseId_[robot_id][entry_id] = pose_id;
    // Update latest pose ID with BoW
    if (pose_id > global_desc_latest_pose_id_[robot_id]) {
      global_desc_latest_pose_id_[robot_id] = pose_id;
    }
  }
  void loadAndInitialize(const LcdParams& params) override {
    params_ = params;
    lcd_tp_wrapper_ = std::unique_ptr<LcdThirdPartyWrapper>(
        new LcdThirdPartyWrapper(params.lcd_tp_params_));

    // Initiate orb matcher:qa
    feature_matcher_ =
        cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_L1);

    // Initialize bag-of-word database
    vocab_.load(params.vocab_path_);
  }

  inline const OrbVocabulary* getVocabulary() const { return &vocab_; }

  void matchFeatures(const std::vector<cv::Point2f>& query_kpts,
                     const cv::Mat& query_desc,
                     std::vector<cv::Point2f>& train_kpts,
                     const cv::Mat& train_desc,
                     std::vector<DMatchVec>& matches) const override {
    feature_matcher_->knnMatch(query_desc, train_desc, matches, 2u);
  }

  bool detectLoopWithRobot(size_t robot,
                           const RobotPoseId& vertex_query,
                           const GlobalDesc& bow_vector_query,
                           std::vector<RobotPoseId>* vertex_matches,
                           std::vector<double>* scores,
                           uint64_t /*time_since_last_loop*/ = 0) override {
    assert(NULL != vertex_matches);
    vertex_matches->clear();
    if (scores) scores->clear();

    // Return false if specified robot does not exist
    if (db_.find(robot) == db_.end()) return false;
    const OrbDatabase* db = db_.at(robot).get();

    // Extract robot and pose id
    RobotId robot_query = vertex_query.first;
    PoseId pose_query = vertex_query.second;

    // If query and database from same robot
    if (params_.inter_robot_only_ && robot_query == robot) return false;

    // Try to locate BoW of previous frame
    if (pose_query == 0) return false;

    GlobalDesc bow_vec_prev;
    if (!findPreviousGlobalDesc(vertex_query, 5, &bow_vec_prev)) {
      ROS_WARN("Cannot find previous BoW for query vertex (%lu,%lu).",
               robot_query,
               pose_query);
      return false;
    }
    // Compute nss factor with the previous keyframe of the query robot
    double nss_factor = db->getVocabulary()->score(bow_vector_query, bow_vec_prev);
    if (nss_factor < params_.min_nss_factor_) return false;

    // Query similar keyframes based on bow
    DBoW2::QueryResults query_result;
    db->query(bow_vector_query, query_result, params_.max_db_results_);

    // Sort query_result in descending score.
    // This should be done by the query function already,
    // but we do it again in case that behavior changes in the future.
    std::sort(query_result.begin(), query_result.end(), std::greater<DBoW2::Result>());

    // Remove low scores from the QueryResults based on nss.
    DBoW2::QueryResults::iterator query_it =
        lower_bound(query_result.begin(),
                    query_result.end(),
                    DBoW2::Result(0, params_.alpha_ * nss_factor),
                    DBoW2::Result::geq);
    if (query_it != query_result.end()) {
      query_result.resize(query_it - query_result.begin());
    }

    if (!query_result.empty()) {
      DBoW2::Result best_result = query_result[0];
      double normalized_score = best_result.Score / nss_factor;
      const PoseId best_match_pose_id = db_EntryId_to_PoseId_[robot][best_result.Id];
      if (robot != robot_query) {
        vertex_matches->push_back(std::make_pair(robot, best_match_pose_id));
        if (scores) scores->push_back(normalized_score);
      } else {
        // Check dist_local param
        int pose_query_int = (int)pose_query;
        int pose_match_int = (int)best_match_pose_id;
        if (std::abs(pose_query_int - pose_match_int) < params_.dist_local_)
          return false;
        // Compute islands in the matches.
        // An island is a group of matches with close frame_ids.
        std::vector<MatchIsland> islands;
        lcd_tp_wrapper_->computeIslands(&query_result, &islands);
        if (!islands.empty()) {
          // Check for temporal constraint if it is an single robot lc
          // Find the best island grouping using MatchIsland sorting.
          const MatchIsland& best_island =
              *std::max_element(islands.begin(), islands.end());

          // Run temporal constraint check on this best island.
          bool pass_temporal_constraint =
              lcd_tp_wrapper_->checkTemporalConstraint(pose_query, best_island);
          if (pass_temporal_constraint) {
            vertex_matches->push_back(std::make_pair(robot, best_match_pose_id));
            if (scores) scores->push_back(normalized_score);
          }
        }
      }
    }
    if (scores) CHECK_EQ(vertex_matches->size(), scores->size());

    if (!vertex_matches->empty()) {
      total_global_desc_matches_ += vertex_matches->size();
      return true;
    }
    return false;
  }

 private:
  OrbVocabulary vocab_;
};
}  // namespace kimera_multi_lcd