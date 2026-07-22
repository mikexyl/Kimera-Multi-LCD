#pragma once

#include <xfeat-cpp/faiss_database.h>
#include <xfeat-cpp/lighterglue_trt.h>

#include "kimera_multi_lcd/loop_closure_detector.h"

namespace kimera_multi_lcd {

struct GlobalDescWithScores {
  cv::Mat descriptor;
  std::vector<float> scores;
};

struct FaissWrapper {
  using GlobalDesc = GlobalDescWithScores;
  using Desc = cv::Mat;
  using DescVector = std::vector<cv::Mat>;
  using DescMat = cv::Mat;
  using Database = xfeat::FaissDatabase;

  template <typename... Args>
  FaissWrapper(std::unique_ptr<Database> faiss_db) : db_(std::move(faiss_db)) {}

  void transform(const DescVector& desc_vec, GlobalDesc& global_desc) {
    LOG(FATAL) << "shouldn't be called!";
  }

  auto add(const GlobalDesc& global_desc) {
    CHECK_NOTNULL(db_);
    CHECK(not global_desc.descriptor.empty());
    try {
      return db_->add(global_desc.descriptor);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to add to database: " << e.what();
      throw;
    }
  }

  auto nTotal() const {
    CHECK_NOTNULL(db_);
    return db_->nTotal();
  }

  template <typename... Args>
  void search(Args&&... args) {
    CHECK_NOTNULL(db_);
    try {
      db_->search(std::forward<Args>(args)...);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to search in database: " << e.what();
      throw;
    }
  }

  template <typename... Args>
  auto sim(Args&&... args) {
    try {
      return db_->cosine_similarity(std::forward<Args>(args)...);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to compute similarity in database: " << e.what();
      throw;
    }
  }

  // GlobalDesc get(const size_t id) const { return db_->get(id); }

 private:
  std::unique_ptr<Database> db_;
};

// dummy feature detector that does nothing
class DummyFeatureDetector : cv::FeatureDetector {
 public:
  DummyFeatureDetector() = default;

  CV_WRAP void compute(cv::InputArray,
                       CV_OUT CV_IN_OUT std::vector<cv::KeyPoint>&,
                       cv::OutputArray) final {}

  CV_WRAP void compute(cv::InputArrayOfArrays,
                       CV_OUT CV_IN_OUT std::vector<std::vector<cv::KeyPoint>>&,
                       cv::OutputArrayOfArrays) final {}
};

class VLADLoopClosureDetector : public LoopClosureDetector<FaissWrapper,
                                                           DummyFeatureDetector,
                                                           xfeat::LighterGlueTRT> {
 public:
  using Database = FaissWrapper;
  using BaseDetector =
      LoopClosureDetector<FaissWrapper, DummyFeatureDetector, xfeat::LighterGlueTRT>;

  static constexpr bool kVLADLCDUseGPU = true;

  template <typename... Args>
  VLADLoopClosureDetector(Args&&... args)
      : BaseDetector(std::forward<Args>(args)...) {}

  /* ------------------------------------------------------------------------
   */
  virtual ~VLADLoopClosureDetector() = default;

  void loadAndInitialize(const LcdParams& params) override {
    LoopClosureDetectorBase::loadAndInitialize(params);

    LOG(INFO) << "dist_local: " << params_.dist_local_;

    lcd_tp_wrapper_ = std::unique_ptr<LcdThirdPartyWrapper>(
        new LcdThirdPartyWrapper(params.lcd_tp_params_));

    LOG(INFO) << "load lg from: " << params_.lcd_lg_model_path_;
    feature_matcher_ =
        cv::makePtr<xfeat::LighterGlueTRT>(params_.lcd_lg_model_path_);
    LOG(INFO) << "Using native TensorRT LighterGlue for distributed loop "
                 "verification.";
    LOG(INFO) << "VLADLoopClosureDetector initialized.";
  }

  virtual std::unique_ptr<Database> createDatabase(int dim) override {
    LOG(INFO) << "Creating CPU flat FAISS database with dim=" << dim;
    auto faiss_db = std::make_unique<Database::Database>(dim);
    return std::make_unique<Database>(std::move(faiss_db));
  }

  void matchFeatures(const std::vector<cv::Point2f>& query_kpts,
                     const cv::Mat& query_desc,
                     std::vector<cv::Point2f>& train_kpts,
                     const cv::Mat& train_desc,
                     std::vector<DMatchVec>& matches) const override {
    VLOG(1) << "VLADLoopClosureDetector: Matching features between query and match.";
    // check any of the descriptors or keypoints is empty
    CHECK(!query_desc.empty()) << "VLADLoopClosureDetector: query_desc is empty";
    CHECK(!train_desc.empty()) << "VLADLoopClosureDetector: train_desc is empty";
    CHECK(!query_kpts.empty()) << "VLADLoopClosureDetector: query_kpts is empty";
    CHECK(!train_kpts.empty()) << "VLADLoopClosureDetector: train_kpts is empty";

    auto lg_matcher =
        std::dynamic_pointer_cast<xfeat::LighterGlueTRT>(feature_matcher_);
    CHECK(lg_matcher.get() != nullptr)
        << "VLADLoopClosureDetector: feature_matcher_ is not LighterGlueTRT";

    CHECK_EQ(query_kpts.size(), (size_t)query_desc.rows);
    CHECK_EQ(train_kpts.size(), (size_t)train_desc.rows);

    xfeat::DetectionResult query_det;
    query_det.keypoints = cv::Mat(query_kpts).reshape(1);
    query_det.descriptors = query_desc;
    // set all scores to 1
    query_det.scores = cv::Mat::ones(query_det.keypoints.rows, 1, CV_32F);

    CHECK_EQ(query_det.keypoints.rows, query_desc.rows)
        << "VLADLoopClosureDetector: query keypoints and descriptors size mismatch";

    xfeat::DetectionResult train_det;
    train_det.keypoints = cv::Mat(train_kpts).reshape(1);
    train_det.descriptors = train_desc;
    CHECK_EQ(train_det.keypoints.rows, train_desc.rows)
        << "VLADLoopClosureDetector: train keypoints and descriptors size mismatch";
    // set all scores to 1
    train_det.scores = cv::Mat::ones(train_det.keypoints.rows, 1, CV_32F);

    LOG(INFO) << "Matching " << query_det.keypoints.rows << " query keypoints with "
              << train_det.keypoints.rows << " train keypoints.";

    const std::array<float, 2> image_size = {
        static_cast<float>(params_.image_width_),
        static_cast<float>(params_.image_height_)};
    std::vector<float> match_scores;
    const auto match_indices = lg_matcher->match(query_det,
                                                 image_size,
                                                 train_det,
                                                 image_size,
                                                 -1.0f,
                                                 &match_scores);

    DMatchVec lg_matches;
    for (size_t query_index = 0; query_index < match_indices.size();
         ++query_index) {
      for (const int train_index : match_indices.at(query_index)) {
        lg_matches.emplace_back(static_cast<int>(query_index),
                                train_index,
                                0,
                                match_scores.at(query_index));
      }
    }

    // compute homography
    LOG(INFO) << "Found " << lg_matches.size() << " matches.";

    matches.clear();
    for (const auto& match : lg_matches) {
      matches.push_back(DMatchVec(1, match));
    }
  }

  // Disambiguate overloaded templated base methods by providing an explicit
  // override that matches the instantiated signature (RobotPoseId, cv::Mat).
  void addGlobalDesc(const RobotPoseId& id,
                     const Database::GlobalDesc& bow_vector) override {
    // time this function
    const size_t robot_id = id.first;
    const size_t pose_id = id.second;
    // Skip if this BoW vector has been added
    if (globalDescExists(id)) return;
    if (db_.find(robot_id) == db_.end()) {
      db_[robot_id] = createDatabase(bow_vector.descriptor.cols);
      global_descs_[robot_id] = PoseGlobalDesc();
      db_EntryId_to_PoseId_[robot_id] = std::unordered_map<size_t, PoseId>();
      global_desc_latest_pose_id_[robot_id] = pose_id;
      LOG(INFO) << "Initialized BoW for robot " << robot_id << ".";
    }
    // Add Bow vector to the robot's database
    // time this
    faiss::idx_t entry_id_faiss = db_[robot_id]->add(bow_vector);

    size_t entry_id = static_cast<size_t>(entry_id_faiss);
    // Save the raw bow vectors
    global_descs_[robot_id][pose_id] = bow_vector;
    db_EntryId_to_PoseId_[robot_id][entry_id] = pose_id;
    // Update latest pose ID with BoW
    if (pose_id > global_desc_latest_pose_id_[robot_id]) {
      global_desc_latest_pose_id_[robot_id] = pose_id;
    }
  }

  bool detectLoopWithRobot(size_t robot,
                           const RobotPoseId& vertex_query,
                           const GlobalDesc& bow_vector_query,
                           std::vector<RobotPoseId>* vertex_matches,
                           std::vector<double>* scores = nullptr,
                           uint64_t time_since_last_loop = 0) override;

  bool detectLoopOutsideLocalWindow(size_t robot,
                                    const RobotPoseId& frame_id,
                                    const Database::GlobalDesc& bow_vec,
                                    std::vector<RobotPoseId>* vertex_matches,
                                    std::vector<double>* scores = nullptr,
                                    uint64_t time_since_last_loop = 0);

  std::optional<RobotPoseId> findFirstRobotPoseIdOutsideLocalWindow(
      const RobotPoseId& frame_id) const {
    if (frame_id.second < static_cast<PoseId>(params_.local_window_size_)) {
      return std::nullopt;  // No frames outside the local window.
    } else {
      return std::make_pair(frame_id.first,
                            frame_id.second - params_.local_window_size_);
      // Return the first frame ID outside the local window.
    }
  }
};

}  // namespace kimera_multi_lcd
