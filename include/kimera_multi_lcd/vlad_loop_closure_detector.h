#pragma once

#include <xfeat-cpp/faiss_database.h>
#include <xfeat-cpp/lighterglue_cv.h>
#include <xfeat-cpp/xfeat_cv.h>
#include <xfeat-cpp/xfeat_netvlad_onnx.h>

#include "kimera_multi_lcd/loop_closure_detector.h"

namespace kimera_multi_lcd {

struct XfeatNVWrapper : xfeat::XfeatNetVLADONNX {
  using Base = xfeat::XfeatNetVLADONNX;
  using GlobalDesc = cv::Mat;
  using Desc = cv::Mat;
  using DescVector = std::vector<cv::Mat>;
  using DescMat = cv::Mat;
  using Database = xfeat::FaissDatabase;

  template <typename... Args>
  XfeatNVWrapper(std::unique_ptr<Database> faiss_db, Args&&... args)
      : Base(std::forward<Args>(args)...), db_(std::move(faiss_db)) {}

  void transform(const DescVector& desc_vec, GlobalDesc& global_desc) {
    CHECK(desc_vec.size() == 2) << "XfeatNVWrapper: the feature vector must be "
                                   "the vector of [M1, x_prep]";

    auto M1 = desc_vec[0];
    auto x_prep = desc_vec[1];

    if (M1.empty()) throw std::runtime_error("XfeatNVWrapper: M1 is empty");
    if (x_prep.empty()) throw std::runtime_error("XfeatNVWrapper: x_prep is empty");

    if (M1.type() != CV_32F || x_prep.type() != CV_32F) {
      throw std::runtime_error("XfeatNVWrapper: M1 and x_prep must be of type CV_32F");
    }

    global_desc = Base::transform(M1, x_prep);
  }

  auto add(const GlobalDesc& global_desc) {
    CHECK_NOTNULL(db_);
    CHECK(not global_desc.empty());
    try {
      return db_->add(global_desc);
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

class VLADLoopClosureDetector : public LoopClosureDetector<XfeatNVWrapper,
                                                           DummyFeatureDetector,
                                                           xfeat::LighterGlueCV> {
 public:
  using Database = XfeatNVWrapper;
  using BaseDetector =
      LoopClosureDetector<XfeatNVWrapper, DummyFeatureDetector, xfeat::LighterGlueCV>;

  static constexpr bool kVLADLCDUseGPU = true;

  template <typename... Args>
  VLADLoopClosureDetector(Args&&... args)
      : BaseDetector(std::forward<Args>(args)...),
        env_(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "kimera_multi_lcd")) {}

  /* ------------------------------------------------------------------------
   */
  virtual ~VLADLoopClosureDetector() = default;

  void loadAndInitialize(const LcdParams& params) override {
    LoopClosureDetectorBase::loadAndInitialize(params);

    LOG(INFO) << "dist_local: " << params_.dist_local_;

    lcd_tp_wrapper_ = std::unique_ptr<LcdThirdPartyWrapper>(
        new LcdThirdPartyWrapper(params.lcd_tp_params_));

    LOG(INFO) << "load lg from: " << params_.lcd_lg_model_path_;
    LOG(INFO) << "load faiss from: " << params_.lcd_faiss_index_path_;
    feature_matcher_ = xfeat::LighterGlueCV::create(
        env_,
        xfeat::LighterGlueCV::Params{
            .model_path = params_.lcd_lg_model_path_,
            .use_gpu = true,
            .min_score = -1,
            .n_kpts = params_.lcd_lg_num_features_,
            // TODO(mike): add params to input real images's size
            .image_size = cv::Size(params_.image_width_, params_.image_height_)});
    LOG(INFO) << "VLADLoopClosureDetector initialized.";
  }

  virtual std::unique_ptr<Database> createDatabase() override {
    LOG(INFO) << "Creating FAISS database";
    auto faiss_mode = Database::Database::IndexMode::kIVFFlat;
    int faiss_dim = 0;
    if (params_.lcd_faiss_index_path_.empty()) {
      faiss_mode = Database::Database::IndexMode::kFlat;
      faiss_dim = 512;
    }
    auto faiss_db = std::make_unique<Database::Database>(
        faiss_mode, params_.lcd_faiss_index_path_, false, faiss_dim);
    // faiss_db.
    return std::make_unique<Database>(std::move(faiss_db),
                                      env_,
                                      params_.xfeat_nv_head_model_path_,
                                      params_.netvlad_model_path_,
                                      kVLADLCDUseGPU,
                                      params_.network_input_height_ / 16,
                                      params_.network_input_width_ / 16);
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

    auto lg_matcher = std::dynamic_pointer_cast<xfeat::LighterGlueCV>(feature_matcher_);
    CHECK(lg_matcher.get() != nullptr)
        << "VLADLoopClosureDetector: feature_matcher_ is not LighterGlueCV";

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

    DMatchVec lg_matches;
    LOG(INFO) << "Matching " << query_det.keypoints.rows << " query keypoints with "
              << train_det.keypoints.rows << " train keypoints.";
    lg_matcher->match(query_det, train_det, lg_matches);

    // compute homography
    LOG(INFO) << "Found " << lg_matches.size() << " matches.";

    matches.clear();
    for (const auto& match : lg_matches) {
      matches.push_back(DMatchVec(1, match));
    }

    // // cv::findHomography()
    // std::vector<cv::Point2f> query_matched_kpts, train_matched_kpts;
    // for (const auto& match : lg_matches) {
    //   query_matched_kpts.push_back(query_kpts[match.queryIdx]);
    //   train_matched_kpts.push_back(train_kpts[match.trainIdx]);
    // }

    // cv::Mat inlier_mask;
    // cv::Mat H_lg;
    // if (lg_matches.size() >= 4) {
    //   H_lg = cv::findHomography(
    //       query_matched_kpts, train_matched_kpts, cv::RANSAC, 5, inlier_mask);

    //   if (H_lg.empty() or cv::countNonZero(inlier_mask) < 4) {
    //     matches.clear();
    //     return;
    //   }

    //   double detH = cv::determinant(H_lg);
    //   double normH = cv::norm(H_lg);

    //   LOG(INFO) << "det(H) = " << detH << ", norm(H) = " << normH;

    //   if (std::abs(detH) < 1e-6 || normH > 1e6) {
    //     matches.clear();
    //     return;
    //   }

    //   // 1. Collect ALL query points (not only matched ones)
    //   std::vector<cv::Point2f> query_pts;
    //   query_pts.reserve(query_kpts.size());
    //   for (const auto& p : query_kpts) {
    //     query_pts.push_back(p);  // or p.pt if query_kpts is KeyPoint
    //   }

    //   // 2. Project them using H_lg
    //   std::vector<cv::Point2f> query_proj_pts;
    //   cv::perspectiveTransform(query_pts, query_proj_pts, H_lg);

    //   // 3. Build full mask: rows = all query desc, cols = all train desc
    //   cv::Mat search_mask = cv::Mat::zeros(query_desc.rows, train_desc.rows, CV_8U);

    //   float max_pixel_error_ratio = 0.05f;
    //   float max_pixel_error =
    //       max_pixel_error_ratio * std::max(params_.image_width_,
    //       params_.image_height_);

    //   for (int qi = 0; qi < query_desc.rows; ++qi) {
    //     const cv::Point2f& p_pred = query_proj_pts[qi];

    //     for (int ti = 0; ti < train_desc.rows; ++ti) {
    //       const cv::Point2f& p2 = train_kpts[ti];
    //       float dx = std::abs(p2.x - p_pred.x);
    //       float dy = std::abs(p2.y - p_pred.y);

    //       if (dx < max_pixel_error && dy < max_pixel_error) {
    //         search_mask.at<uchar>(qi, ti) = 1;
    //       }
    //     }
    //   }

    //   cv::BFMatcher bf_matcher(cv::NORM_L2, /*crossCheck=*/false);
    //   std::vector<std::vector<cv::DMatch>> knn_matches;
    //   lg_matches.clear();
    //   CHECK(bf_matcher.isMaskSupported());
    //   bf_matcher.knnMatch(query_desc, train_desc, knn_matches, 1, search_mask);

    //   // apply a distance threshold to filter matches
    //   const float max_desc_distance = 0.7f;
    //   std::vector<DMatchVec> filtered_matches(knn_matches.size());
    //   for (const auto& knn_match : knn_matches) {
    //     for (const auto& m : knn_match) {
    //       CHECK(search_mask.at<uchar>(m.queryIdx, m.trainIdx) == 1);
    //       if (m.distance < max_desc_distance) {
    //         filtered_matches[m.queryIdx].push_back(m);
    //       }
    //     }
    //   }
    //   matches = filtered_matches;
    //   return;
    // }
    // // convert to knn result format
    // matches.clear();
    // for (const auto& match : lg_matches) {
    //   matches.push_back(DMatchVec(1, match));
    // }
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
      db_[robot_id] = createDatabase();
      global_descs_[robot_id] = PoseGlobalDesc();
      db_EntryId_to_PoseId_[robot_id] = std::unordered_map<size_t, PoseId>();
      global_desc_latest_pose_id_[robot_id] = pose_id;
      ROS_INFO("Initialized BoW for robot %lu.", robot_id);
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
                           std::vector<double>* scores = nullptr) override;

  bool detectLoopOutsideLocalWindow(size_t robot,
                                    const RobotPoseId& frame_id,
                                    const Database::GlobalDesc& bow_vec,
                                    std::vector<RobotPoseId>* vertex_matches,
                                    std::vector<double>* scores = nullptr);

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

  Ort::Env env_;
};

}  // namespace kimera_multi_lcd