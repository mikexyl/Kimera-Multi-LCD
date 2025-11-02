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

  void add(const GlobalDesc& global_desc) {
    CHECK_NOTNULL(db_);
    CHECK(not global_desc.empty());
    faiss::idx_t id = id_to_desc_map_.size();
    id_to_desc_map_.emplace(id, global_desc.clone());
    try {
      db_->add(global_desc);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to add to database: " << e.what();
      throw;
    }
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
  auto distance(Args&&... args) {
    try {
      return db_->l2_distance(std::forward<Args>(args)...);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to compute distance in database: " << e.what();
      throw;
    }
  }

  GlobalDesc get(const faiss::idx_t id) const {
    if (id_to_desc_map_.count(id)) {
      return id_to_desc_map_.at(id);
    } else {
      return GlobalDesc();  // Return an empty cv::Mat if id not found
    }
  }

 private:
  std::unique_ptr<Database> db_;
  std::map<faiss::idx_t, cv::Mat> id_to_desc_map_;
};

// dummy feature detector that does nothing
class DummyFeatureDetector : cv::FeatureDetector {
 public:
  DummyFeatureDetector() = default;

  CV_WRAP void compute(cv::InputArray image,
                       CV_OUT CV_IN_OUT std::vector<cv::KeyPoint>& keypoints,
                       cv::OutputArray descriptors) final {}

  CV_WRAP void compute(cv::InputArrayOfArrays images,
                       CV_OUT CV_IN_OUT
                           std::vector<std::vector<cv::KeyPoint> >& keypoints,
                       cv::OutputArrayOfArrays descriptors) final {}
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
        env_(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "kimera_multi_lcd")) {
    feature_matcher_ =
        xfeat::LighterGlueCV::create(env_,
                                     xfeat::LighterGlueCV::Params{
                                         .model_path = lcd_params_.lcd_lg_model_path_,
                                         .use_gpu = true,
                                         .min_score = -1,
                                         .n_kpts = lcd_params_.lcd_lg_num_features_,
                                     });
  }

  /* ------------------------------------------------------------------------
   */
  virtual ~VLADLoopClosureDetector() = default;

  virtual std::unique_ptr<Database> createDatabase() override {
    auto faiss_db = std::make_unique<Database::Database>(
        Database::Database::IndexMode::kIVFFlat, lcd_params_.lcd_faiss_index_path_);
    return std::make_unique<Database>(std::move(faiss_db),
                                      env_,
                                      lcd_params_.xfeat_nv_head_model_path_,
                                      lcd_params_.netvlad_model_path_,
                                      kVLADLCDUseGPU);
  }

  // Disambiguate overloaded templated base methods by providing an explicit
  // override that matches the instantiated signature (RobotPoseId, cv::Mat).
  void addGlobalDesc(const RobotPoseId& id,
                     const Database::GlobalDesc& bow_vector) override {
    // Forward to the base implementation to preserve default behavior.
    // LoopClosureDetector<XfeatNVWrapper, DummyFeatureDetector, xfeat::LighterGlueCV>::
    //     addGlobalDesc(id, bow_vector);
  }

  bool detectLoop(const RobotPoseId& frame_id,
                  const Database::GlobalDesc& bow_vec,
                  std::vector<RobotPoseId>* vertex_matches,
                  std::vector<double>* scores = nullptr) {
    throw std::runtime_error("Not implemented");
  }

  bool detectLoopWithRobot(size_t robot,
                           const RobotPoseId& vertex_query,
                           const GlobalDesc& bow_vector_query,
                           std::vector<RobotPoseId>* vertex_matches,
                           std::vector<double>* scores = nullptr) override {}

  void detectLoopOutsideLocalWindow(const RobotPoseId& frame_id,
                                    const Database::GlobalDesc& bow_vec,
                                    std::vector<RobotPoseId>* vertex_matches,
                                    std::vector<double>* scores = nullptr) {}

  std::optional<RobotPoseId> findFirstRobotPoseIdOutsideLocalWindow(
      const RobotPoseId& frame_id) const {
    if (frame_id.second <= static_cast<PoseId>(lcd_params_.local_window_size_)) {
      return std::nullopt;  // No frames outside the local window.
    } else {
      return std::make_pair(frame_id.first,
                            frame_id.second - lcd_params_.local_window_size_);
      // Return the first frame ID outside the local window.
    }
  }

  Ort::Env env_;
};

}  // namespace kimera_multi_lcd