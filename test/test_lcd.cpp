#include <gtest/gtest.h>

#include <filesystem>
#include <random>

#include "kimera_multi_lcd/dbow_loop_closure_detector.h"

namespace kimera_multi_lcd {
namespace {

std::string createTestVocabulary() {
  const auto path =
      std::filesystem::temp_directory_path() / "kimera_multi_lcd_test_vocab.yml";
  std::vector<std::vector<cv::Mat>> training_features;
  std::mt19937 generator(7);
  std::uniform_int_distribution<int> distribution(0, 255);
  for (size_t image = 0; image < 8; ++image) {
    std::vector<cv::Mat> descriptors;
    for (size_t row = 0; row < 12; ++row) {
      cv::Mat descriptor(1, 32, CV_8U);
      for (int col = 0; col < descriptor.cols; ++col) {
        descriptor.at<uint8_t>(0, col) =
            static_cast<uint8_t>(distribution(generator));
      }
      descriptors.push_back(descriptor);
    }
    training_features.push_back(std::move(descriptors));
  }
  OrbVocabulary vocabulary(3, 2, DBoW2::TF_IDF, DBoW2::L1_NORM);
  vocabulary.create(training_features);
  vocabulary.save(path.string());
  return path.string();
}

LcdParams testParams() {
  LcdParams params;
  params.vocab_path_ = createTestVocabulary();
  params.inter_robot_only_ = false;
  params.alpha_ = 0.1;
  params.dist_local_ = 3;
  params.max_db_results_ = 5;
  params.min_nss_factor_ = 0.0;
  params.lcd_tp_params_.max_nrFrames_between_queries_ = 2;
  params.lcd_tp_params_.max_nrFrames_between_islands_ = 3;
  params.lcd_tp_params_.min_temporal_matches_ = 0;
  params.lcd_tp_params_.max_intraisland_gap_ = 3;
  params.lcd_tp_params_.min_matches_per_island_ = 1;
  return params;
}

DBoW2::BowVector descriptorToBow(const OrbVocabulary& vocabulary) {
  std::vector<cv::Mat> descriptors;
  for (size_t row = 0; row < 10; ++row) {
    descriptors.push_back(cv::Mat::ones(1, 32, CV_8U) * (row + 1));
  }
  DBoW2::BowVector bow;
  vocabulary.transform(descriptors, bow);
  return bow;
}

}  // namespace

TEST(LcdTest, LoadsGeneratedVocabulary) {
  DBoWLoopClosureDetector detector;
  const auto params = testParams();
  detector.loadAndInitialize(params);
  EXPECT_FALSE(detector.getVocabulary()->empty());
  EXPECT_EQ(detector.getParams(), params);
}

TEST(LcdTest, GlobalDescriptorsAreIdempotentAndQueryable) {
  DBoWLoopClosureDetector detector;
  detector.loadAndInitialize(testParams());
  const auto bow = descriptorToBow(*detector.getVocabulary());
  detector.addGlobalDesc({0, 4}, bow);
  detector.addGlobalDesc({0, 4}, bow);
  EXPECT_TRUE(detector.globalDescExists({0, 4}));
  EXPECT_EQ(detector.numGlobalDescsForRobot(0), 1);
  EXPECT_EQ(detector.latestPoseIdWithGlobalDesc(0), 4);
  EXPECT_EQ(detector.getGlobalDesc({0, 4}), bow);
}

TEST(LcdTest, DetectsInterRobotDescriptorMatch) {
  DBoWLoopClosureDetector detector;
  auto params = testParams();
  params.inter_robot_only_ = true;
  detector.loadAndInitialize(params);
  const auto bow = descriptorToBow(*detector.getVocabulary());
  detector.addGlobalDesc({0, 0}, bow);
  detector.addGlobalDesc({1, 0}, bow);
  std::vector<RobotPoseId> matches;
  std::vector<double> scores;
  EXPECT_TRUE(detector.detectLoopWithRobot(
      0, {1, 1}, bow, &matches, &scores));
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches.front(), RobotPoseId(0, 0));
  EXPECT_EQ(scores.size(), matches.size());
}

}  // namespace kimera_multi_lcd
