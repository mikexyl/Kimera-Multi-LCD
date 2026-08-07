#include <gtest/gtest.h>

#include <limits>

#include "kimera_multi_lcd/orbslam3_sim3_estimator.h"
#include "kimera_multi_lcd/sim3_utils.h"

namespace kimera_multi_lcd {
namespace {

using PointVector = std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>>;

struct SyntheticData {
  PointVector source;
  PointVector destination;
  std::vector<std::pair<unsigned int, unsigned int>> pairs;
  gtsam::Rot3 rotation = gtsam::Rot3::RzRyRx(0.08, -0.06, 0.12);
  gtsam::Point3 translation{0.25, -0.15, 0.45};
  double scale = 1.18;
};

SyntheticData makeSyntheticData() {
  SyntheticData data;
  for (unsigned int index = 0; index < 40; ++index) {
    const double x = -0.8 + 0.13 * (index % 11);
    const double y = -0.5 + 0.11 * (index % 9) + 0.005 * index;
    const double z = 2.0 + 0.09 * (index % 7) + 0.01 * index;
    const gtsam::Point3 source(x, y, z);
    gtsam::Point3 destination =
        data.scale * (data.rotation * source) + data.translation;
    if (index % 7 == 0) {
      destination += gtsam::Point3(1.5 + 0.1 * index, -1.2, 0.8);
    }
    data.source.push_back(source);
    data.destination.push_back(destination);
    data.pairs.emplace_back(100 + index, 200 + index);
  }
  return data;
}

Orbslam3Sim3Params testParams() {
  Orbslam3Sim3Params params;
  params.reprojection_threshold_px = 2.0;
  params.average_focal_length_px = 900.0;
  params.min_scale = 0.5;
  params.max_scale = 2.0;
  params.min_inlier_count = 20;
  params.min_inlier_percentage = 0.5;
  params.max_ransac_iterations = 1000;
  return params;
}

}  // namespace

TEST(Orbslam3Sim3EstimatorTest, RecoversDestinationFromSourceAndMapsDescriptorInliers) {
  const SyntheticData data = makeSyntheticData();
  const auto estimate =
      estimateOrbslam3Sim3(data.source, data.destination, data.pairs, testParams());
  ASSERT_TRUE(estimate.success);
  EXPECT_EQ(estimate.valid_pair_count, data.source.size());
  EXPECT_NEAR(estimate.destination_T_source.scale(), data.scale, 1e-4);
  EXPECT_TRUE(estimate.destination_T_source.rotation().equals(data.rotation, 1e-4));
  EXPECT_TRUE(physicalPose(estimate.destination_T_source)
                  .translation()
                  .isApprox(data.translation, 1e-4));
  ASSERT_GE(estimate.descriptor_inlier_pairs.size(), 20u);
  for (const auto& pair : estimate.descriptor_inlier_pairs) {
    ASSERT_GE(pair.first, 100u);
    const unsigned int index = pair.first - 100u;
    EXPECT_EQ(pair.second, 200u + index);
    EXPECT_NE(index % 7, 0u);
  }
}

TEST(Orbslam3Sim3EstimatorTest, RejectsInvalidDepthAndInsufficientPairs) {
  PointVector source{gtsam::Point3(1.0, 0.0, 2.0),
                     gtsam::Point3::Zero(),
                     gtsam::Point3(0.0, 1.0, 2.0),
                     gtsam::Point3(0.0, 0.0, -1.0),
                     gtsam::Point3(std::numeric_limits<double>::quiet_NaN(), 0.0, 2.0)};
  PointVector destination = source;
  std::vector<std::pair<unsigned int, unsigned int>> pairs{
      {10, 20}, {11, 21}, {12, 22}, {13, 23}, {14, 24}};
  auto params = testParams();
  params.min_inlier_count = 3;
  const auto estimate = estimateOrbslam3Sim3(source, destination, pairs, params);
  EXPECT_FALSE(estimate.success);
  EXPECT_EQ(estimate.valid_pair_count, 2u);
}

TEST(Orbslam3Sim3EstimatorTest, EnforcesScaleBounds) {
  const SyntheticData data = makeSyntheticData();
  auto params = testParams();
  params.max_scale = 1.1;
  const auto estimate =
      estimateOrbslam3Sim3(data.source, data.destination, data.pairs, params);
  EXPECT_FALSE(estimate.success);
  EXPECT_EQ(estimate.valid_pair_count, data.source.size());
}

TEST(Orbslam3Sim3EstimatorTest, RelaxedReprojectionThresholdAcceptsNoisyPairs) {
  SyntheticData data = makeSyntheticData();
  for (size_t index = 0; index < data.destination.size(); ++index) {
    if (index % 7 != 0) {
      const double sign = index % 2 == 0 ? 1.0 : -1.0;
      data.destination[index] +=
          gtsam::Point3(sign * (0.012 + 0.001 * (index % 5)),
                        -sign * (0.010 + 0.001 * (index % 3)),
                        0.0);
    }
  }

  auto tight_params = testParams();
  tight_params.reprojection_threshold_px = 3.0;
  tight_params.min_inlier_count = 25;
  const auto tight = estimateOrbslam3Sim3(
      data.source, data.destination, data.pairs, tight_params);
  EXPECT_FALSE(tight.success);

  auto relaxed_params = tight_params;
  relaxed_params.reprojection_threshold_px = 15.0;
  const auto relaxed = estimateOrbslam3Sim3(
      data.source, data.destination, data.pairs, relaxed_params);
  ASSERT_TRUE(relaxed.success);
  EXPECT_GE(relaxed.descriptor_inlier_pairs.size(), 25u);
  EXPECT_NEAR(relaxed.destination_T_source.scale(), data.scale, 0.03);
}

TEST(Orbslam3Sim3EstimatorTest, RejectsInvalidConfigurationAndShapeMismatch) {
  const SyntheticData data = makeSyntheticData();
  auto params = testParams();
  params.max_ransac_iterations = 0;
  EXPECT_FALSE(
      estimateOrbslam3Sim3(data.source, data.destination, data.pairs, params).success);

  params = testParams();
  auto pairs = data.pairs;
  pairs.pop_back();
  EXPECT_FALSE(
      estimateOrbslam3Sim3(data.source, data.destination, pairs, params).success);
}

}  // namespace kimera_multi_lcd
