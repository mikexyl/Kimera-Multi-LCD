#include <gtest/gtest.h>

#include <limits>

#include "kimera_multi_lcd/sim3_utils.h"
#include "kimera_multi_lcd/teaser_sim3_estimator.h"

namespace kimera_multi_lcd {
namespace {

using PointVector =
    std::vector<gtsam::Point3, Eigen::aligned_allocator<gtsam::Point3>>;

struct SyntheticData {
  PointVector source;
  PointVector destination;
  std::vector<std::pair<unsigned int, unsigned int>> pairs;
  gtsam::Rot3 rotation;
  gtsam::Point3 translation;
  double scale = 1.25;
};

SyntheticData makeSyntheticData() {
  SyntheticData data;
  data.rotation = gtsam::Rot3::RzRyRx(0.18, -0.12, 0.27);
  data.translation = gtsam::Point3(0.7, -0.4, 1.1);
  for (unsigned int index = 0; index < 32; ++index) {
    const double x = 0.2 + 0.11 * index;
    const double y = -0.7 + 0.17 * (index % 7) + 0.013 * index;
    const double z = 1.0 + 0.09 * (index % 5) + 0.007 * index * index;
    const gtsam::Point3 source(x, y, z);
    gtsam::Point3 destination =
        data.scale * (data.rotation * source) + data.translation;
    if (index % 6 == 0) {
      destination += gtsam::Point3(4.0 + index, -3.0, 2.5);
    }
    data.source.push_back(source);
    data.destination.push_back(destination);
    data.pairs.emplace_back(100 + index, 200 + index);
  }
  return data;
}

TeaserSim3Params testParams() {
  TeaserSim3Params params;
  params.noise_bound_m = 0.02;
  params.min_scale = 0.5;
  params.max_scale = 2.0;
  params.min_inlier_count = 16;
  params.min_inlier_percentage = 0.5;
  return params;
}

}  // namespace

TEST(TeaserSim3EstimatorTest, RecoversDestinationFromSourceAndMapsInliers) {
  const SyntheticData data = makeSyntheticData();
  const auto estimate =
      estimateTeaserSim3(data.source, data.destination, data.pairs, testParams());
  ASSERT_TRUE(estimate.success);
  EXPECT_EQ(estimate.valid_pair_count, data.source.size());
  EXPECT_NEAR(estimate.destination_T_source.scale(), data.scale, 1e-6);
  EXPECT_TRUE(estimate.destination_T_source.rotation().equals(data.rotation, 1e-6));
  EXPECT_TRUE(physicalPose(estimate.destination_T_source)
                  .translation()
                  .isApprox(data.translation, 1e-6));
  ASSERT_GE(estimate.descriptor_inlier_pairs.size(), 16u);
  for (const auto& pair : estimate.descriptor_inlier_pairs) {
    ASSERT_GE(pair.first, 100u);
    const unsigned int index = pair.first - 100u;
    EXPECT_EQ(pair.second, 200u + index);
    EXPECT_NE(index % 6, 0u);
  }
}

TEST(TeaserSim3EstimatorTest, RejectsInvalidDepthAndInsufficientPairs) {
  PointVector source{gtsam::Point3(1.0, 0.0, 1.0),
                     gtsam::Point3::Zero(),
                     gtsam::Point3(0.0, 1.0, 1.0),
                     gtsam::Point3(std::numeric_limits<double>::quiet_NaN(),
                                   0.0,
                                   1.0)};
  PointVector destination = source;
  std::vector<std::pair<unsigned int, unsigned int>> pairs{
      {10, 20}, {11, 21}, {12, 22}, {13, 23}};
  auto params = testParams();
  params.min_inlier_count = 3;
  const auto estimate =
      estimateTeaserSim3(source, destination, pairs, params);
  EXPECT_FALSE(estimate.success);
  EXPECT_EQ(estimate.valid_pair_count, 2u);
}

TEST(TeaserSim3EstimatorTest, EnforcesScaleBounds) {
  const SyntheticData data = makeSyntheticData();
  auto params = testParams();
  params.max_scale = 1.1;
  const auto estimate =
      estimateTeaserSim3(data.source, data.destination, data.pairs, params);
  EXPECT_FALSE(estimate.success);
  EXPECT_EQ(estimate.valid_pair_count, data.source.size());
}

TEST(Sim3UtilsTest, StoresPhysicalTranslationUsingGtsamConvention) {
  const gtsam::Rot3 rotation = gtsam::Rot3::Ry(0.3);
  const gtsam::Point3 physical_translation(1.0, -2.0, 3.0);
  const auto similarity =
      similarityFromPhysical(rotation, physical_translation, 1.4);
  EXPECT_TRUE(similarity.translation().isApprox(physical_translation / 1.4));
  EXPECT_TRUE(physicalPose(similarity).translation().isApprox(physical_translation));
}

TEST(Sim3UtilsTest, CameraAndSubmapCompositionPreservePhysicalAction) {
  const gtsam::Pose3 qbody_T_qcam(
      gtsam::Rot3::Rz(0.2), gtsam::Point3(0.3, -0.1, 0.2));
  const gtsam::Pose3 mbody_T_mcam(
      gtsam::Rot3::Ry(-0.15), gtsam::Point3(-0.2, 0.4, 0.1));
  const auto qcam_T_mcam = similarityFromPhysical(
      gtsam::Rot3::Rx(0.1), gtsam::Point3(1.0, 2.0, -0.5), 1.3);
  const auto qbody_T_mbody = similarityFromPose(qbody_T_qcam) *
                             qcam_T_mcam *
                             similarityFromPose(mbody_T_mcam).inverse();

  const gtsam::Point3 point_mbody(0.6, -0.2, 1.1);
  const gtsam::Point3 point_mcam =
      mbody_T_mcam.inverse().transformFrom(point_mbody);
  const gtsam::Point3 point_qcam = qcam_T_mcam.transformFrom(point_mcam);
  const gtsam::Point3 expected_qbody =
      qbody_T_qcam.transformFrom(point_qcam);
  EXPECT_TRUE(qbody_T_mbody.transformFrom(point_mbody)
                  .isApprox(expected_qbody, 1e-9));

  const gtsam::Pose3 submap1_T_qbody(
      gtsam::Rot3::Ry(0.3), gtsam::Point3(2.0, 0.0, -1.0));
  const gtsam::Pose3 submap2_T_mbody(
      gtsam::Rot3::Rz(-0.25), gtsam::Point3(-1.0, 1.5, 0.2));
  const auto submap1_T_submap2 =
      similarityFromPose(submap1_T_qbody) * qbody_T_mbody *
      similarityFromPose(submap2_T_mbody).inverse();
  const gtsam::Point3 point_submap2(0.1, 0.4, -0.3);
  const gtsam::Point3 expected_submap1 = submap1_T_qbody.transformFrom(
      qbody_T_mbody.transformFrom(
          submap2_T_mbody.inverse().transformFrom(point_submap2)));
  EXPECT_TRUE(submap1_T_submap2.transformFrom(point_submap2)
                  .isApprox(expected_submap1, 1e-9));
}

}  // namespace kimera_multi_lcd
