#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <pose_graph_tools_msgs/msg/bow_vector.hpp>
#include <pose_graph_tools_msgs/msg/pose_graph_edge.hpp>
#include <pose_graph_tools_msgs/msg/vlc_frame_msg.hpp>

#include "kimera_multi_lcd/serializer.h"
#include "kimera_multi_lcd/types.h"
#include "kimera_multi_lcd/utils.h"

namespace kimera_multi_lcd {
namespace {

VLCFrame makeFrame() {
  std::vector<cv::Point2f> keypoints{{12.5f, 20.25f}, {30.0f, 40.5f}};
  std::vector<gtsam::Vector3> landmarks{
      gtsam::Vector3(0.5, 1.0, 2.0), gtsam::Vector3::Zero()};
  std::vector<gtsam::Vector3> versors{
      gtsam::Vector3(0.25, 0.5, 1.0), gtsam::Vector3(0.1, -0.2, 1.0)};
  std::vector<std::int64_t> landmark_ids{101, 202};
  cv::Mat descriptors = (cv::Mat_<float>(2, 4) <<
      0.125f, -0.25f, 0.5f, 1.0f,
      -1.0f, 0.75f, 0.375f, -0.125f);
  VLCFrame frame(
      2, 17, keypoints, landmarks, versors, landmark_ids, descriptors);
  frame.submap_id_ = 4;
  frame.T_submap_pose_ = gtsam::Pose3(
      gtsam::Rot3::RzRyRx(0.1, -0.2, 0.3), gtsam::Point3(1.0, 2.0, 3.0));
  frame.T_base_cam_ = gtsam::Pose3(
      gtsam::Rot3::RzRyRx(-0.1, 0.05, 0.2), gtsam::Point3(0.2, -0.1, 0.3));
  return frame;
}

void expectFramesEqual(const VLCFrame& expected, const VLCFrame& actual) {
  EXPECT_EQ(expected.robot_id_, actual.robot_id_);
  EXPECT_EQ(expected.pose_id_, actual.pose_id_);
  EXPECT_EQ(expected.submap_id_, actual.submap_id_);
  ASSERT_EQ(expected.keypoints_.size(), actual.keypoints_.size());
  ASSERT_EQ(expected.versors_.size(), actual.versors_.size());
  ASSERT_EQ(expected.landmarks_.size(), actual.landmarks_.size());
  EXPECT_EQ(expected.landmark_ids_, actual.landmark_ids_);
  for (size_t i = 0; i < expected.keypoints_.size(); ++i) {
    EXPECT_NEAR(expected.keypoints_[i].x, actual.keypoints_[i].x, 1e-5);
    EXPECT_NEAR(expected.keypoints_[i].y, actual.keypoints_[i].y, 1e-5);
    EXPECT_TRUE(gtsam::assert_equal(expected.versors_[i], actual.versors_[i],
                                    1e-5));
    EXPECT_TRUE(gtsam::assert_equal(expected.landmarks_[i], actual.landmarks_[i],
                                    1e-5));
  }
  EXPECT_LE(cv::norm(expected.descriptors_mat_ - actual.descriptors_mat_),
            2e-3);
  EXPECT_TRUE(expected.T_submap_pose_.equals(actual.T_submap_pose_, 1e-6));
  EXPECT_TRUE(expected.T_base_cam_.equals(actual.T_base_cam_, 1e-6));
}

}  // namespace

TEST(UtilsTest, BowVectorRoundTrip) {
  DBoW2::BowVector input;
  input.addWeight(1, 3.0);
  input.addWeight(2, 2.5);
  pose_graph_tools_msgs::msg::BowVector msg;
  BowVectorToMsg(input, &msg);
  DBoW2::BowVector output;
  BowVectorFromMsg(msg, &output);
  EXPECT_EQ(input, output);
}

TEST(UtilsTest, FloatDescriptorRoundTripPreservesFrameData) {
  const VLCFrame input = makeFrame();
  pose_graph_tools_msgs::msg::VLCFrameMsg msg;
  VLCFrameToMsg(input, &msg);
  EXPECT_EQ(msg.descriptors_mat.encoding, "16SC1");
  EXPECT_EQ(msg.depths.size(), input.landmarks_.size());
  VLCFrame output;
  VLCFrameFromMsg(msg, &output);
  expectFramesEqual(input, output);
}

TEST(UtilsTest, JsonRoundTripPreservesSerializedVlcFrame) {
  const VLCFrame input = makeFrame();
  pose_graph_tools_msgs::msg::VLCFrameMsg msg;
  VLCFrameToMsg(input, &msg);
  nlohmann::json json = msg;
  const auto decoded = json.get<pose_graph_tools_msgs::msg::VLCFrameMsg>();
  VLCFrame output;
  VLCFrameFromMsg(decoded, &output);
  expectFramesEqual(input, output);
}

TEST(UtilsTest, VLCEdgeRoundTrip) {
  const gtsam::Pose3 pose(gtsam::Rot3::RzRyRx(0.2, -0.1, 0.4),
                          gtsam::Point3(1.0, 2.0, 3.0));
  const VLCEdge input({0, 7}, {1, 9}, pose);
  pose_graph_tools_msgs::msg::PoseGraphEdge msg;
  VLCEdgeToMsg(input, &msg);
  VLCEdge output;
  VLCEdgeFromMsg(msg, &output);
  EXPECT_EQ(input.vertex_src_, output.vertex_src_);
  EXPECT_EQ(input.vertex_dst_, output.vertex_dst_);
  EXPECT_TRUE(input.T_src_dst_.equals(output.T_src_dst_, 1e-9));
}

}  // namespace kimera_multi_lcd
