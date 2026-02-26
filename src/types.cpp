/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu)
 */
#include "kimera_multi_lcd/types.h"

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <sensor_msgs/image_encodings.h>

namespace kimera_multi_lcd {

VLCFrame::VLCFrame() {}

VLCFrame::VLCFrame(const RobotId& robot_id,
                   const PoseId& pose_id,
                   const std::vector<cv::Point2f>& keypoints,
                   const std::vector<gtsam::Vector3>& landmarks,
                   const std::vector<gtsam::Vector3>& versors,
                   const OrbDescriptor& descriptors_mat)
    : robot_id_(robot_id),
      pose_id_(pose_id),
      keypoints_(keypoints),
      landmarks_(landmarks),
      versors_(versors),
      descriptors_mat_(descriptors_mat) {
  assert(keypoints_.size() == descriptors_mat_.size().height);
  initializeDescriptorsVector();
}

VLCFrame::VLCFrame(const pose_graph_tools_msgs::VLCFrameMsg& msg)
    : robot_id_(msg.robot_id), pose_id_(msg.pose_id), submap_id_(msg.submap_id) {
  T_submap_pose_ = gtsam::Pose3(gtsam::Rot3(msg.T_submap_pose.orientation.w,
                                            msg.T_submap_pose.orientation.x,
                                            msg.T_submap_pose.orientation.y,
                                            msg.T_submap_pose.orientation.z),
                                gtsam::Point3(msg.T_submap_pose.position.x,
                                              msg.T_submap_pose.position.y,
                                              msg.T_submap_pose.position.z));
  T_base_cam_ = gtsam::Pose3(gtsam::Rot3(msg.T_base_cam.orientation.w,
                                         msg.T_base_cam.orientation.x,
                                         msg.T_base_cam.orientation.y,
                                         msg.T_base_cam.orientation.z),
                             gtsam::Point3(msg.T_base_cam.position.x,
                                           msg.T_base_cam.position.y,
                                           msg.T_base_cam.position.z));
  CHECK(not T_base_cam_.equals(gtsam::Pose3::Identity(), 1e-6));
  // Convert keypoints
  keypoints_.resize(msg.keypoints.size() / 2);
  for (size_t i = 0; i < keypoints_.size(); ++i) {
    keypoints_[i].x = msg.keypoints[2 * i];
    keypoints_[i].y = msg.keypoints[2 * i + 1];
  }

  // Convert versors and 3D keypoints
  if (!msg.versors.data.empty()) {
    pcl::PointCloud<pcl::PointXYZ> versors;
    pcl::fromROSMsg(msg.versors, versors);
    for (size_t i = 0; i < versors.size(); ++i) {
      gtsam::Vector3 v(versors[i].x, versors[i].y, versors[i].z);
      versors_.push_back(v);
      const auto depth = msg.depths[i];
      if (depth < 1e-3) {
        // Depth is invalid for this keypoint and the 3D keypoint is set to zero.
        // Zero keypoints will not be used during stereo RANSAC.
        landmarks_.push_back(gtsam::Vector3::Zero());
      } else {
        // Depth is valid for this keypoint.
        // We can recover the 3D point by multiplying with the bearing vector
        // See sparseStereoReconstruction function in Stereo Matcher in Kimera-VIO.
        landmarks_.push_back(depth * v / v(2));
      }
    }
  } else {
    // ROS_WARN("[VLCFrame] Empty versors!");
  }

  // Convert descriptors (FP16 -> FP32)
  // skip if descriptors are empty
  if (msg.descriptors_mat.data.empty()) {
    return;
  }

  sensor_msgs::ImageConstPtr ros_image_ptr(new sensor_msgs::Image(msg.descriptors_mat));
  cv::Mat descriptors_raw =
      cv_bridge::toCvCopy(ros_image_ptr, sensor_msgs::image_encodings::TYPE_16SC1)
          ->image;
  // cv::convertFp16 interprets CV_16S bits as IEEE-754 FP16 and produces CV_32F.
  cv::convertFp16(descriptors_raw, descriptors_mat_);
  initializeDescriptorsVector();
}

void VLCFrame::toROSMessage(pose_graph_tools_msgs::VLCFrameMsg* msg) const {
  msg->robot_id = robot_id_;
  msg->pose_id = pose_id_;

  // Convert submap info
  msg->submap_id = submap_id_;
  geometry_msgs::Pose pose;
  const gtsam::Point3& position = T_submap_pose_.translation();
  const gtsam::Quaternion& orientation = T_submap_pose_.rotation().toQuaternion();
  pose.position.x = position.x();
  pose.position.y = position.y();
  pose.position.z = position.z();
  pose.orientation.x = orientation.x();
  pose.orientation.y = orientation.y();
  pose.orientation.z = orientation.z();
  pose.orientation.w = orientation.w();
  msg->T_submap_pose = pose;

  // Convert keypoints
  pcl::PointCloud<pcl::PointXYZ> versors;
  for (size_t i = 0; i < landmarks_.size(); ++i) {
    // Push bearing vector
    gtsam::Vector3 v_ = versors_[i];
    pcl::PointXYZ v(v_(0), v_(1), v_(2));
    versors.push_back(v);
    // Push keypoint depth
    gtsam::Vector3 p_ = landmarks_[i];
    if (p_.norm() < 1e-3) {
      // This 3D keypoint is not valid
      msg->depths.push_back(0);
    } else {
      // We have valid 3D keypoint and the depth is given by the z component
      // See sparseStereoReconstruction function in Stereo Matcher in
      // Kimera-VIO.
      msg->depths.push_back(p_[2]);
    }
  }
  pcl::toROSMsg(versors, msg->versors);

  // convert keypoints
  msg->keypoints.resize(keypoints_.size() * 2);
  for (size_t i = 0; i < keypoints_.size(); ++i) {
    msg->keypoints[2 * i] = keypoints_[i].x;
    msg->keypoints[2 * i + 1] = keypoints_[i].y;
  }

  // Convert descriptors (FP32 -> FP16 for bandwidth reduction)
  assert(descriptors_mat_.type() ==
         CV_32FC1);  // check that the matrix is of type CV_32F
  // cv::convertFp16 packs CV_32F values into IEEE-754 FP16 stored as CV_16S.
  cv::Mat descriptors_fp16;
  cv::convertFp16(descriptors_mat_, descriptors_fp16);
  cv_bridge::CvImage cv_img;
  cv_img.encoding = sensor_msgs::image_encodings::TYPE_16SC1;
  cv_img.image = descriptors_fp16;
  cv_img.toImageMsg(msg->descriptors_mat);

  msg->T_base_cam.position.x = T_base_cam_.translation().x();
  msg->T_base_cam.position.y = T_base_cam_.translation().y();
  msg->T_base_cam.position.z = T_base_cam_.translation().z();
  const gtsam::Quaternion& quat = T_base_cam_.rotation().toQuaternion();
  msg->T_base_cam.orientation.x = quat.x();
  msg->T_base_cam.orientation.y = quat.y();
  msg->T_base_cam.orientation.z = quat.z();
  msg->T_base_cam.orientation.w = quat.w();
}

void VLCFrame::initializeDescriptorsVector() {
  descriptors_vec_.clear();
  // Create vector of descriptors
  int L = descriptors_mat_.size().width;
  descriptors_vec_.resize(descriptors_mat_.size().height);

  for (size_t i = 0; i < descriptors_vec_.size(); i++) {
    descriptors_vec_[i] = cv::Mat(1, L, descriptors_mat_.type());  // one row only
    descriptors_mat_.row(i).copyTo(descriptors_vec_[i].row(0));
  }
}

}  // namespace kimera_multi_lcd