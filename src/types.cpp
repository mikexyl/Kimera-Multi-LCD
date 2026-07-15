/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu)
 */
#include "kimera_multi_lcd/types.h"

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

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
  assert(keypoints_.size() == static_cast<size_t>(descriptors_mat_.rows));
  initializeDescriptorsVector();
}

VLCFrame::VLCFrame(const pose_graph_tools_msgs::msg::VLCFrameMsg& msg)
    : robot_id_(msg.robot_id), pose_id_(msg.pose_id), submap_id_(msg.submap_id) {
  T_submap_pose_ = gtsam::Pose3(gtsam::Rot3(msg.t_submap_pose.orientation.w,
                                            msg.t_submap_pose.orientation.x,
                                            msg.t_submap_pose.orientation.y,
                                            msg.t_submap_pose.orientation.z),
                                gtsam::Point3(msg.t_submap_pose.position.x,
                                              msg.t_submap_pose.position.y,
                                              msg.t_submap_pose.position.z));
  T_base_cam_ = gtsam::Pose3(gtsam::Rot3(msg.t_base_cam.orientation.w,
                                         msg.t_base_cam.orientation.x,
                                         msg.t_base_cam.orientation.y,
                                         msg.t_base_cam.orientation.z),
                             gtsam::Point3(msg.t_base_cam.position.x,
                                           msg.t_base_cam.position.y,
                                           msg.t_base_cam.position.z));
  keypoints_.resize(msg.keypoints.size() / 2);
  for (size_t i = 0; i < keypoints_.size(); ++i) {
    keypoints_[i].x = msg.keypoints[2 * i];
    keypoints_[i].y = msg.keypoints[2 * i + 1];
  }

  if (!msg.versors.data.empty()) {
    pcl::PointCloud<pcl::PointXYZ> versors;
    pcl::fromROSMsg(msg.versors, versors);
    for (size_t i = 0; i < versors.size(); ++i) {
      gtsam::Vector3 v(versors[i].x, versors[i].y, versors[i].z);
      versors_.push_back(v);
      const auto depth = i < msg.depths.size() ? msg.depths[i] : 0.0f;
      if (depth < 1e-3) {
        landmarks_.push_back(gtsam::Vector3::Zero());
      } else {
        landmarks_.push_back(depth * v / v(2));
      }
    }
  } else {
    LOG(WARNING) << "[VLCFrame] Empty versors!";
  }

  if (msg.descriptors_mat.data.empty()) {
    return;
  }

  auto ros_image_ptr =
      std::make_shared<sensor_msgs::msg::Image>(msg.descriptors_mat);
  cv::Mat descriptors_raw =
      cv_bridge::toCvCopy(ros_image_ptr, sensor_msgs::image_encodings::TYPE_16SC1)
          ->image;
  cv::convertFp16(descriptors_raw, descriptors_mat_);
  initializeDescriptorsVector();
}

void VLCFrame::toROSMessage(pose_graph_tools_msgs::msg::VLCFrameMsg* msg) const {
  msg->robot_id = robot_id_;
  msg->pose_id = pose_id_;
  msg->submap_id = submap_id_;

  geometry_msgs::msg::Pose pose;
  const gtsam::Point3& position = T_submap_pose_.translation();
  const gtsam::Quaternion& orientation = T_submap_pose_.rotation().toQuaternion();
  pose.position.x = position.x();
  pose.position.y = position.y();
  pose.position.z = position.z();
  pose.orientation.x = orientation.x();
  pose.orientation.y = orientation.y();
  pose.orientation.z = orientation.z();
  pose.orientation.w = orientation.w();
  msg->t_submap_pose = pose;

  pcl::PointCloud<pcl::PointXYZ> versors;
  for (size_t i = 0; i < landmarks_.size(); ++i) {
    gtsam::Vector3 v_ = versors_[i];
    versors.push_back(pcl::PointXYZ(v_(0), v_(1), v_(2)));

    gtsam::Vector3 p_ = landmarks_[i];
    if (p_.norm() < 1e-3) {
      msg->depths.push_back(0);
    } else {
      msg->depths.push_back(p_[2]);
    }
  }
  pcl::toROSMsg(versors, msg->versors);

  msg->keypoints.resize(keypoints_.size() * 2);
  for (size_t i = 0; i < keypoints_.size(); ++i) {
    msg->keypoints[2 * i] = keypoints_[i].x;
    msg->keypoints[2 * i + 1] = keypoints_[i].y;
  }

  if (!descriptors_mat_.empty()) {
    CHECK_EQ(descriptors_mat_.type(), CV_32FC1);
    cv::Mat descriptors_fp16;
    cv::convertFp16(descriptors_mat_, descriptors_fp16);
    cv_bridge::CvImage cv_img;
    cv_img.encoding = sensor_msgs::image_encodings::TYPE_16SC1;
    cv_img.image = descriptors_fp16;
    cv_img.toImageMsg(msg->descriptors_mat);
  }

  msg->t_base_cam.position.x = T_base_cam_.translation().x();
  msg->t_base_cam.position.y = T_base_cam_.translation().y();
  msg->t_base_cam.position.z = T_base_cam_.translation().z();
  const gtsam::Quaternion& quat = T_base_cam_.rotation().toQuaternion();
  msg->t_base_cam.orientation.x = quat.x();
  msg->t_base_cam.orientation.y = quat.y();
  msg->t_base_cam.orientation.z = quat.z();
  msg->t_base_cam.orientation.w = quat.w();
}

void VLCFrame::initializeDescriptorsVector() {
  descriptors_vec_.clear();
  int L = descriptors_mat_.cols;
  descriptors_vec_.resize(descriptors_mat_.rows);

  for (size_t i = 0; i < descriptors_vec_.size(); i++) {
    descriptors_vec_[i] = cv::Mat(1, L, descriptors_mat_.type());
    descriptors_mat_.row(i).copyTo(descriptors_vec_[i].row(0));
  }
}

}  // namespace kimera_multi_lcd
