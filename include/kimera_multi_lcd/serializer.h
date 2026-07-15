#pragma once
#include <pose_graph_tools_msgs/msg/bow_vector.hpp>
#include <pose_graph_tools_msgs/msg/vlc_frame_msg.hpp>

#include <nlohmann/json.hpp>

namespace pose_graph_tools_msgs::msg {

void to_json(nlohmann::json& j, const pose_graph_tools_msgs::msg::BowVector& bow_vector);

void from_json(const nlohmann::json& j, pose_graph_tools_msgs::msg::BowVector& bow_vector);

void to_json(nlohmann::json& j, const pose_graph_tools_msgs::msg::VLCFrameMsg& vlc_frame);

void from_json(const nlohmann::json& j, pose_graph_tools_msgs::msg::VLCFrameMsg& vlc_frame);
}  // namespace pose_graph_tools_msgs::msg
