/*
 * Copyright Notes
 *
 * Authors: Yulun Tian (yulun@mit.edu)
 */

#pragma once

#include <DBoW2/DBoW2.h>
#include <pose_graph_tools_msgs/msg/bow_queries.hpp>
#include <pose_graph_tools_msgs/msg/bow_vector.hpp>
#include <pose_graph_tools_msgs/msg/pose_graph.hpp>
#include <pose_graph_tools_msgs/msg/pose_graph_edge.hpp>
#include <pose_graph_tools_msgs/msg/vlc_frame_msg.hpp>

#include <map>

#include "kimera_multi_lcd/types.h"

namespace kimera_multi_lcd {
void BowVectorToMsg(const DBoW2::BowVector& bow_vec, pose_graph_tools_msgs::msg::BowVector* msg);

void BowVectorFromMsg(const pose_graph_tools_msgs::msg::BowVector& msg,
                      DBoW2::BowVector* bow_vec);

void VLCFrameToMsg(const VLCFrame& frame, pose_graph_tools_msgs::msg::VLCFrameMsg* msg);
void VLCFrameFromMsg(const pose_graph_tools_msgs::msg::VLCFrameMsg& msg, VLCFrame* frame);

void VLCEdgeToMsg(const VLCEdge& edge, pose_graph_tools_msgs::msg::PoseGraphEdge* msg);
void VLCEdgeFromMsg(const pose_graph_tools_msgs::msg::PoseGraphEdge& msg, VLCEdge* edge);

// Compute the payload size in a BowQuery message
size_t computeBowQueryPayloadBytes(const pose_graph_tools_msgs::msg::BowQuery& msg);

// Compute the payload size of a VLC frame
size_t computeVLCFramePayloadBytes(const pose_graph_tools_msgs::msg::VLCFrameMsg& msg);

}  // namespace kimera_multi_lcd
