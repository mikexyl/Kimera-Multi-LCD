/*
 * Copyright Notes
 *
 * Authors: Yun Chang (yunchang@mit.edu)
 */

#include "kimera_multi_lcd/io.h"

#include <nlohmann/json.hpp>

#include "kimera_multi_lcd/serializer.h"
#include "kimera_multi_lcd/utils.h"

namespace kimera_multi_lcd {

using nlohmann::json;

void saveBowVectors(const std::map<PoseId, BowVectorMsg>& bow_vectors,
                    const std::string& filename) {
  std::ofstream outfile(filename);
  json record;
  record["ids"] = json::array();
  record["bow_vectors"] = json::array();
  for (const auto& id_bowvec : bow_vectors) {
    record["ids"].push_back(id_bowvec.first);
    json bow_json;
    pose_graph_tools_msgs::msg::to_json(bow_json, id_bowvec.second);
    record["bow_vectors"].push_back(bow_json);
  }
  outfile << record.dump();
}

void saveBowVectors(const std::map<PoseId, DBoW2::BowVector>& bow_vectors,
                    const std::string& filename) {
  std::map<PoseId, pose_graph_tools_msgs::msg::BowVector> pg_bow_vectors;
  for (const auto& id_bow : bow_vectors) {
    pg_bow_vectors[id_bow.first] = pose_graph_tools_msgs::msg::BowVector();
    BowVectorToMsg(id_bow.second, &pg_bow_vectors[id_bow.first]);
  }
  saveBowVectors(pg_bow_vectors, filename);
}

void saveVLCFrames(const std::map<PoseId, VLCFrameMsg>& vlc_frames,
                   const std::string& filename) {
  std::ofstream outfile(filename);
  json record;
  record["ids"] = json::array();
  record["frames"] = json::array();
  for (const auto& id_frame : vlc_frames) {
    record["ids"].push_back(id_frame.first);
    json frame_json;
    pose_graph_tools_msgs::msg::to_json(frame_json, id_frame.second);
    record["frames"].push_back(frame_json);
  }
  outfile << record.dump();
}

void saveVLCFrames(const std::map<PoseId, VLCFrame>& vlc_frames,
                   const std::string& filename) {
  std::map<PoseId, pose_graph_tools_msgs::msg::VLCFrameMsg> pg_vlc_frames;
  for (const auto& id_frame : vlc_frames) {
    pg_vlc_frames[id_frame.first] = pose_graph_tools_msgs::msg::VLCFrameMsg();
    VLCFrameToMsg(id_frame.second, &pg_vlc_frames[id_frame.first]);
  }
  saveVLCFrames(pg_vlc_frames, filename);
}

// Save BoW vectors
void loadBowVectors(const std::string& filename,
                    std::map<PoseId, BowVectorMsg>& bow_vectors) {
  std::stringstream ss;
  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error(filename + " does not exist");
  }
  ss << infile.rdbuf();

  const auto record = json::parse(ss.str());

  for (size_t i = 0; i < record.at("ids").size(); i++) {
    PoseId pose_id = record.at("ids").at(i);
    BowVectorMsg bow_vector;
    pose_graph_tools_msgs::msg::from_json(record.at("bow_vectors").at(i),
                                          bow_vector);
    bow_vectors[pose_id] = bow_vector;
  }
}

void loadBowVectors(const std::string& filename,
                    std::map<PoseId, DBoW2::BowVector>& bow_vectors) {
  std::map<PoseId, pose_graph_tools_msgs::msg::BowVector> pg_bow_vectors;
  loadBowVectors(filename, pg_bow_vectors);

  for (const auto& id_bow : pg_bow_vectors) {
    bow_vectors[id_bow.first] = DBoW2::BowVector();
    BowVectorFromMsg(id_bow.second, &bow_vectors[id_bow.first]);
  }
}

// Save VLC Frames
void loadVLCFrames(const std::string& filename,
                   std::map<PoseId, VLCFrameMsg>& vlc_frames) {
  std::stringstream ss;
  std::ifstream infile(filename);
  if (!infile) {
    throw std::runtime_error(filename + " does not exist");
  }
  ss << infile.rdbuf();

  const auto record = json::parse(ss.str());

  for (size_t i = 0; i < record.at("ids").size(); i++) {
    PoseId pose_id = record.at("ids").at(i);
    VLCFrameMsg vlc_frame;
    pose_graph_tools_msgs::msg::from_json(record.at("frames").at(i), vlc_frame);
    vlc_frames[pose_id] = vlc_frame;
  }
}

void loadVLCFrames(const std::string& filename,
                   std::map<PoseId, VLCFrame>& vlc_frames) {
  std::map<PoseId, VLCFrameMsg> pg_vlc_frames;
  loadVLCFrames(filename, pg_vlc_frames);
  for (const auto& id_frame : pg_vlc_frames) {
    vlc_frames[id_frame.first] = VLCFrame();
    VLCFrameFromMsg(id_frame.second, &vlc_frames[id_frame.first]);
  }
}
}  // namespace kimera_multi_lcd
