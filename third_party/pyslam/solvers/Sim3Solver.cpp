/**
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PYSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */

/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José
 * M.M. Montiel and Juan D. Tardós, University of Zaragoza. Copyright (C) 2014-2016 Raúl
 * Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 */

// Vendored from pySLAM cpp/solvers/Sim3Solver.cpp. Formatting and defensive
// finite/degeneracy checks were adapted for Kimera-Multi-LCD in August 2026.

#include "Sim3Solver.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "Random.h"

namespace utils {
namespace {

constexpr float kChiSquare2D99 = 9.210f;

}  // namespace

Sim3Solver::Sim3Solver(const Sim3SolverInput& input)
    : mN(static_cast<int>(input.mvX3Dw1.size())),
      ms12i(1.0f),
      mnInliersi(0),
      mnIterations(0),
      mnBestInliers(0),
      mBestScale(1.0f),
      mbFixScale(input.bFixScale) {
  assert(input.mvX3Dw1.size() == input.mvX3Dw2.size());
  assert(input.mvX3Dw1.size() == input.mvSigmaSquare1.size());
  assert(input.mvX3Dw1.size() == input.mvSigmaSquare2.size());

  mK1 = input.K1;
  mK2 = input.K2;
  mBestT12.setIdentity();
  mBestRotation.setIdentity();
  mBestTranslation.setZero();
  mvnIndices1.reserve(mN);
  mvX3Dc1.reserve(mN);
  mvX3Dc2.reserve(mN);
  mvAllIndices.reserve(mN);
  for (int i = 0; i < mN; ++i) {
    mvnMaxError1.push_back(kChiSquare2D99 * input.mvSigmaSquare1[i]);
    mvnMaxError2.push_back(kChiSquare2D99 * input.mvSigmaSquare2[i]);
    mvX3Dc1.push_back(input.Rcw1 * input.mvX3Dw1[i] + input.tcw1);
    mvX3Dc2.push_back(input.Rcw2 * input.mvX3Dw2[i] + input.tcw2);
    mvnIndices1.push_back(static_cast<size_t>(i));
    mvAllIndices.push_back(static_cast<size_t>(i));
  }
  FromCameraToImage(mvX3Dc1, mK1, mvP1im1);
  FromCameraToImage(mvX3Dc2, mK2, mvP2im2);
  SetRansacParameters();
}

Sim3Solver::Sim3Solver(const Sim3SolverInput2& input)
    : mvX3Dc1(input.mvX3Dc1),
      mvX3Dc2(input.mvX3Dc2),
      mN(static_cast<int>(input.mvX3Dc1.size())),
      ms12i(1.0f),
      mnInliersi(0),
      mnIterations(0),
      mnBestInliers(0),
      mBestScale(1.0f),
      mbFixScale(input.bFixScale) {
  assert(input.mvX3Dc1.size() == input.mvX3Dc2.size());
  assert(input.mvX3Dc1.size() == input.mvSigmaSquare1.size());
  assert(input.mvX3Dc1.size() == input.mvSigmaSquare2.size());

  mK1 = input.K1;
  mK2 = input.K2;
  mBestT12.setIdentity();
  mBestRotation.setIdentity();
  mBestTranslation.setZero();
  mvnIndices1.reserve(mN);
  mvAllIndices.reserve(mN);
  for (int i = 0; i < mN; ++i) {
    mvnMaxError1.push_back(kChiSquare2D99 * input.mvSigmaSquare1[i]);
    mvnMaxError2.push_back(kChiSquare2D99 * input.mvSigmaSquare2[i]);
    mvnIndices1.push_back(static_cast<size_t>(i));
    mvAllIndices.push_back(static_cast<size_t>(i));
  }
  FromCameraToImage(mvX3Dc1, mK1, mvP1im1);
  FromCameraToImage(mvX3Dc2, mK2, mvP2im2);
  SetRansacParameters();
}

void Sim3Solver::SetRansacParameters(double probability,
                                     int minInliers,
                                     int maxIterations) {
  mRansacProb = probability;
  mRansacMinInliers = minInliers;
  mRansacMaxIts = maxIterations;
  mvbInliersi.assign(static_cast<size_t>(mN), false);

  if (mN <= 0 || minInliers > mN) {
    mRansacMaxIts = 1;
    mnIterations = 0;
    return;
  }
  const double epsilon = static_cast<double>(minInliers) / mN;
  int iterations = 1;
  if (minInliers != mN) {
    const double denominator = std::log(1.0 - std::pow(epsilon, 3));
    if (std::isfinite(denominator) && denominator < 0.0) {
      iterations =
          static_cast<int>(std::ceil(std::log(1.0 - mRansacProb) / denominator));
    }
  }
  mRansacMaxIts = std::max(1, std::min(iterations, mRansacMaxIts));
  mnIterations = 0;
}

Eigen::Matrix4f Sim3Solver::iterate(const int nIterations,
                                    bool& bNoMore,
                                    std::vector<uint8_t>& vbInliers,
                                    int& nInliers,
                                    bool& bConverged) {
  bNoMore = false;
  bConverged = false;
  vbInliers.assign(static_cast<size_t>(mN), 0);
  nInliers = 0;
  if (mN < std::max(3, mRansacMinInliers)) {
    bNoMore = true;
    return Eigen::Matrix4f::Identity();
  }

  Eigen::Matrix4f bestSim3 = Eigen::Matrix4f::Identity();
  int currentIterations = 0;
  while (mnIterations < mRansacMaxIts && currentIterations < nIterations) {
    ++currentIterations;
    ++mnIterations;
    std::vector<size_t> availableIndices = mvAllIndices;
    Eigen::Matrix3f P3Dc1i;
    Eigen::Matrix3f P3Dc2i;
    for (short i = 0; i < 3; ++i) {
      const int randomPosition =
          Random::RandomInt(0, static_cast<int>(availableIndices.size()) - 1);
      const size_t index = availableIndices[static_cast<size_t>(randomPosition)];
      P3Dc1i.col(i) = mvX3Dc1[index];
      P3Dc2i.col(i) = mvX3Dc2[index];
      availableIndices[static_cast<size_t>(randomPosition)] = availableIndices.back();
      availableIndices.pop_back();
    }
    if (!ComputeSim3(P3Dc1i, P3Dc2i)) {
      continue;
    }
    CheckInliers();
    if (mnInliersi >= mnBestInliers) {
      mvbBestInliers = mvbInliersi;
      mnBestInliers = mnInliersi;
      mBestT12 = mT12i;
      mBestRotation = mR12i;
      mBestTranslation = mt12i;
      mBestScale = ms12i;
      bestSim3 = mBestT12;
      if (mnInliersi >= mRansacMinInliers) {
        nInliers = mnInliersi;
        for (int i = 0; i < mN; ++i) {
          if (mvbInliersi[static_cast<size_t>(i)]) {
            vbInliers[mvnIndices1[static_cast<size_t>(i)]] = 1;
          }
        }
        bConverged = true;
        return mBestT12;
      }
    }
  }
  if (mnIterations >= mRansacMaxIts) {
    bNoMore = true;
  }
  if (mnBestInliers > 0) {
    nInliers = mnBestInliers;
    for (int i = 0; i < mN; ++i) {
      if (mvbBestInliers[static_cast<size_t>(i)]) {
        vbInliers[mvnIndices1[static_cast<size_t>(i)]] = 1;
      }
    }
  }
  return bestSim3;
}

Eigen::Matrix4f Sim3Solver::find(std::vector<uint8_t>& vbInliers12,
                                 int& nInliers,
                                 bool& bConverged) {
  bool noMore = false;
  return iterate(mRansacMaxIts, noMore, vbInliers12, nInliers, bConverged);
}

void Sim3Solver::ComputeCentroid(Eigen::Matrix3f& P,
                                 Eigen::Matrix3f& Pr,
                                 Eigen::Vector3f& C) {
  C = P.rowwise().sum() / static_cast<float>(P.cols());
  for (int i = 0; i < P.cols(); ++i) {
    Pr.col(i) = P.col(i) - C;
  }
}

bool Sim3Solver::ComputeSim3(Eigen::Matrix3f& P1, Eigen::Matrix3f& P2) {
  Eigen::Matrix3f Pr1;
  Eigen::Matrix3f Pr2;
  Eigen::Vector3f O1;
  Eigen::Vector3f O2;
  ComputeCentroid(P1, Pr1, O1);
  ComputeCentroid(P2, Pr2, O2);

  const Eigen::Matrix3f M = Pr2 * Pr1.transpose();
  Eigen::Matrix4f N;
  N << M(0, 0) + M(1, 1) + M(2, 2), M(1, 2) - M(2, 1), M(2, 0) - M(0, 2),
      M(0, 1) - M(1, 0), M(1, 2) - M(2, 1), M(0, 0) - M(1, 1) - M(2, 2),
      M(0, 1) + M(1, 0), M(2, 0) + M(0, 2), M(2, 0) - M(0, 2), M(0, 1) + M(1, 0),
      -M(0, 0) + M(1, 1) - M(2, 2), M(1, 2) + M(2, 1), M(0, 1) - M(1, 0),
      M(2, 0) + M(0, 2), M(1, 2) + M(2, 1), -M(0, 0) - M(1, 1) + M(2, 2);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4f> eigSolver(N);
  if (eigSolver.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Vector4f q = eigSolver.eigenvectors().col(3);
  const Eigen::Quaternionf quaternion(q(0), q(1), q(2), q(3));
  if (!quaternion.coeffs().allFinite() || quaternion.norm() < 1e-6f) {
    return false;
  }
  mR12i = quaternion.normalized().toRotationMatrix();

  const Eigen::Matrix3f P3 = mR12i * Pr2;
  if (!mbFixScale) {
    const double denominator = P3.squaredNorm();
    if (!std::isfinite(denominator) || denominator <= 1e-12) {
      return false;
    }
    ms12i = static_cast<float>((Pr1.array() * P3.array()).sum() / denominator);
  } else {
    ms12i = 1.0f;
  }
  if (!std::isfinite(ms12i) || ms12i <= 0.0f) {
    return false;
  }
  mt12i = O1 - ms12i * mR12i * O2;
  if (!mR12i.allFinite() || !mt12i.allFinite()) {
    return false;
  }

  mT12i.setIdentity();
  mT12i.block<3, 3>(0, 0) = ms12i * mR12i;
  mT12i.block<3, 1>(0, 3) = mt12i;
  mT21i.setIdentity();
  const Eigen::Matrix3f sRinv = (1.0f / ms12i) * mR12i.transpose();
  mT21i.block<3, 3>(0, 0) = sRinv;
  mT21i.block<3, 1>(0, 3) = -sRinv * mt12i;
  return true;
}

void Sim3Solver::CheckInliers() {
  std::vector<Eigen::Vector2f> P1im2;
  std::vector<Eigen::Vector2f> P2im1;
  std::vector<float> depths1;
  std::vector<float> depths2;
  Project(mvX3Dc2, mT12i, mK1, P2im1, depths1);
  Project(mvX3Dc1, mT21i, mK2, P1im2, depths2);
  mnInliersi = 0;
  for (size_t i = 0; i < mvP1im1.size(); ++i) {
    const Eigen::Vector2f dist1 = mvP1im1[i] - P2im1[i];
    const Eigen::Vector2f dist2 = P1im2[i] - mvP2im2[i];
    const float err1 = dist1.squaredNorm();
    const float err2 = dist2.squaredNorm();
    const bool isInlier = std::isfinite(err1) && std::isfinite(err2) &&
                          err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i] &&
                          depths1[i] > 0.0f && depths2[i] > 0.0f;
    mvbInliersi[i] = isInlier;
    mnInliersi += isInlier ? 1 : 0;
  }
}

Eigen::Matrix4f Sim3Solver::GetEstimatedTransformation() { return mBestT12; }
Eigen::Matrix3f Sim3Solver::GetEstimatedRotation() { return mBestRotation; }
Eigen::Vector3f Sim3Solver::GetEstimatedTranslation() { return mBestTranslation; }
float Sim3Solver::GetEstimatedScale() { return mBestScale; }

float Sim3Solver::Compute3dRegistrationError() {
  if (mvX3Dc1.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  float error = 0.0f;
  for (size_t i = 0; i < mvX3Dc1.size(); ++i) {
    error += (mBestScale * mBestRotation * mvX3Dc2[i] + mBestTranslation - mvX3Dc1[i])
                 .norm();
  }
  return error / static_cast<float>(mvX3Dc1.size());
}

void Sim3Solver::Project(const std::vector<Eigen::Vector3f>& points,
                         const Eigen::Matrix4f& transform,
                         const Eigen::Matrix3f& calibration,
                         std::vector<Eigen::Vector2f>& projected,
                         std::vector<float>& depths) {
  const Eigen::Matrix3f rotation = transform.block<3, 3>(0, 0);
  const Eigen::Vector3f translation = transform.block<3, 1>(0, 3);
  projected.clear();
  depths.clear();
  projected.reserve(points.size());
  depths.reserve(points.size());
  for (const auto& point : points) {
    const Eigen::Vector3f transformed = rotation * point + translation;
    const float inverseDepth = 1.0f / transformed.z();
    projected.emplace_back(
        calibration(0, 0) * transformed.x() * inverseDepth + calibration(0, 2),
        calibration(1, 1) * transformed.y() * inverseDepth + calibration(1, 2));
    depths.push_back(transformed.z());
  }
}

void Sim3Solver::FromCameraToImage(const std::vector<Eigen::Vector3f>& points,
                                   const Eigen::Matrix3f& calibration,
                                   std::vector<Eigen::Vector2f>& projected) {
  projected.clear();
  projected.reserve(points.size());
  for (const auto& point : points) {
    const float inverseDepth = 1.0f / point.z();
    projected.emplace_back(
        calibration(0, 0) * point.x() * inverseDepth + calibration(0, 2),
        calibration(1, 1) * point.y() * inverseDepth + calibration(1, 2));
  }
}

}  // namespace utils
