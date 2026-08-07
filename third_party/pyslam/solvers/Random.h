/**
 * This file is part of PYSLAM
 *
 * Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
 *
 * PYSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <vector>

namespace utils {

class Random {
 public:
  class UnrepeatedRandomizer;

  static void SeedRand();
  static void SeedRandOnce();
  static void SeedRand(int seed);
  static void SeedRandOnce(int seed);

  template <class T>
  static T RandomValue() {
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
  }

  template <class T>
  static T RandomValue(T min, T max) {
    return Random::RandomValue<T>() * (max - min) + min;
  }

  static int RandomInt(int min, int max);

  template <class T>
  static T RandomGaussianValue(T mean, T sigma) {
    T x1;
    T x2;
    T w;
    do {
      x1 = static_cast<T>(2.0) * RandomValue<T>() - static_cast<T>(1.0);
      x2 = static_cast<T>(2.0) * RandomValue<T>() - static_cast<T>(1.0);
      w = x1 * x1 + x2 * x2;
    } while (w >= static_cast<T>(1.0) || w == static_cast<T>(0.0));
    w = std::sqrt((static_cast<T>(-2.0) * std::log(w)) / w);
    return mean + x1 * w * sigma;
  }

 private:
  static bool m_already_seeded;
};

class Random::UnrepeatedRandomizer {
 public:
  UnrepeatedRandomizer(int min, int max);
  ~UnrepeatedRandomizer() = default;
  UnrepeatedRandomizer(const UnrepeatedRandomizer& rnd);
  UnrepeatedRandomizer& operator=(const UnrepeatedRandomizer& rnd);
  int get();
  inline bool empty() const { return m_values.empty(); }
  inline unsigned int left() const { return m_values.size(); }
  void reset();

 protected:
  void createValues();
  int m_min;
  int m_max;
  std::vector<int> m_values;
};

}  // namespace utils
