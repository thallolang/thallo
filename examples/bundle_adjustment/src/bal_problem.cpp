// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#include "bal_problem.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Core"
#include "ceres/rotation.h"
#include "glog/logging.h"


// Inlining RandDouble and RandNormal from Ceres's examples/random.h

// Return a random number sampled from a uniform distribution in the range
// [0,1].
inline double RandDouble() {
    double r = static_cast<double>(rand());
    return r / RAND_MAX;
}

// Marsaglia Polar method for generation standard normal (pseudo)
// random numbers http://en.wikipedia.org/wiki/Marsaglia_polar_method
inline double RandNormal() {
    double x1, x2, w;
    do {
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0;
        w = x1 * x1 + x2 * x2;
    } while (w >= 1.0 || w == 0.0);

    w = sqrt((-2.0 * log(w)) / w);
    return x1 * w;
}


namespace {
typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
void FscanfOrDie(FILE* fptr, const char* format, T* value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    LOG(FATAL) << "Invalid UW data file.";
  }
}

void PerturbPoint3(const double sigma, double* point) {
  for (int i = 0; i < 3; ++i) {
    point[i] += RandNormal() * sigma;
  }
}

double Median(std::vector<double>* data) {
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n / 2;
  std::nth_element(data->begin(), mid_point, data->end());
  return *mid_point;
}

}  // namespace

BALProblem::BALProblem(const std::string& filename, bool sortForCoherency) {
  FILE* fptr = fopen(filename.c_str(), "r");

  if (fptr == NULL) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  // This wil die horribly on invalid files. Them's the breaks.
  FscanfOrDie(fptr, "%d", &num_cameras_);
  FscanfOrDie(fptr, "%d", &num_points_);
  FscanfOrDie(fptr, "%d", &num_observations_);

  VLOG(1) << "Header: " << num_cameras_
          << " " << num_points_
          << " " << num_observations_;

  point_index_ = new int[num_observations_];
  camera_index_ = new int[num_observations_];
  observations_ = new double[2 * num_observations_];

  num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
  parameters_ = new double[num_parameters_];

  for (int i = 0; i < num_observations_; ++i) {
    FscanfOrDie(fptr, "%d", camera_index_ + i);
    FscanfOrDie(fptr, "%d", point_index_ + i);
    for (int j = 0; j < 2; ++j) {
      FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
    }
  }

  for (int i = 0; i < num_parameters_; ++i) {
    FscanfOrDie(fptr, "%lf", parameters_ + i);
  }

  fclose(fptr);

  if (sortForCoherency) {
      coherenceSort();
  }

}

struct Observation {
    int pointIndex;
    int cameraIndex;
    double x;
    double y;
};

void BALProblem::coherenceSort() {
    printf("Sorting BAL observations\n");

    std::vector<Observation> obs;
    obs.resize(num_observations_);
    for (int i = 0; i < num_observations_; ++i) {
        obs[i].pointIndex = point_index_[i];
        obs[i].cameraIndex = camera_index_[i];
        obs[i].x = observations_[i * 2 + 0];
        obs[i].y = observations_[i * 2 + 1];
    }

    std::sort(obs.begin(), obs.end(), [](const Observation& a, const Observation& b) {
        if (a.cameraIndex == b.cameraIndex) {
            return a.pointIndex < b.pointIndex;
        }
        return a.cameraIndex < b.cameraIndex;
    });
    
    for (int i = 0; i < num_observations_; ++i) {
        point_index_[i] = obs[i].pointIndex;
        camera_index_[i] = obs[i].cameraIndex;
        observations_[i * 2 + 0] = obs[i].x;
        observations_[i * 2 + 1] = obs[i].y;
    }

    printf("Sorted BAL elements\n");
}


// This function writes the problem to a file in the same format that
// is read by the constructor.
void BALProblem::WriteToFile(const std::string& filename) const {
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    LOG(FATAL) << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

  for (int i = 0; i < num_observations_; ++i) {
    fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", observations_[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  for (int i = 0; i < num_cameras(); ++i) {
    double angleaxis[9];
    memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
    for (int j = 0; j < 9; ++j) {
      fprintf(fptr, "%.16g\n", angleaxis[j]);
    }
  }

  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for (int i = 0; i < num_points(); ++i) {
    const double* point = points + i * point_block_size();
    for (int j = 0; j < point_block_size(); ++j) {
      fprintf(fptr, "%.16g\n", point[j]);
    }
  }

  fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare.
void BALProblem::WriteToPLYFile(const std::string& filename) const {
  std::ofstream of(filename.c_str());

  of << "ply"
     << '\n' << "format ascii 1.0"
     << '\n' << "element vertex " << num_cameras_ + num_points_
     << '\n' << "property float x"
     << '\n' << "property float y"
     << '\n' << "property float z"
     << '\n' << "property uchar red"
     << '\n' << "property uchar green"
     << '\n' << "property uchar blue"
     << '\n' << "end_header" << std::endl;

  // Export extrinsic data (i.e. camera centers) as green points.
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras(); ++i)  {
    const double* camera = cameras() + camera_block_size() * i;
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    of << center[0] << ' ' << center[1] << ' ' << center[2]
       << " 0 255 0" << '\n';
  }

  // Export the structure (i.e. 3D Points) as white points.
  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for (int i = 0; i < num_points(); ++i) {
    const double* point = points + i * point_block_size();
    for (int j = 0; j < point_block_size(); ++j) {
      of << point[j] << ' ';
    }
    of << "255 255 255\n";
  }
  of.close();
}

void BALProblem::CameraToAngleAxisAndCenter(const double* camera,
                                            double* angle_axis,
                                            double* center) const {
  VectorRef angle_axis_ref(angle_axis, 3);
  angle_axis_ref = ConstVectorRef(camera, 3);

  // c = -R't
  Eigen::VectorXd inverse_rotation = -angle_axis_ref;
  ceres::AngleAxisRotatePoint(inverse_rotation.data(),
                       camera + camera_block_size() - 6,
                       center);
  VectorRef(center, 3) *= -1.0;
}

void BALProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const {
  ConstVectorRef angle_axis_ref(angle_axis, 3);
  VectorRef(camera, 3) = angle_axis_ref;

  // t = -R * c
  ceres::AngleAxisRotatePoint(angle_axis,
                       center,
                       camera + camera_block_size() - 6);
  VectorRef(camera + camera_block_size() - 6, 3) *= -1.0;
}


void BALProblem::Normalize() {
  // Compute the marginal median of the geometry.
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  double* points = mutable_points();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < num_points_; ++j) {
      tmp[j] = points[3 * j + i];
    }
    median(i) = Median(&tmp);
  }

  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100.
  const double scale = 100.0 / median_absolute_deviation;

  VLOG(2) << "median: " << median.transpose();
  VLOG(2) << "median absolute deviation: " << median_absolute_deviation;
  VLOG(2) << "scale: " << scale;

  // X = scale * (X - median)
  for (int i = 0; i < num_points_; ++i) {
    VectorRef point(points + 3 * i, 3);
    point = scale * (point - median);
  }

  double* cameras = mutable_cameras();
  double angle_axis[3];
  double center[3];
  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = cameras + camera_block_size() * i;
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    // center = scale * (center - median)
    VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
    AngleAxisAndCenterToCamera(angle_axis, center, camera);
  }
}

void BALProblem::Perturb(const double rotation_sigma,
                         const double translation_sigma,
                         const double point_sigma) {
  CHECK_GE(point_sigma, 0.0);
  CHECK_GE(rotation_sigma, 0.0);
  CHECK_GE(translation_sigma, 0.0);

  double* points = mutable_points();
  if (point_sigma > 0) {
    for (int i = 0; i < num_points_; ++i) {
      PerturbPoint3(point_sigma, points + 3 * i);
    }
  }

  for (int i = 0; i < num_cameras_; ++i) {
    double* camera = mutable_cameras() + camera_block_size() * i;

    double angle_axis[3];
    double center[3];
    // Perturb in the rotation of the camera in the angle-axis
    // representation.
    CameraToAngleAxisAndCenter(camera, angle_axis, center);
    if (rotation_sigma > 0.0) {
      PerturbPoint3(rotation_sigma, angle_axis);
    }
    AngleAxisAndCenterToCamera(angle_axis, center, camera);

    if (translation_sigma > 0.0) {
      PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
    }
  }
}

BALProblem::~BALProblem() {
  delete []point_index_;
  delete []camera_index_;
  delete []observations_;
  delete []parameters_;
}

void BALProblem::GetFloatParameters(std::vector<float>& observations, 
    std::vector<float>& cameraParameters, 
    std::vector<float>& pointParameters) const {
  observations.resize(2*num_observations_);
  for (int i = 0; i < observations.size();++i) {
    observations[i] = (float)observations_[i];
  }
  cameraParameters.resize(num_cameras()*camera_block_size());
  for (int i = 0; i < cameraParameters.size();++i) {
    cameraParameters[i] = (float)(cameras()[i]);
  }
  pointParameters.resize(num_points()*point_block_size());
  for (int i = 0; i < pointParameters.size();++i) {
    pointParameters[i] = (float)(points()[i]);
  }
}
void BALProblem::SetFloatParameters(const std::vector<float>& cameraParameters, const std::vector<float>& pointParameters) {
  double* cam = mutable_cameras();
  double* points = mutable_points();
  for (int i = 0; i < cameraParameters.size();++i) {
    cam[i] = (double)cameraParameters[i];
  }
  for (int i = 0; i < pointParameters.size();++i) {
    points[i] = (double)pointParameters[i];
  }
}
