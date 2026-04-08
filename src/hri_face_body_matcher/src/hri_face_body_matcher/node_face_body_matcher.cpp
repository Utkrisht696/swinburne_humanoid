// Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hri_face_body_matcher/node_face_body_matcher.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include "dlib/matrix.h"
#include "dlib/optimization/max_cost_assignment.h"
#include "hri/body.hpp"
#include "hri/face.hpp"
#include "hri/hri.hpp"
#include "hri/types.hpp"
#include "hri_msgs/msg/ids_match.hpp"
#include "hri_msgs/msg/skeleton2_d.hpp"
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_updater/diagnostic_status_wrapper.hpp>
#include "lifecycle_msgs/msg/state.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/rclcpp.hpp"

namespace hri_face_body_matcher
{

NodeFaceBodyMatcher::NodeFaceBodyMatcher(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("hri_face_body_matcher", "", options)
{
  auto descriptor = rcl_interfaces::msg::ParameterDescriptor{};

  descriptor.description = "Discards candidate matches with confidence lower that this threshold";
  this->declare_parameter("confidence_threshold", 0.5, descriptor);

  descriptor.description = "Scales the estimated confidence drop with the face-body distance";
  this->declare_parameter("confidence_scaling_factor", 2., descriptor);

  RCLCPP_INFO(this->get_logger(), "State: Unconfigured");
}

LifecycleCallbackReturn NodeFaceBodyMatcher::on_configure(const rclcpp_lifecycle::State &)
{
  using namespace std::chrono_literals;

  confidence_threshold_ = this->get_parameter("confidence_threshold").as_double();
  confidence_scaling_factor_ = this->get_parameter("confidence_scaling_factor").as_double();

  diagnostics_pub_ =
    this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    "/diagnostics", 1);
  diagnostics_timer_ = rclcpp::create_timer(
    this, this->get_clock(), 1s,
    std::bind(&NodeFaceBodyMatcher::publishDiagnostics, this));

  RCLCPP_INFO(this->get_logger(), "State: Inactive");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceBodyMatcher::on_cleanup(const rclcpp_lifecycle::State &)
{
  internal_cleanup();
  RCLCPP_INFO(this->get_logger(), "State: Unconfigured");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceBodyMatcher::on_activate(const rclcpp_lifecycle::State &)
{
  match_pub_ = this->create_publisher<hri_msgs::msg::IdsMatch>(
    "/humans/candidate_matches", 10);

  hri_listener_ = hri::HRIListener::create(shared_from_this());

  match_timer_ = rclcpp::create_timer(
    this, this->get_clock(), std::chrono::milliseconds(100),
    std::bind(&NodeFaceBodyMatcher::match, this));

  RCLCPP_INFO(this->get_logger(), "State: Active");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceBodyMatcher::on_deactivate(const rclcpp_lifecycle::State &)
{
  internal_deactivate();
  RCLCPP_INFO(this->get_logger(), "State: Inactive");
  return LifecycleCallbackReturn::SUCCESS;
}

LifecycleCallbackReturn NodeFaceBodyMatcher::on_shutdown(const rclcpp_lifecycle::State & state)
{
  if (state.id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    internal_deactivate();
  }
  internal_cleanup();
  RCLCPP_INFO(this->get_logger(), "State: Finalized");
  return LifecycleCallbackReturn::SUCCESS;
}

void NodeFaceBodyMatcher::internal_deactivate()
{
  match_timer_.reset();
  hri_listener_.reset();
  match_pub_.reset();
}

void NodeFaceBodyMatcher::internal_cleanup()
{
  diagnostics_timer_.reset();
  diagnostics_pub_.reset();
}

void NodeFaceBodyMatcher::match()
{
  // Using the Hungarian algorithm to find the maximum cost assignment between bodies and faces.
  // The cost is represented by the likelihood of the assignment in a scale from 0 to 100.
  // Since the algorithm requires a balanced assignment (number of bodies equal to the number of
  // faces) which is not guaranteed, dummy elements are added to the smaller set with zero
  // likelihood associated.
  // See http://dlib.net/max_cost_assignment_ex.cpp.html.

  std::vector<hri::ID> body_ids_{};
  std::vector<hri::ID> face_ids_{};

  auto bodies{hri_listener_->getBodies()};
  for (const auto & [id, body] : bodies) {
    body_ids_.push_back(id);
  }

  auto faces{hri_listener_->getFaces()};
  for (const auto & [id, face] : faces) {
    face_ids_.push_back(id);
  }

  auto body_and_face_exist = [&](int row, int col) -> bool {
      return (row < static_cast<int>(body_ids_.size())) &&
             (col < static_cast<int>(face_ids_.size()));
    };

  int dim{static_cast<int>(std::max(body_ids_.size(), face_ids_.size()))};
  dlib::matrix<int> cost(dim, dim);
  for (int row = 0; row < dim; ++row) {
    for (int col = 0; col < dim; ++col) {
      if (body_and_face_exist(row, col)) {
        cost(row, col) = matchingConfidence(bodies[body_ids_[row]], faces[face_ids_[col]]);
      } else {
        cost(row, col) = 0;
      }
    }
  }

  auto assignment{dlib::max_cost_assignment(cost)};

  for (int row = 0; row < dim; ++row) {
    auto matched_col = assignment[row];
    auto confidence = cost(row, matched_col) / 100.;
    if (body_and_face_exist(row, matched_col) && (confidence > confidence_threshold_)) {
      publishMatch(body_ids_[row], face_ids_[matched_col], confidence);
    }
  }
}

int NodeFaceBodyMatcher::matchingConfidence(hri::BodyPtr body, hri::FacePtr face)
{
  // It returns the confidence of the face to body matching in a scale from 0 to 100, based on face
  // center to body nose distance. It is 100 if the points coincide and decreases linearly to 50
  // when the points distance is equal to the face RoI diagonal length scaled by the
  // "confidence scaling factor".

  int confidence{0};
  auto body_skeleton = body->skeleton();
  auto face_roi = face->roi();
  if (body_skeleton && face_roi) {
    auto body_nose{body_skeleton->at(hri::SkeletalKeypoint::kNose)};
    cv::Point2f body_nose_point{body_nose.x, body_nose.y};
    cv::Point2f face_top_left_point{face_roi->x, face_roi->y};
    cv::Point2f face_bot_right_point{face_roi->x + face_roi->width, face_roi->y + face_roi->height};
    cv::Point2f face_center_point{(face_top_left_point + face_bot_right_point) / 2.};

    double distance = cv::norm(body_nose_point - face_center_point);
    double half_confidence_distance = cv::norm(
      face_bot_right_point - face_top_left_point) / confidence_scaling_factor_;

    confidence = static_cast<int>(std::max(0., 100. - (50. * distance / half_confidence_distance)));
  }
  return confidence;
}

void NodeFaceBodyMatcher::publishMatch(hri::ID body_id, hri::ID face_id, double confidence)
{
  hri_msgs::msg::IdsMatch match_msg{};
  match_msg.id1 = body_id;
  match_msg.id1_type = match_msg.BODY;
  match_msg.id2 = face_id;
  match_msg.id2_type = match_msg.FACE;
  match_msg.confidence = confidence;
  match_pub_->publish(match_msg);
}

void NodeFaceBodyMatcher::publishDiagnostics()
{
  diagnostic_updater::DiagnosticStatusWrapper status;
  status.name = "/social_perception/fusion/hri_face_body_matcher";
  status.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "");
  status.add("Current lifecycle state", this->get_current_state().label());

  diagnostic_msgs::msg::DiagnosticArray msg;
  msg.header.stamp = this->get_clock()->now();
  msg.status.push_back(status);
  diagnostics_pub_->publish(msg);
}

}  // namespace hri_face_body_matcher
