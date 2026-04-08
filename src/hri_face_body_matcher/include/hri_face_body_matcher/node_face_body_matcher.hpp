// Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
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

#ifndef HRI_FACE_BODY_MATCHER__NODE_FACE_BODY_MATCHER_HPP_
#define HRI_FACE_BODY_MATCHER__NODE_FACE_BODY_MATCHER_HPP_

#include <map>
#include <memory>
#include <vector>

#include "hri/hri.hpp"
#include "hri/types.hpp"
#include "hri_msgs/msg/ids_match.hpp"

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace hri_face_body_matcher
{

using LifecycleCallbackReturn =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class NodeFaceBodyMatcher : public rclcpp_lifecycle::LifecycleNode
{
public:
  explicit NodeFaceBodyMatcher(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  LifecycleCallbackReturn on_configure(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_cleanup(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_activate(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_deactivate(const rclcpp_lifecycle::State &);
  LifecycleCallbackReturn on_shutdown(const rclcpp_lifecycle::State &);

private:
  void internal_cleanup();
  void internal_deactivate();
  void match();
  int matchingConfidence(hri::BodyPtr body, hri::FacePtr face);
  void publishMatch(hri::ID body_id, hri::ID face_id, double confidence);
  void publishDiagnostics();

  std::shared_ptr<hri::HRIListener> hri_listener_;
  std::shared_ptr<rclcpp::TimerBase> match_timer_;
  rclcpp::Publisher<hri_msgs::msg::IdsMatch>::SharedPtr match_pub_;
  double confidence_threshold_;       // lower confidence matches are not published
  double confidence_scaling_factor_;  // face to body face distance, relative to face diagonal,
                                      // correspondent to 0.5 confidence
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
  std::shared_ptr<rclcpp::TimerBase> diagnostics_timer_;
};

}  // namespace hri_face_body_matcher

#endif  // HRI_FACE_BODY_MATCHER__NODE_FACE_BODY_MATCHER_HPP_
