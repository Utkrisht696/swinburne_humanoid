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

#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "hri_msgs/msg/ids_list.hpp"
#include "hri_msgs/msg/ids_match.hpp"
#include "hri_msgs/msg/normalized_point_of_interest2_d.hpp"
#include "hri_msgs/msg/normalized_region_of_interest2_d.hpp"
#include "hri_msgs/msg/skeleton2_d.hpp"
#include "lifecycle_msgs/srv/change_state.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rcl_interfaces/srv/set_parameters.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

#include "hri_face_body_matcher/node_face_body_matcher.hpp"

const int kMaxNumberPeople = 2;
const int kNodeCycleMs = 100;
const std::pair<float, float> kPositionCenter{0.5, 0.5};
const std::pair<float, float> kPositionFarLeft{0.25, 0.5};
const std::pair<float, float> kPositionFarRight{0.75, 0.5};
const std::pair<float, float> kPositionCloseLeft{0.35, 0.5};
const std::pair<float, float> kPositionCloseRight{0.65, 0.5};
const float kSizeStandard = 0.3;
const float kSizeSmall = 0.1;
const float kSizeBig = 0.9;
const float kDiscrepancyStandard = 0.3;
const float kDiscrepancyNoMatch = 0.8;

MATCHER(IdMatchIgnoringConfidenceAreEq, "")
{
  return
    (std::get<0>(arg).id1 == std::get<1>(arg).id1) &&
    (std::get<0>(arg).id1_type == std::get<1>(arg).id1_type) &&
    (std::get<0>(arg).id2 == std::get<1>(arg).id2) &&
    (std::get<0>(arg).id2_type == std::get<1>(arg).id2_type);
}

struct Person
{
  // (x,y) relative position of the person face center in the image
  std::pair<float, float> face_center_rel = kPositionCenter;
  // length of the face height relative to the image height
  float face_size_rel = kSizeStandard;
  // distance between body nose and face center, relative to the face diagonal length
  float discrepancy = kDiscrepancyStandard;
  bool has_face = true;
  bool has_body = true;
  // if a match is expected between the person head and body
  bool test_matching = true;
};

class NodeFaceBodyMatcherTestBase : public ::testing::Test
{
protected:
  explicit NodeFaceBodyMatcherTestBase(const std::vector<rclcpp::Parameter> & parameters)
  : parameters_(parameters)
  {}

  void SetUp() override
  {
    face_body_matcher_node_ = std::make_shared<hri_face_body_matcher::NodeFaceBodyMatcher>();
    face_body_matcher_node_->set_parameter({"use_sim_time", true});
    face_body_matcher_executor_ = rclcpp::executors::SingleThreadedExecutor::make_shared();
    face_body_matcher_executor_->add_node(face_body_matcher_node_->get_node_base_interface());

    tester_node_ = rclcpp::Node::make_shared("tester_node");
    tester_executor_ = rclcpp::executors::SingleThreadedExecutor::make_shared();
    tester_executor_->add_node(tester_node_);

    time_ = tester_node_->get_clock()->now();
    clock_pub_ = tester_node_->create_publisher<rosgraph_msgs::msg::Clock>("/clock", 1);

    bodies_tracked_pub_ = tester_node_->create_publisher<hri_msgs::msg::IdsList>(
      "/humans/bodies/tracked", 1);
    faces_tracked_pub_ = tester_node_->create_publisher<hri_msgs::msg::IdsList>(
      "/humans/faces/tracked", 1);
    for (int id = 0; id < kMaxNumberPeople; ++id) {
      body_skeleton_pubs_[id] = tester_node_->create_publisher<hri_msgs::msg::Skeleton2D>(
        "/humans/bodies/body" + std::to_string(id) + "/skeleton2d", 1);
    }
    for (int id = 0; id < kMaxNumberPeople; ++id) {
      face_roi_pubs_[id] =
        tester_node_->create_publisher<hri_msgs::msg::NormalizedRegionOfInterest2D>(
        "/humans/faces/face" + std::to_string(id) + "/roi", 1);
    }
    matches_sub_ = tester_node_->create_subscription<hri_msgs::msg::IdsMatch>(
      "/humans/candidate_matches", 10,
      [&match_msgs = match_msgs_](const hri_msgs::msg::IdsMatch & msg) {
        match_msgs.push_back(msg);
      });

    auto parameters_client = tester_node_->create_client<rcl_interfaces::srv::SetParameters>(
      "/hri_face_body_matcher/set_parameters");
    auto set_parameters_req = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();
    for (auto const & parameter : parameters_) {
      set_parameters_req->parameters.emplace_back(parameter.to_parameter_msg());
    }
    auto set_parameters_future = parameters_client->async_send_request(set_parameters_req);
    spin();
    ASSERT_THAT(
      set_parameters_future.get()->results,
      testing::Each(
        testing::Field(
          &rcl_interfaces::msg::SetParametersResult::successful, testing::Eq(true))));

    auto change_state_client = tester_node_->create_client<lifecycle_msgs::srv::ChangeState>(
      "/hri_face_body_matcher/change_state");
    auto change_state_req = std::make_shared<lifecycle_msgs::srv::ChangeState::Request>();
    change_state_req->transition.id = lifecycle_msgs::msg::Transition::TRANSITION_CONFIGURE;
    auto change_state_future = change_state_client->async_send_request(change_state_req);
    spin();
    ASSERT_TRUE(change_state_future.get()->success);
    change_state_req->transition.id = lifecycle_msgs::msg::Transition::TRANSITION_ACTIVATE;
    change_state_future = change_state_client->async_send_request(change_state_req);
    spin();
    ASSERT_TRUE(change_state_future.get()->success);
  }

  void TearDown() override
  {
    bodies_tracked_pub_->publish(hri_msgs::msg::IdsList());
    faces_tracked_pub_->publish(hri_msgs::msg::IdsList());
    spin();

    matches_sub_.reset();
    face_roi_pubs_.clear();
    body_skeleton_pubs_.clear();
    faces_tracked_pub_.reset();
    bodies_tracked_pub_.reset();

    match_msgs_.clear();
    face_body_matcher_node_.reset();
    face_body_matcher_executor_.reset();
    tester_node_.reset();
    tester_executor_.reset();
  }

  void spin(std::chrono::nanoseconds timeout = std::chrono::seconds(10))
  {
    face_body_matcher_executor_->spin_all(timeout);

    time_ += rclcpp::Duration(std::chrono::milliseconds(kNodeCycleMs + 1));
    rosgraph_msgs::msg::Clock clock_msg;
    clock_msg.clock = time_;
    clock_pub_->publish(clock_msg);
    face_body_matcher_executor_->spin_all(timeout);

    tester_executor_->spin_all(timeout);
  }

  void test(const std::vector<Person> & people)
  {
    ASSERT_LE(static_cast<int>(people.size()), kMaxNumberPeople) << "Unsupported number of people";

    const float face_ratio = 1.62;
    hri_msgs::msg::IdsList faces_tracked_msg;
    hri_msgs::msg::IdsList bodies_tracked_msg;
    std::map<int, hri_msgs::msg::NormalizedRegionOfInterest2D> face_roi_msgs;
    std::map<int, hri_msgs::msg::Skeleton2D> body_skeleton_msgs;
    std::vector<hri_msgs::msg::IdsMatch> expected_match_msgs;

    for (int id = 0; id < static_cast<int>(people.size()); ++id) {
      auto & person = people[id];
      auto face_center_x = person.face_center_rel.first;
      auto face_center_y = person.face_center_rel.second;
      auto face_height = person.face_size_rel;
      auto face_width = face_height / face_ratio;

      if (person.has_face) {
        auto face_min_x = face_center_x - (face_width / 2);
        auto face_max_x = face_center_x + (face_width / 2);
        auto face_min_y = face_center_y - (face_height / 2);
        auto face_max_y = face_center_y + (face_height / 2);
        ASSERT_GE(face_min_x, 0.) << "Computed face RoI is out of bounds";
        ASSERT_GE(face_min_y, 0.) << "Computed face RoI is out of bounds";
        ASSERT_LE(face_max_x, 1.) << "Computed face RoI is out of bounds";
        ASSERT_LE(face_max_y, 1.) << "Computed face RoI is out of bounds";

        faces_tracked_msg.ids.emplace_back("face" + std::to_string(id));
        face_roi_msgs[id] = hri_msgs::msg::NormalizedRegionOfInterest2D();
        face_roi_msgs[id].xmin = face_min_x;
        face_roi_msgs[id].ymin = face_min_y;
        face_roi_msgs[id].xmax = face_max_x;
        face_roi_msgs[id].ymax = face_max_y;
      }

      if (person.has_body) {
        auto nose_x = person.face_center_rel.first;
        auto nose_y = person.face_center_rel.second;
        if (person.has_face) {
          // shift the node position from the image center always towards the image center
          float image_center_x = 0.5;
          float image_center_y = 0.5;
          auto face_to_image_center_angle = std::atan2(
            image_center_y - face_center_y, image_center_x - face_center_x);
          auto face_diag_length = std::sqrt(face_height * face_height + face_width * face_width);
          nose_x = face_center_x + (
            std::cos(face_to_image_center_angle) * face_diag_length * person.discrepancy);
          nose_y = face_center_y + (
            std::sin(face_to_image_center_angle) * face_diag_length * person.discrepancy);
        }
        ASSERT_GE(nose_x, 0.) << "Computed body nose position is out of bounds";
        ASSERT_GE(nose_y, 0.) << "Computed body nose position is out of bounds";
        ASSERT_LE(nose_x, 1.) << "Computed body nose position is out of bounds";
        ASSERT_LE(nose_y, 1.) << "Computed body nose position is out of bounds";

        bodies_tracked_msg.ids.emplace_back("body" + std::to_string(id));
        body_skeleton_msgs[id] = hri_msgs::msg::Skeleton2D();
        body_skeleton_msgs[id].skeleton[hri_msgs::msg::Skeleton2D::NOSE].x = nose_x;
        body_skeleton_msgs[id].skeleton[hri_msgs::msg::Skeleton2D::NOSE].y = nose_y;
      }

      if (person.test_matching) {
        hri_msgs::msg::IdsMatch msg;
        msg.id1 = "body" + std::to_string(id);
        msg.id1_type = hri_msgs::msg::IdsMatch::BODY;
        msg.id2 = "face" + std::to_string(id);
        msg.id2_type = hri_msgs::msg::IdsMatch::FACE;
        expected_match_msgs.push_back(msg);
      }
    }

    bodies_tracked_pub_->publish(bodies_tracked_msg);
    faces_tracked_pub_->publish(faces_tracked_msg);
    spin();

    for (auto const & [id, msg] : body_skeleton_msgs) {
      body_skeleton_pubs_[id]->publish(msg);
    }
    for (auto const & [id, msg] : face_roi_msgs) {
      face_roi_pubs_[id]->publish(msg);
    }
    spin();
    EXPECT_THAT(
      match_msgs_,
      testing::UnorderedPointwise(
        IdMatchIgnoringConfidenceAreEq(),
        expected_match_msgs));
  }

  rclcpp::Publisher<hri_msgs::msg::IdsList>::SharedPtr bodies_tracked_pub_;
  rclcpp::Publisher<hri_msgs::msg::IdsList>::SharedPtr faces_tracked_pub_;
  std::map<int, rclcpp::Publisher<hri_msgs::msg::Skeleton2D>::SharedPtr> body_skeleton_pubs_;
  std::map<int, rclcpp::Publisher<hri_msgs::msg::NormalizedRegionOfInterest2D>::SharedPtr>
  face_roi_pubs_;
  rclcpp::Subscription<hri_msgs::msg::IdsMatch>::SharedPtr matches_sub_;
  std::vector<hri_msgs::msg::IdsMatch> match_msgs_;

private:
  std::shared_ptr<hri_face_body_matcher::NodeFaceBodyMatcher> face_body_matcher_node_;
  rclcpp::Node::SharedPtr tester_node_;
  rclcpp::Executor::SharedPtr face_body_matcher_executor_;
  rclcpp::Executor::SharedPtr tester_executor_;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
  rclcpp::Time time_;
  std::vector<rclcpp::Parameter> parameters_;
};

class NodeFaceBodyMatcherTestDefault : public NodeFaceBodyMatcherTestBase
{
protected:
  NodeFaceBodyMatcherTestDefault()
  : NodeFaceBodyMatcherTestBase({}) {}
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultOnePerson)
{
  test({Person{}});
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultOnePersonNoMatch)
{
  std::vector<Person> people(1);
  people[0].discrepancy = kDiscrepancyNoMatch;
  people[0].test_matching = false;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultOneSmallPerson)
{
  std::vector<Person> people(1);
  people[0].face_size_rel = kSizeSmall;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultOneBigPerson)
{
  std::vector<Person> people(1);
  people[0].face_size_rel = kSizeBig;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultNoFace)
{
  std::vector<Person> people(1);
  people[0].has_face = false;
  people[0].test_matching = false;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultTwoPeople)
{
  std::vector<Person> people(2);
  people[0].face_center_rel = kPositionFarLeft;
  people[1].face_center_rel = kPositionFarRight;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultTwoPeopleNoMatch)
{
  std::vector<Person> people(2);
  people[0].face_center_rel = kPositionFarLeft;
  people[0].discrepancy = kDiscrepancyNoMatch;
  people[0].test_matching = false;
  people[1].face_center_rel = kPositionFarRight;
  people[1].discrepancy = kDiscrepancyNoMatch;
  people[1].test_matching = false;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultTwoClosePeople)
{
  std::vector<Person> people(2);
  people[0].face_center_rel = kPositionCloseLeft;
  people[1].face_center_rel = kPositionCloseRight;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultTwoPeopleOneWithoutFace)
{
  std::vector<Person> people(2);
  people[0].face_center_rel = kPositionFarLeft;
  people[1].face_center_rel = kPositionFarRight;
  people[1].has_face = false;
  people[1].test_matching = false;
  test(people);
};
TEST_F(NodeFaceBodyMatcherTestDefault, DefaultTwoPeopleOneWithoutBody)
{
  std::vector<Person> people(2);
  people[0].face_center_rel = kPositionFarLeft;
  people[0].has_body = false;
  people[0].test_matching = false;
  people[1].face_center_rel = kPositionFarRight;
  test(people);
};

class NodeFaceBodyMatcherTestHighConfidenceScaling : public NodeFaceBodyMatcherTestBase
{
protected:
  NodeFaceBodyMatcherTestHighConfidenceScaling()
  : NodeFaceBodyMatcherTestBase({rclcpp::Parameter("confidence_scaling_factor", 5.)}) {}
};
TEST_F(NodeFaceBodyMatcherTestHighConfidenceScaling, HighConfidenceScalingOnePerson)
{
  std::vector<Person> people(1);
  people[0].test_matching = false;
  test(people);
};

class NodeFaceBodyMatcherTestHighThreshold : public NodeFaceBodyMatcherTestBase
{
protected:
  NodeFaceBodyMatcherTestHighThreshold()
  : NodeFaceBodyMatcherTestBase({rclcpp::Parameter("confidence_threshold", 0.9)}) {}
};
TEST_F(NodeFaceBodyMatcherTestHighThreshold, HighThresholdOnePerson)
{
  std::vector<Person> people(1);
  people[0].test_matching = false;
  test(people);
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
