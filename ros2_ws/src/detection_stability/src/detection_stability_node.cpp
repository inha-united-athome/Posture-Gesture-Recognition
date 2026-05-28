#include "detection_stability/detection_stability_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>

#include "inha_interfaces/msg/posture_and_gesture_stability.hpp"
#include "rmw/qos_profiles.h"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/exceptions.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#ifdef DETECTION_STABILITY_HAVE_OPENCV
#include "opencv2/imgcodecs.hpp"
#endif

namespace detection_stability
{
namespace
{

double clamp01(const double value)
{
  return std::max(0.0, std::min(1.0, value));
}

double safe_positive(const double value, const double fallback)
{
  return value > 0.0 ? value : fallback;
}

double median_in_place(std::vector<double> & values)
{
  if (values.empty()) {
    return 0.0;
  }

  const auto size = values.size();
  const auto mid = size / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const double upper = values[mid];
  if (size % 2 == 1) {
    return upper;
  }
  const auto lower_it = std::max_element(values.begin(), values.begin() + mid);
  return 0.5 * (*lower_it + upper);
}

double median_copy(std::vector<double> values)
{
  return median_in_place(values);
}

std::string current_time_for_filename()
{
  const auto now = std::chrono::system_clock::now();
  const auto now_time_t = std::chrono::system_clock::to_time_t(now);
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
    now.time_since_epoch()) % 1000;

  std::tm local_time{};
  localtime_r(&now_time_t, &local_time);

  std::ostringstream stamp;
  stamp << std::put_time(&local_time, "%Y%m%d_%H%M%S") << "_"
        << std::setw(3) << std::setfill('0') << ms.count();
  return stamp.str();
}

std::string csv_escape(const std::string & value)
{
  const bool needs_quotes =
    value.find_first_of(",\"\n\r") != std::string::npos;
  if (!needs_quotes) {
    return value;
  }

  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char ch : value) {
    if (ch == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(ch);
  }
  escaped.push_back('"');
  return escaped;
}

BBox bbox_from_msg(const vision_msgs::msg::BoundingBox2D & msg)
{
  BBox bbox;
  bbox.cx = msg.center.position.x;
  bbox.cy = msg.center.position.y;
  bbox.w = msg.size_x;
  bbox.h = msg.size_y;
  bbox.min_x = bbox.cx - 0.5 * bbox.w;
  bbox.max_x = bbox.cx + 0.5 * bbox.w;
  bbox.min_y = bbox.cy - 0.5 * bbox.h;
  bbox.max_y = bbox.cy + 0.5 * bbox.h;
  bbox.valid = bbox.w > 1.0 && bbox.h > 1.0;
  return bbox;
}

double bbox_iou(const BBox & lhs, const BBox & rhs)
{
  if (!lhs.valid || !rhs.valid) {
    return 0.0;
  }

  const double ix_min = std::max(lhs.min_x, rhs.min_x);
  const double iy_min = std::max(lhs.min_y, rhs.min_y);
  const double ix_max = std::min(lhs.max_x, rhs.max_x);
  const double iy_max = std::min(lhs.max_y, rhs.max_y);
  const double iw = std::max(0.0, ix_max - ix_min);
  const double ih = std::max(0.0, iy_max - iy_min);
  const double intersection = iw * ih;
  const double lhs_area = lhs.w * lhs.h;
  const double rhs_area = rhs.w * rhs.h;
  const double union_area = lhs_area + rhs_area - intersection;
  if (union_area <= 0.0) {
    return 0.0;
  }
  return intersection / union_area;
}

}  // namespace

DetectionStabilityNode::DetectionStabilityNode()
: Node("detection_stability_node"),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_)
{
  detections_topic_ =
    this->declare_parameter<std::string>("detections_topic", "/gesture_and_posture/detection/detections");
  camera_info_topic_ = this->declare_parameter<std::string>(
    "camera_info_topic", "/camera/camera_head/color/camera_info");
  lidar_topic_ = this->declare_parameter<std::string>("lidar_topic", "/livox/lidar");
  output_topic_ = this->declare_parameter<std::string>("output_topic", "/gesture_and_posture/detection_stability");
  image_topic_ =
    this->declare_parameter<std::string>("image_topic", "/camera/camera_head/color/image_raw/compressed");
  selected_point_topic_ =
    this->declare_parameter<std::string>("selected_point_topic", "/gesture_and_posture/selected_person/point");
  selected_image_path_ =
    this->declare_parameter<std::string>("selected_image_path", "/home/thor/inha_log/selected_person.jpg");
  feedback_log_enabled_ =
    this->declare_parameter<bool>("feedback_log_enabled", feedback_log_enabled_);
  feedback_log_root_dir_ =
    this->declare_parameter<std::string>("feedback_log_root_dir", "/home/thor/inha_log/module/gesture_and_posture/detection_stability_logs");
  action_name_ =
    this->declare_parameter<std::string>("action_name", "select_stable_person");
  camera_frame_ = this->declare_parameter<std::string>("camera_frame", "camera_head_color_optical_frame");
  output_frame_ = this->declare_parameter<std::string>("output_frame", "map");

  sync_queue_size_ = this->declare_parameter<int>("sync_queue_size", 10);
  max_lidar_age_sec_ = this->declare_parameter<double>("max_lidar_age_sec", 0.25);
  transform_timeout_sec_ = this->declare_parameter<double>("transform_timeout_sec", 0.05);
  min_depth_m_ = this->declare_parameter<double>("min_depth_m", 0.10);
  max_depth_m_ = this->declare_parameter<double>("max_depth_m", 30.0);
  min_lidar_points_ = this->declare_parameter<int>("min_lidar_points", 5);
  lidar_point_step_ = this->declare_parameter<int>("lidar_point_step", 2);
  max_projected_lidar_points_ = this->declare_parameter<int>("max_projected_lidar_points", 4000);
  manual_camera_width_ =
    this->declare_parameter<int>("manual_camera_width", manual_camera_width_);
  manual_camera_height_ =
    this->declare_parameter<int>("manual_camera_height", manual_camera_height_);
  manual_camera_fx_ =
    this->declare_parameter<double>("manual_camera_fx", manual_camera_fx_);
  manual_camera_fy_ =
    this->declare_parameter<double>("manual_camera_fy", manual_camera_fy_);
  manual_camera_cx_ =
    this->declare_parameter<double>("manual_camera_cx", manual_camera_cx_);
  manual_camera_cy_ =
    this->declare_parameter<double>("manual_camera_cy", manual_camera_cy_);

  depth_spread_tolerance_ = this->declare_parameter<double>("depth_spread_tolerance", 0.08);
  depth_jump_tolerance_m_ = this->declare_parameter<double>("depth_jump_tolerance_m", 0.40);
  depth_spread_weight_ = this->declare_parameter<double>("depth_spread_weight", 0.70);
  depth_jump_weight_ = this->declare_parameter<double>("depth_jump_weight", 0.30);

  bbox_center_tolerance_ = this->declare_parameter<double>("bbox_center_tolerance", 0.35);
  bbox_size_tolerance_ = this->declare_parameter<double>("bbox_size_tolerance", 0.35);
  bbox_center_weight_ = this->declare_parameter<double>("bbox_center_weight", 0.70);
  bbox_size_weight_ = this->declare_parameter<double>("bbox_size_weight", 0.30);

  stable_age_frames_ = this->declare_parameter<int>("stable_age_frames", 10);
  max_missing_frames_ = this->declare_parameter<int>("max_missing_frames", 30);
  id_switch_memory_frames_ = this->declare_parameter<int>("id_switch_memory_frames", 5);
  id_switch_iou_threshold_ = this->declare_parameter<double>("id_switch_iou_threshold", 0.50);

  total_depth_weight_ = this->declare_parameter<double>("total_depth_weight", 0.45);
  total_bbox_weight_ = this->declare_parameter<double>("total_bbox_weight", 0.35);
  total_id_weight_ = this->declare_parameter<double>("total_id_weight", 0.20);

  default_min_class_score_ =
    this->declare_parameter<double>("default_min_class_score", 0.50);
  default_min_stability_score_ =
    this->declare_parameter<double>("default_min_stability_score", 0.70);
  depth_tie_tolerance_m_ =
    this->declare_parameter<double>("depth_tie_tolerance_m", 0.25);
  crop_margin_ratio_ =
    this->declare_parameter<double>("crop_margin_ratio", 0.15);
  min_selection_observations_ =
    this->declare_parameter<int>("min_selection_observations", 1);
  image_buffer_size_ =
    this->declare_parameter<int>("image_buffer_size", 30);

  sanitize_parameters();
  build_manual_camera_info();

  stability_pub_ =
    this->create_publisher<inha_interfaces::msg::PostureAndGestureStabilityArray>(
    output_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
  selected_point_pub_ =
    this->create_publisher<geometry_msgs::msg::PointStamped>(
    selected_point_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable());

  select_action_server_ = rclcpp_action::create_server<SelectStablePerson>(
    this,
    action_name_,
    std::bind(&DetectionStabilityNode::handle_select_goal, this, std::placeholders::_1,
    std::placeholders::_2),
    std::bind(&DetectionStabilityNode::handle_select_cancel, this, std::placeholders::_1),
    std::bind(&DetectionStabilityNode::handle_select_accepted, this, std::placeholders::_1));

  RCLCPP_INFO(
    this->get_logger(),
    "detection_stability_node ready: action=%s, detections=%s, camera_info=%s, "
    "lidar_pointcloud=%s, image=%s, output=%s, selected_point=%s. "
    "Input subscriptions stay idle until start=true.",
    action_name_.c_str(),
    detections_topic_.c_str(), camera_info_topic_.c_str(), lidar_topic_.c_str(),
    image_topic_.c_str(), output_topic_.c_str(), selected_point_topic_.c_str());
}

rclcpp_action::GoalResponse DetectionStabilityNode::handle_select_goal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const SelectStablePerson::Goal> goal)
{
  (void)uuid;
  if (!goal->start) {
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  if (goal->seconds == 0.0F) {
    RCLCPP_WARN(
      this->get_logger(),
      "Rejecting selection goal: seconds must be positive, or negative for unlimited.");
    return rclcpp_action::GoalResponse::REJECT;
  }

  std::lock_guard<std::mutex> lock(selection_mutex_);
  if (selection_active_) {
    RCLCPP_WARN(this->get_logger(), "Rejecting selection goal: another selection is active.");
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse DetectionStabilityNode::handle_select_cancel(
  const std::shared_ptr<GoalHandleSelectStablePerson> goal_handle)
{
  (void)goal_handle;
  request_stop_selection();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void DetectionStabilityNode::handle_select_accepted(
  const std::shared_ptr<GoalHandleSelectStablePerson> goal_handle)
{
  std::thread{
    std::bind(&DetectionStabilityNode::execute_select_goal, this, std::placeholders::_1),
    goal_handle}.detach();
}

void DetectionStabilityNode::execute_select_goal(
  const std::shared_ptr<GoalHandleSelectStablePerson> goal_handle)
{
  const auto goal = goal_handle->get_goal();
  auto result = std::make_shared<SelectStablePerson::Result>();

  if (!goal->start) {
    request_stop_selection();
    result->success = true;
    goal_handle->succeed(result);
    RCLCPP_INFO(this->get_logger(), "Selection stop goal handled. Input subscriptions are idle.");
    return;
  }

  reset_selection_state(*goal);
  open_feedback_log(*goal);
  {
    std::lock_guard<std::mutex> lock(selection_mutex_);
    active_goal_ = goal_handle;
  }
  start_input_subscriptions();

  const auto start_time = this->get_clock()->now();
  const double duration_sec = static_cast<double>(goal->seconds);
  const bool unlimited = duration_sec < 0.0;
  bool canceled = false;
  bool stopped = false;
  std::optional<CandidateSummary> early_selected;

  if (unlimited) {
    RCLCPP_INFO(
      this->get_logger(),
      "Selection goal running in unlimited mode (seconds=%.1f). Will stop when a valid "
      "candidate is found.", duration_sec);
  }

  while (rclcpp::ok()) {
    if (goal_handle->is_canceling()) {
      canceled = true;
      break;
    }

    {
      std::lock_guard<std::mutex> lock(selection_mutex_);
      stopped = stop_selection_requested_;
    }
    if (stopped) {
      break;
    }

    if (unlimited) {
      std::lock_guard<std::mutex> lock(selection_mutex_);
      early_selected = choose_best_candidate_locked();
      if (early_selected) {
        break;
      }
    } else {
      const auto now = this->get_clock()->now();
      if ((now - start_time).seconds() >= duration_sec) {
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  stop_input_subscriptions();

  std::optional<CandidateSummary> selected;
  if (!canceled && !stopped) {
    if (early_selected) {
      selected = early_selected;
    } else {
      std::lock_guard<std::mutex> lock(selection_mutex_);
      selected = choose_best_candidate_locked();
    }
  }

  if (selected) {
    auto point = selected->point;
    if (transform_point_to_output_frame(point)) {
      selected->point = point;
      result->best_point = point;
      result->success = publish_and_save_selection(*selected);
    } else {
      result->success = false;
    }
  } else {
    result->success = false;
  }

  {
    std::lock_guard<std::mutex> lock(selection_mutex_);
    if (active_goal_ == goal_handle) {
      active_goal_.reset();
    }
    selection_active_ = false;
    stop_selection_requested_ = false;
    candidate_summaries_.clear();
    image_buffer_.clear();
  }
  close_feedback_log();

  if (canceled) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Selection goal canceled.");
    return;
  }

  goal_handle->succeed(result);
  if (stopped) {
    RCLCPP_INFO(this->get_logger(), "Selection goal stopped by start=false goal.");
  } else if (result->success) {
    RCLCPP_INFO(this->get_logger(), "Selection goal succeeded.");
  } else {
    RCLCPP_WARN(this->get_logger(), "Selection goal finished without a saved selected person.");
  }
}

void DetectionStabilityNode::start_input_subscriptions()
{
  std::lock_guard<std::mutex> lock(input_mutex_);
  if (inputs_active_) {
    return;
  }

  {
    std::lock_guard<std::mutex> camera_lock(camera_info_mutex_);
    if (!latest_camera_info_) {
      camera_info_sub_ =
        this->create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_, rclcpp::SensorDataQoS(),
        std::bind(&DetectionStabilityNode::on_camera_info, this, std::placeholders::_1));
    }
  }

  image_sub_ =
    this->create_subscription<sensor_msgs::msg::CompressedImage>(
    image_topic_, rclcpp::SensorDataQoS(),
    std::bind(&DetectionStabilityNode::on_image, this, std::placeholders::_1));

  filtered_detections_sub_.subscribe(this, detections_topic_, rmw_qos_profile_sensor_data);
  filtered_cloud_sub_.subscribe(this, lidar_topic_, rmw_qos_profile_sensor_data);
  cloud_sync_ = std::make_unique<message_filters::Synchronizer<CloudSyncPolicy>>(
    CloudSyncPolicy(static_cast<uint32_t>(sync_queue_size_)),
    filtered_detections_sub_,
    filtered_cloud_sub_);
  cloud_sync_->registerCallback(
    std::bind(&DetectionStabilityNode::on_point_cloud_pair, this, std::placeholders::_1,
    std::placeholders::_2));

  frame_index_ = 0;
  tracks_.clear();
  inputs_active_ = true;

  RCLCPP_INFO(
    this->get_logger(),
    "Selection input subscriptions started: detections=%s, lidar=%s, image=%s",
    detections_topic_.c_str(), lidar_topic_.c_str(), image_topic_.c_str());
}

void DetectionStabilityNode::stop_input_subscriptions()
{
  std::lock_guard<std::mutex> lock(input_mutex_);
  cloud_sync_.reset();
  filtered_detections_sub_.unsubscribe();
  filtered_cloud_sub_.unsubscribe();
  image_sub_.reset();
  bool has_camera_info = false;
  {
    std::lock_guard<std::mutex> camera_lock(camera_info_mutex_);
    has_camera_info = static_cast<bool>(latest_camera_info_);
  }
  if (!has_camera_info) {
    camera_info_sub_.reset();
  }
  inputs_active_ = false;
}

void DetectionStabilityNode::request_stop_selection()
{
  {
    std::lock_guard<std::mutex> lock(selection_mutex_);
    stop_selection_requested_ = true;
  }
  stop_input_subscriptions();
}

void DetectionStabilityNode::reset_selection_state(const SelectStablePerson::Goal & goal)
{
  std::lock_guard<std::mutex> lock(selection_mutex_);
  target_class_name_ = goal.class_name;
  active_min_class_score_ = goal.min_class_score > 0.0F ?
    goal.min_class_score : static_cast<float>(default_min_class_score_);
  active_min_stability_score_ = goal.min_stability_score > 0.0F ?
    goal.min_stability_score : static_cast<float>(default_min_stability_score_);
  active_min_class_score_ = static_cast<float>(clamp01(active_min_class_score_));
  active_min_stability_score_ = static_cast<float>(clamp01(active_min_stability_score_));
  stop_selection_requested_ = false;
  selection_active_ = true;
  candidate_summaries_.clear();
  image_buffer_.clear();
}

void DetectionStabilityNode::sanitize_parameters()
{
  sync_queue_size_ = std::max(1, sync_queue_size_);
  max_lidar_age_sec_ = std::max(0.0, max_lidar_age_sec_);
  transform_timeout_sec_ = std::max(0.0, transform_timeout_sec_);
  min_depth_m_ = std::max(0.0, min_depth_m_);
  max_depth_m_ = std::max(min_depth_m_ + 0.01, max_depth_m_);
  min_lidar_points_ = std::max(1, min_lidar_points_);
  lidar_point_step_ = std::max(1, lidar_point_step_);
  max_projected_lidar_points_ = std::max(0, max_projected_lidar_points_);
  manual_camera_width_ = std::max(0, manual_camera_width_);
  manual_camera_height_ = std::max(0, manual_camera_height_);
  manual_camera_fx_ = std::max(0.0, manual_camera_fx_);
  manual_camera_fy_ = std::max(0.0, manual_camera_fy_);
  manual_camera_cx_ = std::max(0.0, manual_camera_cx_);
  manual_camera_cy_ = std::max(0.0, manual_camera_cy_);
  depth_spread_tolerance_ = safe_positive(depth_spread_tolerance_, 0.08);
  depth_jump_tolerance_m_ = safe_positive(depth_jump_tolerance_m_, 0.40);
  bbox_center_tolerance_ = safe_positive(bbox_center_tolerance_, 0.35);
  bbox_size_tolerance_ = safe_positive(bbox_size_tolerance_, 0.35);
  stable_age_frames_ = std::max(1, stable_age_frames_);
  max_missing_frames_ = std::max(1, max_missing_frames_);
  id_switch_memory_frames_ = std::max(1, id_switch_memory_frames_);
  id_switch_iou_threshold_ = clamp01(id_switch_iou_threshold_);
  total_depth_weight_ = std::max(0.0, total_depth_weight_);
  total_bbox_weight_ = std::max(0.0, total_bbox_weight_);
  total_id_weight_ = std::max(0.0, total_id_weight_);
  default_min_class_score_ = clamp01(default_min_class_score_);
  default_min_stability_score_ = clamp01(default_min_stability_score_);
  depth_tie_tolerance_m_ = std::max(0.0, depth_tie_tolerance_m_);
  crop_margin_ratio_ = std::max(0.0, crop_margin_ratio_);
  min_selection_observations_ = std::max(1, min_selection_observations_);
  image_buffer_size_ = std::max(1, image_buffer_size_);
}

void DetectionStabilityNode::build_manual_camera_info()
{
  const bool has_any_manual_value =
    manual_camera_width_ > 0 ||
    manual_camera_height_ > 0 ||
    manual_camera_fx_ > 0.0 ||
    manual_camera_fy_ > 0.0 ||
    manual_camera_cx_ > 0.0 ||
    manual_camera_cy_ > 0.0;

  const bool has_complete_manual_info =
    manual_camera_width_ > 0 &&
    manual_camera_height_ > 0 &&
    manual_camera_fx_ > 0.0 &&
    manual_camera_fy_ > 0.0 &&
    manual_camera_cx_ > 0.0 &&
    manual_camera_cy_ > 0.0;

  if (!has_complete_manual_info) {
    if (has_any_manual_value) {
      RCLCPP_WARN(
        this->get_logger(),
        "Manual camera intrinsics are incomplete. Set manual_camera_width, "
        "manual_camera_height, manual_camera_fx, manual_camera_fy, manual_camera_cx, "
        "and manual_camera_cy to enable CameraInfo fallback.");
    }
    return;
  }

  auto camera_info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  camera_info->header.frame_id = camera_frame_;
  camera_info->width = static_cast<uint32_t>(manual_camera_width_);
  camera_info->height = static_cast<uint32_t>(manual_camera_height_);
  camera_info->k[0] = manual_camera_fx_;
  camera_info->k[2] = manual_camera_cx_;
  camera_info->k[4] = manual_camera_fy_;
  camera_info->k[5] = manual_camera_cy_;
  camera_info->k[8] = 1.0;
  camera_info->p[0] = manual_camera_fx_;
  camera_info->p[2] = manual_camera_cx_;
  camera_info->p[5] = manual_camera_fy_;
  camera_info->p[6] = manual_camera_cy_;
  camera_info->p[10] = 1.0;
  camera_info->r[0] = 1.0;
  camera_info->r[4] = 1.0;
  camera_info->r[8] = 1.0;
  manual_camera_info_ = camera_info;

  RCLCPP_WARN(
    this->get_logger(),
    "CameraInfo fallback enabled from manual intrinsics: width=%d height=%d "
    "fx=%.3f fy=%.3f cx=%.3f cy=%.3f frame=%s",
    manual_camera_width_, manual_camera_height_, manual_camera_fx_, manual_camera_fy_,
    manual_camera_cx_, manual_camera_cy_, camera_frame_.c_str());
}

void DetectionStabilityNode::on_camera_info(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
{
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    if (latest_camera_info_) {
      return;
    }
    latest_camera_info_ = msg;
  }

  camera_info_sub_.reset();
  RCLCPP_INFO(
    this->get_logger(),
    "CameraInfo received once from %s. CameraInfo subscription stopped.",
    camera_info_topic_.c_str());
}

sensor_msgs::msg::CameraInfo::ConstSharedPtr DetectionStabilityNode::current_camera_info()
{
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info;
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_info = latest_camera_info_;
  }
  if (!camera_info) {
    camera_info = manual_camera_info_;
  }
  return camera_info;
}

void DetectionStabilityNode::on_image(
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(selection_mutex_);
  if (!selection_active_) {
    return;
  }

  image_buffer_.push_back({rclcpp::Time(msg->header.stamp), msg});
  while (image_buffer_.size() > static_cast<size_t>(image_buffer_size_)) {
    image_buffer_.pop_front();
  }
}

void DetectionStabilityNode::on_point_cloud_pair(
  const DetectionMsg::ConstSharedPtr & detections,
  const CloudMsg::ConstSharedPtr & cloud)
{
  process_detections(detections, cloud);
}

void DetectionStabilityNode::process_detections(
  const DetectionMsg::ConstSharedPtr & msg,
  const CloudMsg::ConstSharedPtr & cloud)
{
  {
    std::lock_guard<std::mutex> lock(input_mutex_);
    if (!inputs_active_) {
      return;
    }
  }

  ++frame_index_;

  const auto camera_info = current_camera_info();

  auto output = inha_interfaces::msg::PostureAndGestureStabilityArray();
  output.header = msg->header;
  std::unordered_set<std::string> seen_tracks;

  rclcpp::Time target_stamp(msg->header.stamp);
  if (target_stamp.nanoseconds() == 0) {
    if (cloud) {
      target_stamp = rclcpp::Time(cloud->header.stamp);
    }
  }

  std::vector<DetectionContext> contexts;
  contexts.reserve(msg->detections.size());
  for (size_t index = 0; index < msg->detections.size(); ++index) {
    const auto & detection = msg->detections[index];
    const auto bbox = bbox_from_msg(detection.bbox);
    if (!bbox.valid) {
      continue;
    }

    auto track_id = extract_track_id(detection.id);
    if (track_id.empty()) {
      track_id = "det:" + std::to_string(index);
    }

    contexts.push_back({index, bbox, track_id, extract_class_name(detection)});
  }

  if (contexts.empty()) {
    mark_missing_tracks(seen_tracks);
    stability_pub_->publish(output);
    update_active_selection(output);
    return;
  }

  const auto search_bbox = make_search_bbox(contexts);
  const auto projected_lidar =
    project_lidar_once(camera_info, cloud, msg->header.frame_id, target_stamp, search_bbox);

  for (const auto & context : contexts) {
    const auto & detection = msg->detections[context.index];
    const auto prior_it = tracks_.find(context.track_id);
    const bool has_prior = prior_it != tracks_.end();
    const TrackState prior = has_prior ? prior_it->second : TrackState();

    const auto depth_stats = compute_depth_stats(context.bbox, projected_lidar);
    const double depth_cost = compute_depth_cost(depth_stats, has_prior ? &prior : nullptr);
    const double bbox_cost = compute_bbox_cost(context.bbox, has_prior ? &prior : nullptr);
    const double id_cost =
      compute_id_cost(context.track_id, context.bbox, has_prior ? &prior : nullptr);
    const double total_cost = compute_total_cost(depth_cost, bbox_cost, id_cost);

    auto stability = inha_interfaces::msg::PostureAndGestureStability();
    stability.header = detection.header;
    if (stability.header.frame_id.empty()) {
      stability.header = msg->header;
    }
    stability.detection = detection;
    stability.track_id = context.track_id;
    stability.class_name = context.class_name;
    stability.total_cost = static_cast<float>(total_cost);
    stability.stability_score = static_cast<float>(1.0 - total_cost);
    stability.depth_cost = static_cast<float>(depth_cost);
    stability.bbox_cost = static_cast<float>(bbox_cost);
    stability.id_cost = static_cast<float>(id_cost);
    stability.has_depth = depth_stats.has_depth;
    stability.depth_median_m = static_cast<float>(depth_stats.median_m);
    stability.depth_mad_m = static_cast<float>(depth_stats.mad_m);
    stability.lidar_points = depth_stats.point_count;
    stability.track_age = has_prior ? prior.age + 1 : 1;
    stability.missed_frames = has_prior ? prior.missed_frames : 0;
    output.stabilities.push_back(stability);

    auto & state = tracks_[context.track_id];
    state.has_bbox = true;
    state.bbox = context.bbox;
    if (depth_stats.has_depth) {
      state.has_depth = true;
      state.depth_median_m = depth_stats.median_m;
    }
    state.age = stability.track_age;
    state.missed_frames = 0;
    state.last_seen_frame = frame_index_;
    seen_tracks.insert(context.track_id);
  }

  mark_missing_tracks(seen_tracks);
  stability_pub_->publish(output);
  update_active_selection(output);
}

void DetectionStabilityNode::update_active_selection(
  const inha_interfaces::msg::PostureAndGestureStabilityArray & output)
{
  const auto camera_info = current_camera_info();
  CameraProjection projection;
  if (camera_info) {
    projection = make_projection(camera_info, output.header.frame_id);
  }

  auto feedback = std::make_shared<SelectStablePerson::Feedback>();
  std::shared_ptr<GoalHandleSelectStablePerson> goal_handle;

  {
    std::lock_guard<std::mutex> lock(selection_mutex_);
    if (!selection_active_ || stop_selection_requested_ || !active_goal_) {
      return;
    }
    goal_handle = active_goal_;

    const auto make_log_line =
      [this, &output](
        const inha_interfaces::msg::PostureAndGestureStability & stability,
        const double class_score,
        const geometry_msgs::msg::PointStamped * point,
        const bool accepted_candidate,
        const std::string & reject_reason)
      {
        std::ostringstream line;
        line << std::fixed << std::setprecision(6)
             << csv_escape(feedback_action_id_) << ','
             << feedback_goal_seconds_ << ','
             << csv_escape(feedback_target_class_name_) << ','
             << active_min_class_score_ << ','
             << active_min_stability_score_ << ','
             << output.header.stamp.sec << "." << std::setw(9) << std::setfill('0')
             << output.header.stamp.nanosec << std::setfill(' ') << ','
             << frame_index_ << ','
             << csv_escape(stability.track_id) << ','
             << csv_escape(stability.class_name) << ','
             << class_score << ','
             << stability.stability_score << ','
             << stability.depth_median_m << ','
             << stability.depth_cost << ','
             << stability.bbox_cost << ','
             << stability.id_cost << ','
             << stability.lidar_points << ',';
        if (point) {
          line << point->point.x << ','
               << point->point.y << ','
               << point->point.z << ',';
        } else {
          line << ",,,";
        }
        line << (accepted_candidate ? "true" : "false") << ','
             << reject_reason;
        return line.str();
      };

    for (const auto & stability : output.stabilities) {
      const double class_score = extract_class_score(stability);
      std::string reject_reason;
      if (!class_matches(stability.class_name)) {
        reject_reason = "class_mismatch";
        write_feedback_log_line(
          make_log_line(stability, class_score, nullptr, false, reject_reason));
        continue;
      }

      geometry_msgs::msg::PointStamped point;
      if (!make_point_from_stability(stability, projection, point)) {
        reject_reason = "no_depth_or_projection";
        write_feedback_log_line(
          make_log_line(stability, class_score, nullptr, false, reject_reason));
        continue;
      }

      feedback->track_ids.push_back(stability.track_id);
      feedback->class_scores.push_back(static_cast<float>(class_score));
      feedback->stability_scores.push_back(stability.stability_score);
      feedback->depths_m.push_back(stability.depth_median_m);
      feedback->points.push_back(point);

      if (class_score < active_min_class_score_ ||
        stability.stability_score < active_min_stability_score_)
      {
        reject_reason = class_score < active_min_class_score_ ?
          "class_score_low" : "stability_score_low";
        write_feedback_log_line(
          make_log_line(stability, class_score, &point, false, reject_reason));
        continue;
      }

      auto & candidate = candidate_summaries_[stability.track_id];
      candidate.track_id = stability.track_id;
      candidate.class_name = stability.class_name;
      candidate.bbox = bbox_from_msg(stability.detection.bbox);
      candidate.point = point;
      candidate.stamp = rclcpp::Time(stability.header.stamp);
      candidate.observations += 1;
      candidate.depths_m.push_back(stability.depth_median_m);
      candidate.class_score_sum += class_score;
      candidate.stability_score_sum += stability.stability_score;
      candidate.combined_score_sum += class_score * stability.stability_score;

      write_feedback_log_line(make_log_line(stability, class_score, &point, true, ""));
    }

    feedback->person_count = static_cast<uint32_t>(feedback->track_ids.size());
  }

  goal_handle->publish_feedback(feedback);
}

void DetectionStabilityNode::open_feedback_log(const SelectStablePerson::Goal & goal)
{
  if (!feedback_log_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(feedback_log_mutex_);
  if (feedback_log_file_.is_open()) {
    feedback_log_file_.close();
  }

  feedback_action_id_ = "action_" + current_time_for_filename();
  feedback_goal_seconds_ = goal.seconds;
  feedback_target_class_name_ = goal.class_name;

  const std::filesystem::path root_dir(feedback_log_root_dir_);
  feedback_action_output_dir_ = root_dir / feedback_action_id_;
  std::error_code error;
  std::filesystem::create_directories(feedback_action_output_dir_, error);
  if (error) {
    RCLCPP_WARN(
      this->get_logger(), "Failed to create feedback log directory %s: %s",
      feedback_action_output_dir_.string().c_str(), error.message().c_str());
    return;
  }

  feedback_log_path_ = feedback_action_output_dir_ / "feedback.csv";
  feedback_log_file_.open(feedback_log_path_, std::ios::out | std::ios::trunc);
  if (!feedback_log_file_.is_open()) {
    RCLCPP_WARN(
      this->get_logger(), "Failed to open feedback log: %s",
      feedback_log_path_.string().c_str());
    return;
  }

  feedback_log_file_
    << "action_id,goal_seconds,target_class,min_class_score,min_stability_score,"
       "stamp,frame_index,track_id,class_name,class_score,stability_score,depth_m,"
       "depth_cost,bbox_cost,id_cost,lidar_points,point_x,point_y,point_z,"
       "accepted_candidate,reject_reason\n";
  feedback_log_file_.flush();

  RCLCPP_INFO(
    this->get_logger(), "Selection feedback CSV will be saved to %s",
    feedback_log_path_.string().c_str());
}

void DetectionStabilityNode::close_feedback_log()
{
  std::lock_guard<std::mutex> lock(feedback_log_mutex_);
  if (feedback_log_file_.is_open()) {
    feedback_log_file_.flush();
    feedback_log_file_.close();
  }
}

void DetectionStabilityNode::write_feedback_log_line(const std::string & line)
{
  if (!feedback_log_enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(feedback_log_mutex_);
  if (!feedback_log_file_.is_open()) {
    return;
  }
  feedback_log_file_ << line << '\n';
}

bool DetectionStabilityNode::class_matches(const std::string & class_name) const
{
  return target_class_name_.empty() || class_name == target_class_name_;
}

double DetectionStabilityNode::extract_class_score(
  const inha_interfaces::msg::PostureAndGestureStability & stability) const
{
  if (stability.detection.results.empty()) {
    return 0.0;
  }
  return stability.detection.results.front().hypothesis.score;
}

bool DetectionStabilityNode::make_point_from_stability(
  const inha_interfaces::msg::PostureAndGestureStability & stability,
  const CameraProjection & projection,
  geometry_msgs::msg::PointStamped & point) const
{
  if (!stability.has_depth || stability.depth_median_m <= 0.0F ||
    projection.frame_id.empty() || projection.fx <= 0.0 || projection.fy <= 0.0)
  {
    return false;
  }

  const double u = stability.detection.bbox.center.position.x;
  const double v = stability.detection.bbox.center.position.y;
  const double z = stability.depth_median_m;
  if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(z)) {
    return false;
  }

  point.header = stability.header;
  if (point.header.frame_id.empty()) {
    point.header = stability.detection.header;
  }
  point.header.frame_id = projection.frame_id;
  point.point.z = z;
  point.point.x = (u - projection.cx) * z / projection.fx;
  point.point.y = (v - projection.cy) * z / projection.fy;
  return true;
}

std::optional<CandidateSummary> DetectionStabilityNode::choose_best_candidate_locked() const
{
  std::optional<CandidateSummary> best;
  double best_depth = std::numeric_limits<double>::infinity();
  double best_combined_score = -std::numeric_limits<double>::infinity();

  for (const auto & [track_id, candidate] : candidate_summaries_) {
    (void)track_id;
    if (candidate.observations < static_cast<uint32_t>(min_selection_observations_) ||
      candidate.depths_m.empty())
    {
      continue;
    }

    const double median_depth = median_copy(candidate.depths_m);
    const double combined_score =
      candidate.combined_score_sum / std::max(1U, candidate.observations);
    const bool closer = median_depth < best_depth - depth_tie_tolerance_m_;
    const bool tied_and_better =
      std::abs(median_depth - best_depth) <= depth_tie_tolerance_m_ &&
      combined_score > best_combined_score;

    if (!best || closer || tied_and_better) {
      best = candidate;
      best->selected_depth_m = median_depth;
      best_depth = median_depth;
      best_combined_score = combined_score;
      if (best->point.point.z > 0.0) {
        const double scale = median_depth / best->point.point.z;
        best->point.point.x *= scale;
        best->point.point.y *= scale;
        best->point.point.z = median_depth;
      }
    }
  }

  return best;
}

sensor_msgs::msg::CompressedImage::ConstSharedPtr DetectionStabilityNode::find_nearest_image(
  const rclcpp::Time & target_stamp)
{
  std::lock_guard<std::mutex> lock(selection_mutex_);
  if (image_buffer_.empty()) {
    return nullptr;
  }

  if (target_stamp.nanoseconds() == 0) {
    return image_buffer_.back().msg;
  }

  auto best_it = image_buffer_.begin();
  int64_t best_distance = std::numeric_limits<int64_t>::max();
  for (auto it = image_buffer_.begin(); it != image_buffer_.end(); ++it) {
    const int64_t raw_distance = (it->stamp - target_stamp).nanoseconds();
    const int64_t distance = raw_distance < 0 ? -raw_distance : raw_distance;
    if (distance < best_distance) {
      best_distance = distance;
      best_it = it;
    }
  }
  return best_it->msg;
}

bool DetectionStabilityNode::transform_point_to_output_frame(
  geometry_msgs::msg::PointStamped & point) const
{
  if (output_frame_.empty() || point.header.frame_id == output_frame_) {
    return true;
  }
  try {
    const auto timeout = tf2::durationFromSec(transform_timeout_sec_);
    point = tf_buffer_.transform(point, output_frame_, timeout);
    return true;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(
      this->get_logger(),
      "Cannot transform selected point from '%s' to '%s': %s",
      point.header.frame_id.c_str(), output_frame_.c_str(), ex.what());
    return false;
  }
}

bool DetectionStabilityNode::publish_and_save_selection(const CandidateSummary & selected)
{
  selected_point_pub_->publish(selected.point);

  const auto image_msg = find_nearest_image(selected.stamp);
  if (!image_msg) {
    RCLCPP_WARN(
      this->get_logger(),
      "Selected track %s at %.3fm, but no compressed image was buffered.",
      selected.track_id.c_str(), selected.selected_depth_m);
    return false;
  }

  const bool saved = save_selected_image(selected, image_msg);
  if (saved) {
    RCLCPP_INFO(
      this->get_logger(),
      "Selected track %s class=%s depth=%.3fm point=(%.3f, %.3f, %.3f). "
      "Saved image to %s",
      selected.track_id.c_str(), selected.class_name.c_str(), selected.selected_depth_m,
      selected.point.point.x, selected.point.point.y, selected.point.point.z,
      selected_image_path_.c_str());
  }
  return saved;
}

bool DetectionStabilityNode::save_selected_image(
  const CandidateSummary & selected,
  const sensor_msgs::msg::CompressedImage::ConstSharedPtr & image_msg) const
{
  if (!image_msg || image_msg->data.empty()) {
    return false;
  }

  const std::filesystem::path output_path(selected_image_path_);
  if (output_path.has_parent_path()) {
    std::error_code error;
    std::filesystem::create_directories(output_path.parent_path(), error);
    if (error) {
      RCLCPP_WARN(
        this->get_logger(), "Failed to create image output directory %s: %s",
        output_path.parent_path().string().c_str(), error.message().c_str());
      return false;
    }
  }

  const auto write_compressed_frame = [&]() {
      std::ofstream output(output_path.string(), std::ios::binary);
      if (!output) {
        return false;
      }
      output.write(
        reinterpret_cast<const char *>(image_msg->data.data()),
        static_cast<std::streamsize>(image_msg->data.size()));
      return static_cast<bool>(output);
    };

#ifdef DETECTION_STABILITY_HAVE_OPENCV
  try {
    cv::Mat encoded(
      1,
      static_cast<int>(image_msg->data.size()),
      CV_8UC1,
      const_cast<uint8_t *>(image_msg->data.data()));
    const cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (image.empty()) {
      RCLCPP_WARN(
        this->get_logger(),
        "Failed to decode selected compressed image. Saving the compressed frame instead.");
      return write_compressed_frame();
    }

    cv::Mat output_image = image;
    if (selected.bbox.valid) {
      const double margin_x = selected.bbox.w * crop_margin_ratio_;
      const double margin_y = selected.bbox.h * crop_margin_ratio_;
      const int x0 = std::clamp(
        static_cast<int>(std::floor(selected.bbox.min_x - margin_x)), 0, image.cols - 1);
      const int y0 = std::clamp(
        static_cast<int>(std::floor(selected.bbox.min_y - margin_y)), 0, image.rows - 1);
      const int x1 = std::clamp(
        static_cast<int>(std::ceil(selected.bbox.max_x + margin_x)), x0 + 1, image.cols);
      const int y1 = std::clamp(
        static_cast<int>(std::ceil(selected.bbox.max_y + margin_y)), y0 + 1, image.rows);
      output_image = image(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
    }

    if (!cv::imwrite(output_path.string(), output_image)) {
      RCLCPP_WARN(
        this->get_logger(),
        "Failed to write cropped selected image to %s. Saving the compressed frame instead.",
        selected_image_path_.c_str());
      return write_compressed_frame();
    }
  } catch (const cv::Exception & ex) {
    RCLCPP_WARN(
      this->get_logger(),
      "OpenCV failed while saving selected image: %s. Saving the compressed frame instead.",
      ex.what());
    return write_compressed_frame();
  }

  return true;
#else
  (void)selected;
  RCLCPP_WARN_ONCE(
    this->get_logger(),
    "OpenCV imgcodecs was not found at build time. Saving the compressed frame without cropping.");
  return write_compressed_frame();
#endif
}

BBox DetectionStabilityNode::make_search_bbox(
  const std::vector<DetectionContext> & contexts) const
{
  BBox search_bbox;
  search_bbox.min_x = std::numeric_limits<double>::infinity();
  search_bbox.min_y = std::numeric_limits<double>::infinity();
  search_bbox.max_x = -std::numeric_limits<double>::infinity();
  search_bbox.max_y = -std::numeric_limits<double>::infinity();

  for (const auto & context : contexts) {
    search_bbox.min_x = std::min(search_bbox.min_x, context.bbox.min_x);
    search_bbox.min_y = std::min(search_bbox.min_y, context.bbox.min_y);
    search_bbox.max_x = std::max(search_bbox.max_x, context.bbox.max_x);
    search_bbox.max_y = std::max(search_bbox.max_y, context.bbox.max_y);
  }

  search_bbox.w = std::max(0.0, search_bbox.max_x - search_bbox.min_x);
  search_bbox.h = std::max(0.0, search_bbox.max_y - search_bbox.min_y);
  search_bbox.cx = search_bbox.min_x + 0.5 * search_bbox.w;
  search_bbox.cy = search_bbox.min_y + 0.5 * search_bbox.h;
  search_bbox.valid = search_bbox.w > 0.0 && search_bbox.h > 0.0;
  return search_bbox;
}

ProjectedLidar DetectionStabilityNode::project_lidar_once(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
  const CloudMsg::ConstSharedPtr & cloud,
  const std::string & detection_frame,
  const rclcpp::Time & target_stamp,
  const BBox & search_bbox)
{
  ProjectedLidar projected_lidar;
  if (!camera_info) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "CameraInfo is not available. Publishing depth_cost=1.0.");
    return projected_lidar;
  }

  if (!cloud) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "LiDAR input is not available. Publishing depth_cost=1.0.");
    return projected_lidar;
  }

  const auto projection = make_projection(camera_info, detection_frame);
  if (projection.frame_id.empty() || projection.fx <= 0.0 || projection.fy <= 0.0) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Camera projection is invalid. Check CameraInfo K and frame_id.");
    return projected_lidar;
  }

  const auto lidar_frame = cloud->header.frame_id;
  const rclcpp::Time lidar_stamp(cloud->header.stamp);
  if (max_lidar_age_sec_ > 0.0 && target_stamp.nanoseconds() > 0 &&
    lidar_stamp.nanoseconds() > 0)
  {
    const double lidar_age = std::abs((target_stamp - lidar_stamp).seconds());
    if (lidar_age > max_lidar_age_sec_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 3000,
        "Latest LiDAR message is %.3fs away from detection stamp. Publishing depth_cost=1.0.",
        lidar_age);
      return projected_lidar;
    }
  }

  RigidTransform transform;
  if (!lookup_lidar_to_camera_transform(
      projection.frame_id, lidar_frame, target_stamp, transform))
  {
    return projected_lidar;
  }

  projected_lidar.valid = true;
  if (max_projected_lidar_points_ > 0) {
    projected_lidar.points.reserve(static_cast<size_t>(max_projected_lidar_points_));
  }

  append_projected_cloud_points(*cloud, projection, transform, search_bbox, projected_lidar.points);

  std::sort(
    projected_lidar.points.begin(), projected_lidar.points.end(),
    [](const ProjectedPoint & lhs, const ProjectedPoint & rhs) {
      return lhs.u < rhs.u;
    });
  return projected_lidar;
}

DepthStats DetectionStabilityNode::compute_depth_stats(
  const BBox & bbox,
  const ProjectedLidar & projected_lidar) const
{
  DepthStats stats;
  if (!projected_lidar.valid) {
    return stats;
  }

  // Focus on the upper-center body region to reduce background/outlier points.
  const double core_min_x = bbox.cx - (bbox.w * 0.25);
  const double core_max_x = bbox.cx + (bbox.w * 0.25);
  const double core_min_y = bbox.min_y;
  const double core_max_y = bbox.min_y + (bbox.h * 0.60);

  std::vector<double> depths;
  depths.reserve(128);
  auto begin = std::lower_bound(
    projected_lidar.points.begin(), projected_lidar.points.end(), core_min_x,
    [](const ProjectedPoint & point, const double u) {
      return point.u < u;
    });

  for (auto it = begin; it != projected_lidar.points.end() && it->u <= core_max_x; ++it) {
    if (it->v >= core_min_y && it->v <= core_max_y) {
      depths.push_back(it->depth_m);
    }
  }

  if (static_cast<int>(depths.size()) < min_lidar_points_) {
    stats.point_count = static_cast<uint32_t>(depths.size());
    return stats;
  }

  const double median = median_in_place(depths);
  std::vector<double> deviations;
  deviations.reserve(depths.size());
  for (const auto depth : depths) {
    deviations.push_back(std::abs(depth - median));
  }

  stats.has_depth = true;
  stats.median_m = median;
  stats.mad_m = median_in_place(deviations);
  stats.point_count = static_cast<uint32_t>(depths.size());
  return stats;
}

CameraProjection DetectionStabilityNode::make_projection(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
  const std::string & detection_frame) const
{
  CameraProjection projection;
  projection.frame_id = camera_frame_;
  if (projection.frame_id.empty()) {
    projection.frame_id = camera_info->header.frame_id;
  }
  if (projection.frame_id.empty()) {
    projection.frame_id = detection_frame;
  }

  projection.fx = camera_info->k[0];
  projection.fy = camera_info->k[4];
  projection.cx = camera_info->k[2];
  projection.cy = camera_info->k[5];
  projection.width = camera_info->width;
  projection.height = camera_info->height;
  return projection;
}

bool DetectionStabilityNode::lookup_lidar_to_camera_transform(
  const std::string & camera_frame,
  const std::string & lidar_frame,
  const rclcpp::Time & target_stamp,
  RigidTransform & transform)
{
  try {
    const auto timeout = rclcpp::Duration::from_nanoseconds(
      static_cast<int64_t>(transform_timeout_sec_ * 1e9));
    const auto tf_msg =
      tf_buffer_.lookupTransform(camera_frame, lidar_frame, target_stamp, timeout);

    const auto & q_msg = tf_msg.transform.rotation;
    const tf2::Quaternion q(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
    transform.rotation = tf2::Matrix3x3(q);
    transform.tx = tf_msg.transform.translation.x;
    transform.ty = tf_msg.transform.translation.y;
    transform.tz = tf_msg.transform.translation.z;
    return true;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 3000,
      "Cannot transform LiDAR frame '%s' to camera frame '%s': %s",
      lidar_frame.c_str(), camera_frame.c_str(), ex.what());
    return false;
  }
}

void DetectionStabilityNode::append_projected_cloud_points(
  const CloudMsg & cloud,
  const CameraProjection & projection,
  const RigidTransform & transform,
  const BBox & search_bbox,
  std::vector<ProjectedPoint> & projected_points)
{
  try {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

    size_t point_index = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if ((point_index++ % static_cast<size_t>(lidar_point_step_)) != 0) {
        continue;
      }

      append_projected_point(
        static_cast<double>(*iter_x),
        static_cast<double>(*iter_y),
        static_cast<double>(*iter_z),
        projection, transform, search_bbox, projected_points);

      if (max_projected_lidar_points_ > 0 &&
        projected_points.size() >= static_cast<size_t>(max_projected_lidar_points_))
      {
        break;
      }
    }
  } catch (const std::runtime_error & ex) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "PointCloud2 input must contain float x/y/z fields: %s", ex.what());
  }
}

void DetectionStabilityNode::append_projected_point(
  const double x,
  const double y,
  const double z,
  const CameraProjection & projection,
  const RigidTransform & transform,
  const BBox & search_bbox,
  std::vector<ProjectedPoint> & projected_points) const
{
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    return;
  }

  const auto & r = transform.rotation;
  const double camera_x = r[0][0] * x + r[0][1] * y + r[0][2] * z + transform.tx;
  const double camera_y = r[1][0] * x + r[1][1] * y + r[1][2] * z + transform.ty;
  const double camera_z = r[2][0] * x + r[2][1] * y + r[2][2] * z + transform.tz;

  if (camera_z < min_depth_m_ || camera_z > max_depth_m_) {
    return;
  }

  const double u = projection.fx * camera_x / camera_z + projection.cx;
  const double v = projection.fy * camera_y / camera_z + projection.cy;
  if (!std::isfinite(u) || !std::isfinite(v)) {
    return;
  }
  if (u < search_bbox.min_x || u > search_bbox.max_x ||
    v < search_bbox.min_y || v > search_bbox.max_y)
  {
    return;
  }
  if (projection.width > 0 && (u < 0.0 || u >= static_cast<double>(projection.width))) {
    return;
  }
  if (projection.height > 0 && (v < 0.0 || v >= static_cast<double>(projection.height))) {
    return;
  }

  projected_points.push_back({u, v, camera_z});
}

double DetectionStabilityNode::compute_depth_cost(
  const DepthStats & stats,
  const TrackState * prior) const
{
  if (!stats.has_depth) {
    return 1.0;
  }

  const double relative_spread = stats.mad_m / std::max(stats.median_m, 0.10);
  const double spread_cost = clamp01(relative_spread / depth_spread_tolerance_);

  double weighted_cost = depth_spread_weight_ * spread_cost;
  double weight_sum = depth_spread_weight_;
  if (prior != nullptr && prior->has_depth) {
    const double depth_jump = std::abs(stats.median_m - prior->depth_median_m);
    const double jump_cost = clamp01(depth_jump / depth_jump_tolerance_m_);
    weighted_cost += depth_jump_weight_ * jump_cost;
    weight_sum += depth_jump_weight_;
  }

  if (weight_sum <= 0.0) {
    return spread_cost;
  }
  return clamp01(weighted_cost / weight_sum);
}

double DetectionStabilityNode::compute_bbox_cost(
  const BBox & bbox,
  const TrackState * prior) const
{
  if (prior == nullptr || !prior->has_bbox) {
    return 1.0;
  }

  const double center_distance =
    std::hypot(bbox.cx - prior->bbox.cx, bbox.cy - prior->bbox.cy);
  const double previous_diagonal = std::max(1.0, std::hypot(prior->bbox.w, prior->bbox.h));
  const double center_cost =
    clamp01((center_distance / previous_diagonal) / bbox_center_tolerance_);

  const double area = std::max(1.0, bbox.w * bbox.h);
  const double previous_area = std::max(1.0, prior->bbox.w * prior->bbox.h);
  const double size_cost =
    clamp01(std::abs(std::log(area / previous_area)) / bbox_size_tolerance_);

  const double weight_sum = bbox_center_weight_ + bbox_size_weight_;
  if (weight_sum <= 0.0) {
    return std::max(center_cost, size_cost);
  }
  return clamp01(
    (bbox_center_weight_ * center_cost + bbox_size_weight_ * size_cost) / weight_sum);
}

double DetectionStabilityNode::compute_id_cost(
  const std::string & track_id,
  const BBox & bbox,
  const TrackState * prior) const
{
  double maturity_cost = 1.0;
  double reappearance_cost = 0.0;
  if (prior != nullptr) {
    maturity_cost = clamp01(1.0 - static_cast<double>(prior->age) / stable_age_frames_);
    reappearance_cost = clamp01(static_cast<double>(prior->missed_frames) / stable_age_frames_);
  }

  double switch_cost = 0.0;
  for (const auto & [other_id, other_state] : tracks_) {
    if (other_id == track_id || !other_state.has_bbox) {
      continue;
    }
    if (other_state.last_seen_frame >= frame_index_) {
      continue;
    }
    const auto frames_since_seen = frame_index_ - other_state.last_seen_frame;
    if (frames_since_seen > static_cast<uint64_t>(id_switch_memory_frames_)) {
      continue;
    }

    const double iou = bbox_iou(bbox, other_state.bbox);
    if (iou > id_switch_iou_threshold_) {
      switch_cost = std::max(switch_cost, iou);
    }
  }

  return clamp01(std::max({maturity_cost, reappearance_cost, switch_cost}));
}

double DetectionStabilityNode::compute_total_cost(
  const double depth_cost,
  const double bbox_cost,
  const double id_cost) const
{
  const double weight_sum = total_depth_weight_ + total_bbox_weight_ + total_id_weight_;
  if (weight_sum <= 0.0) {
    return clamp01((depth_cost + bbox_cost + id_cost) / 3.0);
  }

  return clamp01(
    (total_depth_weight_ * depth_cost +
    total_bbox_weight_ * bbox_cost +
    total_id_weight_ * id_cost) /
    weight_sum);
}

void DetectionStabilityNode::mark_missing_tracks(
  const std::unordered_set<std::string> & seen_tracks)
{
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (seen_tracks.find(it->first) == seen_tracks.end()) {
      ++it->second.missed_frames;
      if (it->second.missed_frames > static_cast<uint32_t>(max_missing_frames_)) {
        it = tracks_.erase(it);
        continue;
      }
    }
    ++it;
  }
}

std::string DetectionStabilityNode::extract_track_id(const std::string & detection_id) const
{
  const std::string prefix = "track:";
  const auto begin = detection_id.find(prefix);
  if (begin == std::string::npos) {
    return detection_id;
  }

  const auto value_begin = begin + prefix.size();
  const auto value_end = detection_id.find(';', value_begin);
  if (value_end == std::string::npos) {
    return detection_id.substr(value_begin);
  }
  return detection_id.substr(value_begin, value_end - value_begin);
}

std::string DetectionStabilityNode::extract_class_name(
  const vision_msgs::msg::Detection2D & detection) const
{
  if (detection.results.empty()) {
    return "";
  }
  return detection.results.front().hypothesis.class_id;
}

}  // namespace detection_stability

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<detection_stability::DetectionStabilityNode>());
  rclcpp::shutdown();
  return 0;
}
