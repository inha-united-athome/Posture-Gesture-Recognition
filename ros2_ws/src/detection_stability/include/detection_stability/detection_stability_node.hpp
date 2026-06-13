#ifndef DETECTION_STABILITY__DETECTION_STABILITY_NODE_HPP_
#define DETECTION_STABILITY__DETECTION_STABILITY_NODE_HPP_

#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "inha_interfaces/action/select_stable_person.hpp"
#include "inha_interfaces/msg/posture_and_gesture_stability_array.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace detection_stability
{

struct BBox
{
  double cx{0.0};
  double cy{0.0};
  double w{0.0};
  double h{0.0};
  double min_x{0.0};
  double min_y{0.0};
  double max_x{0.0};
  double max_y{0.0};
  bool valid{false};
};

struct DepthStats
{
  bool has_depth{false};
  double median_m{0.0};
  double mad_m{0.0};
  uint32_t point_count{0};
};

struct TrackState
{
  bool has_bbox{false};
  bool has_depth{false};
  BBox bbox;
  double depth_median_m{0.0};
  uint32_t age{0};
  uint32_t missed_frames{0};
  uint64_t last_seen_frame{0};
};

struct CameraProjection
{
  std::string frame_id;
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  uint32_t width{0};
  uint32_t height{0};
};

struct RigidTransform
{
  tf2::Matrix3x3 rotation;
  double tx{0.0};
  double ty{0.0};
  double tz{0.0};
};

struct ProjectedPoint
{
  double u{0.0};
  double v{0.0};
  double depth_m{0.0};
};

struct ProjectedLidar
{
  bool valid{false};
  std::vector<ProjectedPoint> points;
};

struct DetectionContext
{
  size_t index{0};
  BBox bbox;
  std::string track_id;
  std::string class_name;
};

struct CandidateSummary
{
  std::string track_id;
  std::string class_name;
  BBox bbox;
  geometry_msgs::msg::PointStamped point;
  rclcpp::Time stamp;
  uint32_t observations{0};
  std::vector<double> depths_m;
  double class_score_sum{0.0};
  double stability_score_sum{0.0};
  double combined_score_sum{0.0};
  double selected_depth_m{0.0};
};

struct ImageFrame
{
  rclcpp::Time stamp;
  sensor_msgs::msg::CompressedImage::ConstSharedPtr msg;
};

struct InstanceMaskFrame
{
  rclcpp::Time stamp;
  uint32_t width{0};
  uint32_t height{0};
  std::vector<uint16_t> labels;
};

class DetectionStabilityNode : public rclcpp::Node
{
public:
  using DetectionMsg = vision_msgs::msg::Detection2DArray;
  using CloudMsg = sensor_msgs::msg::PointCloud2;
  using SelectStablePerson = inha_interfaces::action::SelectStablePerson;
  using GoalHandleSelectStablePerson =
    rclcpp_action::ServerGoalHandle<SelectStablePerson>;
  using CloudSyncPolicy =
    message_filters::sync_policies::ApproximateTime<DetectionMsg, CloudMsg>;

  DetectionStabilityNode();

private:
  rclcpp_action::GoalResponse handle_select_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const SelectStablePerson::Goal> goal);
  rclcpp_action::CancelResponse handle_select_cancel(
    const std::shared_ptr<GoalHandleSelectStablePerson> goal_handle);
  void handle_select_accepted(
    const std::shared_ptr<GoalHandleSelectStablePerson> goal_handle);
  void execute_select_goal(
    const std::shared_ptr<GoalHandleSelectStablePerson> goal_handle);

  void sanitize_parameters();
  void build_manual_camera_info();
  void start_input_subscriptions();
  void stop_input_subscriptions();
  void request_stop_selection();
  void reset_selection_state(const SelectStablePerson::Goal & goal);
  void on_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
  void on_image(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg);
  void on_instance_mask(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void on_point_cloud_pair(
    const DetectionMsg::ConstSharedPtr & detections,
    const CloudMsg::ConstSharedPtr & cloud);
  void process_detections(
    const DetectionMsg::ConstSharedPtr & msg,
    const CloudMsg::ConstSharedPtr & cloud);

  BBox make_search_bbox(const std::vector<DetectionContext> & contexts) const;
  ProjectedLidar project_lidar_once(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
    const CloudMsg::ConstSharedPtr & cloud,
    const std::string & detection_frame,
    const rclcpp::Time & target_stamp,
    const BBox & search_bbox);
  DepthStats compute_depth_stats(
    const BBox & bbox,
    const ProjectedLidar & projected_lidar,
    const InstanceMaskFrame * instance_mask = nullptr) const;
  CameraProjection make_projection(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
    const std::string & detection_frame) const;
  bool lookup_lidar_to_camera_transform(
    const std::string & camera_frame,
    const std::string & lidar_frame,
    const rclcpp::Time & target_stamp,
    RigidTransform & transform);
  bool transform_point_to_output_frame(
    geometry_msgs::msg::PointStamped & point) const;

  void append_projected_cloud_points(
    const CloudMsg & cloud,
    const CameraProjection & projection,
    const RigidTransform & transform,
    const BBox & search_bbox,
    std::vector<ProjectedPoint> & projected_points);
  void append_projected_point(
    double x,
    double y,
    double z,
    const CameraProjection & projection,
    const RigidTransform & transform,
    const BBox & search_bbox,
    std::vector<ProjectedPoint> & projected_points) const;

  double compute_depth_cost(const DepthStats & stats, const TrackState * prior) const;
  double compute_bbox_cost(const BBox & bbox, const TrackState * prior) const;
  double compute_id_cost(
    const std::string & track_id,
    const BBox & bbox,
    const TrackState * prior) const;
  double compute_total_cost(double depth_cost, double bbox_cost, double id_cost) const;
  void mark_missing_tracks(const std::unordered_set<std::string> & seen_tracks);
  std::string extract_track_id(const std::string & detection_id) const;
  std::string extract_class_name(const vision_msgs::msg::Detection2D & detection) const;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr current_camera_info();
  std::optional<InstanceMaskFrame> current_instance_mask(
    const rclcpp::Time & target_stamp) const;

  void update_active_selection(
    const inha_interfaces::msg::PostureAndGestureStabilityArray & output);
  bool class_matches(const std::string & class_name) const;
  double extract_class_score(
    const inha_interfaces::msg::PostureAndGestureStability & stability) const;
  bool make_point_from_stability(
    const inha_interfaces::msg::PostureAndGestureStability & stability,
    const CameraProjection & projection,
    geometry_msgs::msg::PointStamped & point) const;
  std::optional<CandidateSummary> choose_best_candidate_locked() const;
  sensor_msgs::msg::CompressedImage::ConstSharedPtr find_nearest_image(
    const rclcpp::Time & target_stamp);
  bool publish_and_save_selection(const CandidateSummary & selected);
  bool save_selected_image(
    const CandidateSummary & selected,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr & image_msg) const;
  void open_feedback_log(const SelectStablePerson::Goal & goal);
  void close_feedback_log();
  void write_feedback_log_line(const std::string & line);

  std::string detections_topic_;
  std::string camera_info_topic_;
  std::string lidar_topic_;
  std::string output_topic_;
  std::string image_topic_;
  std::string instance_mask_topic_;
  std::string selected_point_topic_;
  std::string selected_image_path_;
  std::string action_name_;
  std::string camera_frame_;
  std::string output_frame_;
  bool feedback_log_enabled_{true};
  std::string feedback_log_root_dir_;


  int sync_queue_size_{10};
  double max_lidar_age_sec_{0.25};
  double transform_timeout_sec_{0.05};
  double min_depth_m_{0.10};
  double max_depth_m_{30.0};
  int min_lidar_points_{5};
  int lidar_point_step_{2};
  int max_projected_lidar_points_{4000};
  bool use_instance_mask_depth_{true};
  double max_instance_mask_age_sec_{0.50};
  double instance_depth_cluster_tolerance_m_{0.30};

  int manual_camera_width_{640};
  int manual_camera_height_{480};
  double manual_camera_fx_{605.784249722};
  double manual_camera_fy_{604.492919921875};
  double manual_camera_cx_{323.7266845};
  double manual_camera_cy_{250.73828125};

  double depth_spread_tolerance_{0.08};
  double depth_jump_tolerance_m_{0.40};
  double depth_spread_weight_{0.70};
  double depth_jump_weight_{0.30};

  double bbox_center_tolerance_{0.35};
  double bbox_size_tolerance_{0.35};
  double bbox_center_weight_{0.70};
  double bbox_size_weight_{0.30};

  int stable_age_frames_{10};
  int max_missing_frames_{30};
  int id_switch_memory_frames_{5};
  double id_switch_iou_threshold_{0.50};

  double total_depth_weight_{0.45};
  double total_bbox_weight_{0.35};
  double total_id_weight_{0.20};

  double default_min_class_score_{0.50};
  double default_min_stability_score_{0.70};
  double depth_tie_tolerance_m_{0.25};
  double crop_margin_ratio_{0.15};
  int min_selection_observations_{1};
  int image_buffer_size_{30};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr instance_mask_sub_;
  rclcpp::Publisher<inha_interfaces::msg::PostureAndGestureStabilityArray>::SharedPtr stability_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr selected_point_pub_;
  rclcpp_action::Server<SelectStablePerson>::SharedPtr select_action_server_;

  message_filters::Subscriber<DetectionMsg> filtered_detections_sub_;
  message_filters::Subscriber<CloudMsg> filtered_cloud_sub_;
  std::unique_ptr<message_filters::Synchronizer<CloudSyncPolicy>> cloud_sync_;

  std::mutex input_mutex_;
  bool inputs_active_{false};

  std::mutex camera_info_mutex_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr latest_camera_info_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr manual_camera_info_;

  std::mutex selection_mutex_;
  std::shared_ptr<GoalHandleSelectStablePerson> active_goal_;
  bool selection_active_{false};
  bool stop_selection_requested_{false};
  std::string target_class_name_;
  std::vector<std::string> target_class_names_;
  float active_min_class_score_{0.0F};
  float active_min_stability_score_{0.0F};
  std::unordered_map<std::string, CandidateSummary> candidate_summaries_;
  std::deque<ImageFrame> image_buffer_;

  mutable std::mutex instance_mask_mutex_;
  std::optional<InstanceMaskFrame> latest_instance_mask_;

  std::mutex feedback_log_mutex_;
  std::ofstream feedback_log_file_;
  std::string feedback_action_id_;
  float feedback_goal_seconds_{0.0F};
  std::string feedback_target_class_name_;
  std::filesystem::path feedback_action_output_dir_;
  std::filesystem::path feedback_log_path_;

  uint64_t frame_index_{0};
  std::unordered_map<std::string, TrackState> tracks_;
};

}  // namespace detection_stability

#endif  // DETECTION_STABILITY__DETECTION_STABILITY_NODE_HPP_
