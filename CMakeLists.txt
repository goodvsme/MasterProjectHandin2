cmake_minimum_required(VERSION 3.5)
project(master_project2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rosidl_default_generators REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY config description launch worlds scripts maps srv models db
  DESTINATION share/${PROJECT_NAME}
)

install(
    PROGRAMS scripts/__init__.py 
    scripts/map_cleaner.py 
    scripts/database_handler.py 
    scripts/test_flowgrid.py 
    scripts/waypoints.py 
    scripts/multibot_path_planner.py
    scripts/multibot_test.py
    scripts/test.py 
    scripts/Astar_pathplanner.py 
    scripts/image_flowgrid.py
    scripts/waypoint_maker.py 
    scripts/spawn_humans.py 
    scripts/yolo_visualizer.py 
    scripts/flowgrid.py 
    scripts/map_update_depth.py 
    scripts/map_update_lidar.py 
    scripts/waypoint_updater.py
    scripts/multibot_yolo_visualizer.py
    scripts/mission_monitor.py
    scripts/mission_completion_handler.py
    scripts/multibot_coordinator.py
    scripts/spawn_boxes.py
    scripts/multibot_human_detection_aggregator.py  
    scripts/multibot_flowgrid.py    
    scripts/multibot_waypoint_updater.py
    scripts/scan_collector.py
    scripts/map_processor.py
    scripts/test_human.py
  DESTINATION lib/${PROJECT_NAME}
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AddWaypoint.srv"
  "srv/RemoveWaypoint.srv"
  "srv/UpdateWaypointCost.srv"
  "srv/NavigateToWaypoint.srv"
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()