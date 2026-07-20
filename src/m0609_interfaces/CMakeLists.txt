cmake_minimum_required(VERSION 3.8)
project(m0609_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ClassifyShoes.srv"
)

ament_package()
