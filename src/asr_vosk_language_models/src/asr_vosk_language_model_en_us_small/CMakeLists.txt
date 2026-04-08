cmake_minimum_required(VERSION 3.8)
project(asr_vosk_language_model_en_us_small)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

set(MODEL_ARCHIVE_NAME "vosk-model-small-en-us-0.15.zip")
set(MODEL_URL "https://alphacephei.com/vosk/models/${MODEL_ARCHIVE_NAME}")
set(MODEL_EXTRACTED_DIR "vosk-model-small-en-us-0.15")
set(MODEL_ARCHIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_ARCHIVE_NAME}")
set(MODEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/model")
set(MODEL_STAMP "${CMAKE_CURRENT_BINARY_DIR}/.model_unpacked")

if(NOT EXISTS "${MODEL_ARCHIVE_PATH}")
  file(DOWNLOAD "${MODEL_URL}" "${MODEL_ARCHIVE_PATH}" SHOW_PROGRESS)
endif()

add_custom_command(
  OUTPUT "${MODEL_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${MODEL_BUILD_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_EXTRACTED_DIR}"
  COMMAND ${CMAKE_COMMAND} -E tar xvf "${MODEL_ARCHIVE_NAME}"
  COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_EXTRACTED_DIR}" "${MODEL_BUILD_DIR}"
  COMMAND ${CMAKE_COMMAND} -E touch "${MODEL_STAMP}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${MODEL_ARCHIVE_PATH}"
  COMMENT "Unpacking ${MODEL_ARCHIVE_NAME}"
  VERBATIM
)

add_custom_target(unpack_model ALL DEPENDS "${MODEL_STAMP}")

install(DIRECTORY
  "${MODEL_BUILD_DIR}"
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_index_register_resource("asr.vosk.model" CONTENT "config/vosk_model.yml")

ament_package()
