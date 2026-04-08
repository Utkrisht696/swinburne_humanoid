cmake_minimum_required(VERSION 3.8)
project(asr_vosk_language_model_en_us_gigaspeech)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

set(VOSK_MODEL_ARCHIVE "/home/agi/models/vosk-model-en-us-0.42-gigaspeech.zip"
  CACHE FILEPATH "Path to the local Vosk GigaSpeech model archive")

if(NOT EXISTS "${VOSK_MODEL_ARCHIVE}")
  message(FATAL_ERROR
    "VOSK model archive not found: ${VOSK_MODEL_ARCHIVE}\n"
    "Set -DVOSK_MODEL_ARCHIVE=/path/to/vosk-model-en-us-0.42-gigaspeech.zip")
endif()

set(MODEL_ARCHIVE_BASENAME "vosk-model-en-us-0.42-gigaspeech.zip")
set(MODEL_EXTRACTED_DIR "vosk-model-en-us-0.42-gigaspeech")
set(MODEL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/model")
set(MODEL_STAMP "${CMAKE_CURRENT_BINARY_DIR}/.model_unpacked")

add_custom_command(
  OUTPUT "${MODEL_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${MODEL_BUILD_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_EXTRACTED_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VOSK_MODEL_ARCHIVE}" "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_ARCHIVE_BASENAME}"
  COMMAND ${CMAKE_COMMAND} -E tar xvf "${MODEL_ARCHIVE_BASENAME}"
  COMMAND ${CMAKE_COMMAND} -E rename "${CMAKE_CURRENT_BINARY_DIR}/${MODEL_EXTRACTED_DIR}" "${MODEL_BUILD_DIR}"
  COMMAND ${CMAKE_COMMAND} -E touch "${MODEL_STAMP}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${VOSK_MODEL_ARCHIVE}"
  COMMENT "Unpacking ${MODEL_ARCHIVE_BASENAME}"
  VERBATIM
)

add_custom_target(unpack_model ALL DEPENDS "${MODEL_STAMP}")

install(
  DIRECTORY
    "${MODEL_BUILD_DIR}"
    config
  DESTINATION share/${PROJECT_NAME}
)

ament_index_register_resource("asr.vosk.model" CONTENT "config/vosk_model.yml")

ament_package()
