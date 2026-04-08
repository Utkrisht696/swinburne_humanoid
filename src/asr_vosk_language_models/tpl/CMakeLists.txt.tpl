cmake_minimum_required(VERSION 3.8)
project(asr_vosk_language_model_{{ locale.lower() }}_{{ size }})

if(NOT EXISTS ${CMAKE_BINARY_DIR}/{{ model }})
    file(DOWNLOAD {{ url }}
        ${CMAKE_BINARY_DIR}/{{ model }}
        SHOW_PROGRESS
    )
endif()

string(REGEX REPLACE "\\.[^.]*$" "" MODEL_BASE_NAME {{ model }})  # strips the suffix
add_custom_target(unpacked_model ALL)
add_custom_command(
    TARGET unpacked_model
    COMMAND ${CMAKE_COMMAND} -E remove_directory model
    COMMAND ${CMAKE_COMMAND} -E tar xzf {{ model }} ${MODEL_BASE_NAME}
    COMMAND ${CMAKE_COMMAND} -E rename ${MODEL_BASE_NAME} model
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS ${CMAKE_BINARY_DIR}/{{ model }}
    COMMENT "Unpacking {{ model }}"
    VERBATIM
)

install(DIRECTORY
    ${CMAKE_BINARY_DIR}/model config
    DESTINATION share/${PROJECT_NAME}
)


# manual ament_index registration
file(WRITE ${CMAKE_BINARY_DIR}/ament_marker_package "")
install(FILES ${CMAKE_BINARY_DIR}/ament_marker_package
    DESTINATION share/ament_index/resource_index/packages
    RENAME ${PROJECT_NAME})

file(WRITE ${CMAKE_BINARY_DIR}/ament_marker_model "config/vosk_model.yml")
install(FILES ${CMAKE_BINARY_DIR}/ament_marker_model
    DESTINATION share/ament_index/resource_index/asr.vosk.model
    RENAME ${PROJECT_NAME})


