^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package vosk_asr
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.5.1 (2025-07-09)
------------------
* Add diagnostic analyzers
* Contributors: Noel Jimenez

2.5.0 (2025-06-26)
------------------
* Do not run the model if empty message is received and there is not currently and incremental
* add rosbag2 default plugins dependency
* remove asr_vosk_with_args.launch.py
* Contributors: Luka Juricic, ferrangebelli

2.4.0 (2024-11-04)
------------------
* use ament index for localized nodes
* Contributors: Luka Juricic

2.3.2 (2024-08-19)
------------------
* rename diagnostics msg to match documentation (and diagnostic_aggregator) categories
* Contributors: Séverin Lemaignan

2.3.1 (2024-08-19)
------------------
* update diagnostics to use 'Communication' category + export module name and lifecycle state
* [launch] disable the automatic transition to 'active' after initial launch
  Note that this indirectly cause a crash in asr_vosk when setting the lifecycle state
  to 'deactivate'. This will need to be addressed separately.
* Contributors: Séverin Lemaignan

2.3.0 (2024-08-07)
------------------
* clarify message explaining the purpose of the pal_node_vosk.py wrapper
* switch from ament_virtualenv to vosk debian
* Contributors: Luka Juricic, Séverin Lemaignan

2.2.3 (2024-07-24)
------------------
* update dependencies
* Contributors: Luka Juricic

2.2.2 (2024-07-19)
------------------
* republish anonymous voice id for latecomers
* Contributors: Luka Juricic

2.2.1 (2024-07-18)
------------------

2.2.0 (2024-07-17)
------------------
* add PAL configuration compliant launch file
* Contributors: Luka Juricic

2.1.0 (2024-07-11)
------------------
* Rework speech API
  - rename tts package to tts_engine and tts_plugin\_*
  - select model by family name
  - update i18n support
  - additional diagnostic info
  - rename parameter locale to default_locale
  - use robot_speaking topic
* Contributors: Luka Juricic

2.0.0 (2024-05-21)
------------------
* port to humble; rename package to asr_vosk
* vosk_tutorial.py: remove deprecated StartASR/StopASR
* update README
* Contributors: Luka Juricic, Séverin Lemaignan

0.2.2 (2023-05-23)
------------------
* {language_center_msgs -> i18n_msgs}
* Contributors: Séverin Lemaignan

0.2.1 (2023-05-16)
------------------
* remove un-needed catkin deps + update vosk to latest
* remove actions start_asr/stop_asr
  To start or stop the ASR, better to pal-start or pal-stop the startup
* add action /asr/set_locale
* publish diagnostics
* do not hard-code vosk_language_models path
  While here, better handle errors during initialization
* Contributors: Séverin Lemaignan

0.2.0 (2023-05-15)
------------------
* add CMake workaround to support pal_deploy with catkin-venv
* workaround https://github.com/locusrobotics/catkin_virtualenv/pull/89
* use catkin venv to add a dependency on vosk pip
* cleanly stop the node, even when no audio is published + avoid busy waits
* Contributors: Séverin Lemaignan

0.1.11 (2023-05-02)
-------------------
* publish final ASR result when speaker stops speaking
* Contributors: Séverin Lemaignan

0.1.10 (2023-04-14)
-------------------
* do not explicitely set the default language in launch file
  -> otherwise, impossible to set the value to a different default, eg for a specific customer
* Contributors: Séverin Lemaignan

0.1.9 (2023-04-14)
------------------
* publish /humans/voices/tracked + minor cleanup
* Contributors: Séverin Lemaignan

0.1.8 (2023-03-09)
------------------
* immediately start listening
  before, /start_asr/goal had to be called first
  While here, minor code improvmeent
* Contributors: Séverin Lemaignan

0.1.7 (2023-03-06)
------------------
* do not spam the console when no text is recognised
* Contributors: Séverin Lemaignan

0.1.6 (2023-03-06)
------------------
* {->audio}/voice_detected
* Contributors: Séverin Lemaignan

0.1.5 (2023-03-06)
------------------
* minor: code readability
* VAD now published on /voice_detected instead of /is_speeching
* Contributors: Séverin Lemaignan

0.1.4 (2023-01-24)
------------------
* refactor model loading
  - improved model size selection (added ROS param)
  - improved fallback mechanisms
  - use pathlib instead of os.path
* Contributors: Séverin Lemaignan

0.1.3 (2023-01-20)
------------------
* update path to default location for language models
* remove small en_US model from repo
  The model is available in package vosk-language-model-en-us-small
* [doc]
* Contributors: Séverin Lemaignan

0.1.2 (2023-01-17)
------------------
* fix some default path for the vosk docker image
* Contributors: Séverin Lemaignan

0.1.1 (2023-01-17)
------------------
* Start/StopASR actions are in hri_actions_msgs
* Contributors: Séverin Lemaignan

0.1.0 (2023-01-17)
------------------
* remove docker bash script
* Update README.md
* missing function and updating docker call
* channel input topic
* tutorial example, change ROS action type
* reset model path
* convert srvs to actions
* remove unused service
* Contributors: saracooper

0.0.2 (2022-11-17)
------------------
* 0.0.1
* install launch and configs
* Merge branch 'gallium' into 'main'
  Gallium
  See merge request interaction/vosk_asr!3
* check right directory of lang models
* example config of gallium lang models
* Merge branch 'multiple-languages' into 'main'
  Multiple languages
  See merge request interaction/vosk_asr!2
* add services to start and stop processing
* add options to switch language
* install config and launch files
* change model path
* Merge branch 'master' into 'main'
  ROS based vosk recognizer wrapper
  See merge request interaction/vosk_asr!1
* enable word by word publishing
* modify service names, typos
* not publish if tts running
* fix suggested changes
* Merge branch 'main' into 'master'
  # Conflicts:
  #   README.md
* add new rosbag data
* modify model paths
* ROS based vosk recognizer wrapper
* Initial commit
* Contributors: Sara Cooper, Séverin Lemaignan, saracooper
