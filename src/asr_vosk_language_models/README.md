Vosk language models
--------------------

This repository contains a script ([generate_language_packages.py](generate_language_packages.py))
that creates a ROS package per available language.
These packages install the related language model for [vosk_asr](https://github.com/ros4hri/asr_vosk).

Models are downloaded at build time from [here](https://alphacephei.com/vosk/models).
The downloaded models are cached in the build directory.

Note that many languages have a *small* model (~40MB) and a *large* model (~1GB).
Currently ROS package generation for large models is disabled.
