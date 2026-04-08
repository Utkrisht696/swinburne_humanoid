<?xml version="1.0"?>
<package format="3">
  <name>asr_vosk_language_model_{{ locale.lower() }}_{{ size }}</name>
  <version>{{ pal_version }}</version>
  <description>
    This package contains the pre-trained VOSK and SpaCy language models for {{ lang_long }} ({{ size }} model).
    Upstream model: {{ model }}
  </description>
  <maintainer email="severin.lemaignan@pal-robotics.com">Séverin Lemaignan</maintainer>
  <author email="severin.lemaignan@pal-robotics.com">Séverin Lemaignan</author>
  <author email="luka.juricic@pal-robotics.com">Luka Juricic</author>
  <license>{{ license }}</license>

  <build_depend>cmake</build_depend>

  <export>
    <build_type>cmake</build_type>
  </export>

</package>
