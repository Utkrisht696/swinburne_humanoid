# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
from jinja2 import Environment, select_autoescape, FileSystemLoader
import os
import pathlib
from urllib.parse import urlparse

env = Environment(loader=FileSystemLoader("tpl"),
                  autoescape=select_autoescape())

package_tpl = env.get_template("package.xml.tpl")
cmakelists_tpl = env.get_template("CMakeLists.txt.tpl")
config_tpl = env.get_template("config/vosk_model.yml.tpl")

PAL_VERSION = "2.1.0"

ROOT = pathlib.Path("./src/")

with open("vosk_models.csv", "r") as csvfile:
    lang = csv.DictReader(csvfile)
    lang_list = [row for row in lang]

    for row in lang_list:

        locale = row["locale"]
        size = row["size"]

        # skip large models
        if size == "large":
            continue

        model = urlparse(row["url"]).path.split("/")[-1]

        print(f"Generating package for locale: {locale}, size: {size}")
        path = ROOT / ("asr_vosk_language_model_" +
                       locale.lower() + "_" + size)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "package.xml", "w") as fh:
            fh.write(package_tpl.render(
                row, model=model, pal_version=PAL_VERSION))

        with open(path / "CMakeLists.txt", "w") as fh:
            fh.write(
                cmakelists_tpl.render(
                    row, model=model, pal_version=PAL_VERSION))

        path = path / "config"
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "vosk_model.yml", "w") as fh:
            fh.write(
                config_tpl.render(
                    row, name=f"vosk_model_{size}"))
