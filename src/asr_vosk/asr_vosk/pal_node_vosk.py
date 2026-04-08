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

# !! PAL Robotics specific !!
# To use vosk outside of a PAL robot or a PAL developer docker, use the
# standard entry point in node_vosk.py.
#
# This wrapper for the Vosk ASR node is needed to add the Vosk Python module to the Python path,
# due to the PAL `vosk` debian installing a vosk virtual environment in /opt/pal/venvs.
import sys
from pathlib import Path

python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
pal_vosk_site_packages = Path(f'/opt/pal/venvs/vosk/lib/python{python_version}/site-packages')
if pal_vosk_site_packages.is_dir():
    sys.path.insert(0, str(pal_vosk_site_packages))

try:
    import vosk  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'vosk' Python module is not available. "
        "Expected PAL venv path "
        f"'{pal_vosk_site_packages}' or a regular Python installation of 'vosk'."
    ) from exc

from asr_vosk.node_vosk import NodeVosk, main  # noqa: E402, F401

if __name__ == '__main__':
    main()
