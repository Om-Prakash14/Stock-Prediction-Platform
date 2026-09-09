# Copyright 2020-2026 Jordi Corbilla. All Rights Reserved.
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
# ==============================================================================
import os
from urllib.parse import quote


class ReadmeGenerator:
    def __init__(self, base_url, project_folder, short_name):
        self.base_url = base_url
        self.project_folder = project_folder
        self.short_name = short_name.strip().replace('.', '')

    def write(self):
        # Create directory if it doesn't exist yet
        os.makedirs(self.project_folder, exist_ok=True)
        project_url = quote(os.path.relpath(self.project_folder, os.getcwd()).replace(os.sep, '/'), safe='/')
        base_url = self.base_url.rstrip('/')
        def image_url(filename):
            return base_url + '/' + project_url + '/' + quote(filename, safe='')
        
        readme_path = os.path.join(self.project_folder, 'README.md')
        with open(readme_path, "w+", encoding="utf-8") as my_file:
            my_file.write('![](' + image_url(self.short_name + '_price.png') + ')\n')
            my_file.write('![](' + image_url(self.short_name + '_hist.png') + ')\n')
            my_file.write('![](' + image_url(self.short_name + '_prediction.png') + ')\n')
            my_file.write('![](' + image_url('MSE.png') + ')\n')
            my_file.write('![](' + image_url('loss.png') + ')\n')