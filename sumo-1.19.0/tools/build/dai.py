#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2008-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    dailyBuildMSVC.py
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @author  Laura Bieker
# @date    2008

"""
Does the nightly git pull on the windows server and the visual
studio build. The script is also used for the meso build.
Some paths especially for the names of the texttest output dirs are
hard coded into this script.
"""
from __future__ import absolute_import
from __future__ import print_function
import datetime
import os
import glob
import zipfile
import shutil
import sys

import status
import wix

env = os.environ
if "SUMO_HOME" not in env:
    env["SUMO_HOME"] = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMO_HOME = env["SUMO_HOME"]
env["PYTHON"] = "python"
env["SMTP_SERVER"] = "smtprelay.dlr.de"

sys.path += [os.path.join(SUMO_HOME, "tools"), os.path.join(SUMO_HOME, "tests")]
import sumolib  # noqa
import runExtraTests  # noqa

BINARIES = ("activitygen", "emissionsDrivingCycle", "emissionsMap",
            "dfrouter", "duarouter", "jtrrouter", "marouter",
            "netconvert", "netedit", "netgenerate",
            "od2trips", "polyconvert", "sumo", "sumo-gui")


status.killall(("", "D"), BINARIES)
