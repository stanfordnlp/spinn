import subprocess
import sys
import random
import re

# Print the name of a device to use, either 'cpu' or 'gpu0', 'gpu1',...
# GPUs with usage under the constant threshold will be chosen first,
# but subject to that constraint, selection is random.
#
# Warning: This is hacky and brittle, and can break if nvidia-smi changes 
# in the way it formats its output.
#
# Maintainer: sbowman@stanford.edu

USAGE_THRESHOLD = 90

proc = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
output, error = proc.communicate()
if error:
    sys.stderr.write()
    print "cpu"

usage_re = re.compile(r"(?<=   )\d{1,3}(?=%)")
matches = usage_re.findall(output)
usage_amts = [int(usage_amt) for usage_amt in matches]
print usage_amts

# Get GPU
gpus = [index for index in range(len(usage_amts))]
open_gpus = [index for index in range(len(usage_amts)) if usage_amts[index] < USAGE_THRESHOLD]

if open_gpus:
    print "gpu" + str(random.choice(open_gpus))
elif gpus:
    print "gpu" + str(random.choice(gpus))
else:
    print "cpu"
