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

USAGE_THRESHOLD = 0.8

proc = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
output, error = proc.communicate()
if error:
    sys.stderr.write()
    sys.stdout.write("cpu")

usage_re = re.compile(r"(?<= )\d{1,8}(?=MiB /)")
matches = usage_re.findall(output)
usage_amts = [int(usage_amt) for usage_amt in matches]

total_re = re.compile(r"(?<=/)\s*\d{1,8}(?=MiB)")
matches = total_re.findall(output)
total_amts = [int(total) for total in matches]

pct_used = [float(usage_amt)/float(total) for (usage_amt, total) in zip(usage_amts, total_amts)]
print pct_used

open_gpus = [index for index in range(len(pct_used)) if pct_used[index] < USAGE_THRESHOLD]

if open_gpus:
    sys.stdout.write("gpu" + str(random.choice(open_gpus)))
else:
    sys.stdout.write("cpu")
