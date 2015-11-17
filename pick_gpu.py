import subprocess
import re
import random

USAGE_THRESHOLD = 90

output = """Mon Nov 16 19:41:43 2015       
+------------------------------------------------------+                       
| NVIDIA-SMI 352.39     Driver Version: 352.39         |                       
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K40c          Off  | 0000:02:00.0     Off |                    0 |
| 29%   59C    P0   136W / 235W |   4897MiB / 11519MiB |     83%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla K40m          Off  | 0000:81:00.0     Off |                    0 |
| N/A   61C    P0   158W / 235W |   6892MiB / 11519MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla K40c          Off  | 0000:82:00.0     Off |                    0 |
| 25%   48C    P0    94W / 235W |   9365MiB / 11519MiB |     31%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     19867    C   ...b-r2014b/matlab/r2014b/bin/glnxa64/MATLAB  4872MiB |
|    1     19774    C   ...b-r2014b/matlab/r2014b/bin/glnxa64/MATLAB  6867MiB |
|    2     20285    C   ...b-r2014b/matlab/r2014b/bin/glnxa64/MATLAB  9340MiB |
+-----------------------------------------------------------------------------+"""

p = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
output, error = p.communicate()
if error:
    print "cpu"

usage_re = re.compile(r"(?<=   )\d{1,3}(?=%)")
matches = usage_re.findall(output)
usage_amts = [int(usage_amt) for usage_amt in matches]
print usage_amts

# Get GPU
open_gpus = [index for index in range(len(usage_amts)) if usage_amts[index] < USAGE_THRESHOLD]
if open_gpus:
    print random.choice(open_gpus)
else:
    print "cpu"
