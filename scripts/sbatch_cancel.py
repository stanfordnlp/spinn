import subprocess
import sys
import getpass

### Kill a job and its chain of dependents (as created by sbatch_submit).                                                                                                                      
### Usage: python sbatch_cancel.py [Name of first running job in chain] [Name of first running job in chain for a second chain] ...                                                                                                                        

USER = getpass.getuser()

lines = subprocess.check_output(['squeue', '-u', USER, '-o', '"%.8A %.20E"'])
lines = lines.split('\n')
lines.sort()

for current_job in sys.argv[1].split():
	if len(current_job) < 5:
		continue

	to_kill = [current_job]
	for line in lines:
	        s = line.split()
	        if (len(s) > 0) and (to_kill[-1] in s[2]):
	                to_kill.append(s[1])

	subprocess.call(['scancel'] + to_kill)
