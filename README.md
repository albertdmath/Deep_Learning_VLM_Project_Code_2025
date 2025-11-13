=== PROJECT ===
VLM project for deep learning course ethz 2025.

=== HELPFUL LINKS (STUDENT CLUSTER) ===
https://www.isg.inf.ethz.ch/HelpClusterComputingStudentCluster
https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentClusterRunningJobs

=== FOR GPU USAGE (STUDENT CLUSTER) ===
# Copy local folder to remote with scp
scp -r /path/to/my_project your_username@student-cluster.inf.ethz.ch:~
ssh your_username@student-cluster.inf.ethz.ch
module load cuda/12.9
srun --pty -A deep_learning -t 60 bash

# Create a python virtual environment (important)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install transformers torch torchvision
python3 load_vlm_model.py

