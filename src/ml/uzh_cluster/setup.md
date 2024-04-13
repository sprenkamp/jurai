git
 
This workflow still works: [[Working on remote learnings#git ssh]]
 
# poetry / envs
 
The Science Cluster only works with conda - so eg. installing another version of python than the default will not work (no sudo, no apt get)-
 
But we can use conda with any version of Python and then use poetry within a conda environment.
(https://michhar.github.io/2023-07-poetry-with-conda/)
## create a conda env
 
Following this: https://docs.s3it.uzh.ch/how-to_articles/how_to_use_conda/#create-your-environment
First get an interactive session going for an amount of time. This is just to save the resources of everyone else.
 
```bash
srun --pty -n 1 -c 2 --time=01:00:00 --mem=7G bash -l
```
 
Now the signature in front of the shell changes to reflect that you are not in the "login" environment anymore.
If this happened, now we can launch the anaconda module (I do not yet completely understand this, but it is necessary to launch conda envs)
 
```bash
module load anaconda3
```
 
Now to get an env with python 3.10
 
```bash
conda create --name <myenv> python=3.10
```
 
And activate it
 
```bash
source activate <myenv>
```
## install poetry
 
Now that the conda env was activated, what worked was:
 
```
pip install pipx
```
 
```
pipx install poetry
```
 
Then (path was "/home/maangs/data/conda/envs") for me
 
```bash
poetry config virtualenvs.path <path_to_conda_envs>
poetry config virtualenvs.create false
```
 
To actually use poetry, I afterwards needed **open a new terminal**
## use poetry
 
Now I could enter any poetry project with a pyproject.toml and use `poetry install` inside an activated conda env.
 
Somehow poetry shell does not work though, but as poetry install installs everything within the conda env, this did not matter so much.
 
# interactive sessions
 
Launch an interactive GPU session:
 
First restrict to use A100 only:
 
```bash
module load a100
```
 
Then the job scheduler assigns only that when running:
```bash
srun --pty --time=1:0:0 --mem-per-cpu=7800 --cpus-per-task=2 --gpus=1 bash -l
```
 
Interactive sessions are like jobs in the queue, to cancel:
Find out the job id:
```
squeue -u $USER
```
cancel it:
```
scancel <id-of-the-job>
```