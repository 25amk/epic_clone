EPIC
====

# Quickstart

## Pre-requisites

Below are some of the pre-requisites required to do work with this application. In particular, we assume basic AI inference services are provided by the users.

### Python Version

EPIC requires python 3.12

### Inference servers

#### OpenAI API Key

As an auxiliary for the locally hosted models, the environment will use the OpenAI API Key if the environment variable `OPENAI_API_KEY` is set.  The environment variable will be picked up by pipelines and tests.

## make init / make fini

Install requirements using python3 (>=3.12) and virtualenv.
Virtual environment is automatically created using the `Makefile` included in
the repository.

```bash
# Initialize
$ make init
... install virtual environment & requirements ...
... runs the ./setup.py and installs the necessary packages in edit mode...

# Finalize (if needed to clean)
$ make fini
... deletes the .venv directory ...

```

## Hugging Face Repository Access

EPIC's LLM training datasets and models are located on Hugging Face repositories.
You should create an API key to be able to leverage these repositories.

```shell

# Tokens are acquired from https://huggingface.co/settings/tokens
#
# After this, tokens are stored in cat ~/.cache/huggingface/stored_tokens
#

# Below assumes you are in the environment
$ . ./.venv/bin/activate

# Logging in
$ huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
    Setting a new token will erase the existing one.
    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
$
```

This is a one-time task.  The tokens will be stored locally and would be re-used.


## `make env` / `make env-fini`

### Local development (default)

The application requires a `.env` file to acquire the knowledge of the current environment.  This is achieved by "logging in" to the environment.  `make env / make env-fini` pairs will manage this process.

```shell
# Logging in will create a `.env` file based on the default local
# development profile.
$ make env
# Logging out will erase the `.env` file
$ make env-fini
```

Once "logged in" to the environemnt, you can edit the `.env` file as fit (i.e., adding an OpenAI API Key).

#### Overriding GPU device instance designation (Nvidia)

On a system that has multiple Nvidia GPU instances, you can designate a list of GPUs you
want EPIC to leverage.  This is setup by the `CUDA_VISIBLE_DEVICES` variable in the generated
`.env` file.

By default this is set to `0` in which it uses the first GPU device found in the system.

You can set this to leverage multiple GPUs by providing a comma separated list with no spaces in between the device IDs.

In the example below, EPIC will use only GPU 3, 4, 5 on the system to run GPU workloads.

```shell
# .env file
...
CUDA_VISIBLE_DEVICES=3,4,5
...
```

The main mechanism for this limit achieved by the `get_settings()` function within `src.config`.  It will load the Settings() object from the `.env` file and export key environment
variables for the process.  For the unit and integratino tests, you need to load the settings by either calling the `get_settings()` function or by using the `settings` pytest fixture.

```python
import pytest

# Recommended way 
@pytest.mark.integration
def test_using_gpus(settings):
    print(os.environ['CUDA_VISIBLE_DEVICES'])


# Alternate way
def test_using_gpus_direct_load():
    from src.config import get_settings
    get_settings()

    print(os.environ['CUDA_VISIBLE_DEVICES'])
```

### Alternative site profiles

You can initialize the environment with different system profiles which are located as `./site/***-envs.sh` in the `./site` directory.  The `***` portion of the `***-envs.sh` file will be the profile name.

```shell
$ SITE_PROFILE=<profile_name> make env
# Logging out does not require you to specify the profile
$ make env-fini
```

## Generate Data

If you are running in `localdev`, you'll need to place some sample data in `.data/private/ornl/frontier/jobsummary/`.
If you aren't using real data, you can generate synthetic data using:
```shell
$ make generate-data
```

## Prefetch Hugging Face Models for Local Inference

For local inference, EPIC uses a selected set of Hugging Face models. If you have checked out
EPIC for instruction tuning, testing, evaluation, or web application development, you need this
step.

These models are downloaded in-line when the models are instantiated in the code, but would result in significant delays due to the large amount of data to be downloaded.  To avoid this,
prefetch the models by issuing the command below.


This step relies on the `make init` phase and the Hugging Face credentials
programmed in the previous steps.

```shell
# Download the models
$ make fetch-models

Fetching Hugging Face Models (pre-download)

- SQL model
.venv/bin/huggingface-cli download defog/llama-3-sqlcoder-8b
Fetching 15 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 12478.10it/s]
/ccsopen/home/wg8/.cache/huggingface/hub/models--defog--llama-3-sqlcoder-8b/snapshots/0f96d32e16737bda1bbe0d8fb13a932a8a3fa0bb

- RAG Embedding model
.venv/bin/huggingface-cli download mixedbread-ai/mxbai-embed-large-v1
Fetching 21 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:00<00:00, 5473.21it/s]
/ccsopen/home/wg8/.cache/huggingface/hub/models--mixedbread-ai--mxbai-embed-large-v1/snapshots/e7857440379da569f68f19e8403b69cd7be26e50

- RAG Reranker model
.venv/bin/huggingface-cli download mixedbread-ai/mxbai-rerank-large-v1
Fetching 13 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 9599.64it/s]
/ccsopen/home/wg8/.cache/huggingface/hub/models--mixedbread-ai--mxbai-rerank-large-v1/snapshots/11ec161b6e8bed3aaf24797717198eb6191e1fb7

- General Purpose Instruct model
.venv/bin/huggingface-cli download meta-llama/Llama-3.2-3B-Instruct
Fetching 16 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:00<00:00, 12690.78it/s]
/ccsopen/home/wg8/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
$ 
```

Among the models, the Meta-Llama models would potentially fail as they are "gated" models.
Simply go to the corresponding model hub webpage and ask for usage permission.  The request
will be approved few hours later.

Note that the models are stored in `~/.cache/huggingface/hub`. The steps above are idempotent.
The downloads would simply finish quickly if the models are already downloaded and exists
in the cache.

## Running the chat UI

Using the CLI tool `epc`, run the chat UI locally.
You need to make sure you have activated the python virtual environment created
by `make init`.

```shell
# Type epc chat from the root of the repository
$ epc chat
2024-10-16 14:08:49 - Loaded .env file
INFO:     Started server process [59277]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:3000 (Press CTRL+C to quit)
```

Optionally, the `Makefile` provides a directive that automatically activates
the virtual environment when starting the web chat UI.

```shell
$ make chat
2024-10-16 14:08:49 - Loaded .env file
INFO:     Started server process [59277]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:3000 (Press CTRL+C to quit)
```

## Running jupyter lab

Running below will create a jupyterlab instance that installs a kernel spec
attached to the local virtual environment.  Choose a notebook that uses the
`epic` kernelspec which will give you everything in `requirements.txt`.

```bash
$ make jupyter
```

## Jumping into the Shell

```bash
# Spawns a unix shell
$ make shell

# Spawns an ipython shell
$ make ipython
```

## make up / make down

Starts the necessary background servers for EPIC. (WIP - not implemented yet)

```bash
$ make up
... services come up ...
$ make down
... services go down ...
```

# Directory Map

TBD: Add more

* `./src`: Directory for python programs we will be using. Most of the code
  will be implemented within this python package.
   * `./src/models`: Model implementations
   * `TBD...TBD...`: Many more package implementations
* `./site`: Site dependencies.
* `./data`: directory for version controlled data (DVC or LFS)
* `./nb`: dated jupyter notebooks for ephemeral explorations
* `./lab`: temporary explorations
* `./pipelines`: metaflow workflows

# Workflows (`./pipelines`)

This project uses a workflow engine to implement its batch pipelines. 
Currently we use [metaflow](https://metaflow.org) in order to manage DAG pipelines.
Example use cases span from RAG ingest pipelines, model finetuning pipelines,
evaluation pipelines and etc.

Pipelines in the project are meant to be run in an out-of-band fashion. In
itself the pipelines are stand-alone applications that achieve certain goals
mainly producing artifacts on the other side. The main application (i.e., web UI)
will then consume the artifacts.

## Conventions

Workflows are located in the `./pipelines` directory where any subdirectories
within that directory tree is a certain `project` boundary within this project.
Each project can have multiple workflows (DAG) implemented.  Each project would
have its own README.md document that describes the details of the workflow.

The workflows currently support running on a local development laptop or a node
that has multiple GPUs, or AWS, GCP kubernetes.  On-premise kubernetes is not
supported yet (TBD).

The workflows would either load environmental configuration from the
`.env` file or the configurations defined in `~/.metaflowconfig/` to leverage
the current environment.  These configurations are meant to be managed by the
site profile functionality.

Beyond, we simply follow the conventions provided by metaflow.
We recommend following the [metaflow tutorials](https://docs.metaflow.org/getting-started/tutorials)
before working on creating a pipeline.

A model workflow can be found in `./pipelines/example` (FIXME: improve example)
which is good place to start. Simply copy the example directory as a new
project and start implementing a new workflow.


## Working in the environment

Running the pipeline requires you to activate the repository virtual
environment

```shell
# Assuming you are at the root of the repository
# Assuming you have already done `make init`

# Activate the environment
$ . ./.venv/bin/activate

# Or, jump into the shell using the provided makefile directive
$ make shell
```

Note that this virtual environment is meant to be minimal to the repository
just to get to the point being able to execute the pipelines.

The workflow should mainly rely on the `@conda` or `@pypi` metaflow
decorators to fulfill its dependencies for reproducibility. Metaflow has a
capability providing its own python version in this situation.

## Running the pipelines

The recommended way of running the pipeline is from the repository root using
the full path to the python file that implements the workflow.

```shell
# Example of running an example Tutorial pipeline
$ python3 ./pipelines/example/flow.py --environment=pypi run [options]
```

FIXME: this restriction will be lifted when we gain more experience on running
metaflow pipelines in the future.


## Pipelines vs. src

The `src` directory is where almost all the common code needs to be
implemented, shared across many pipelines. Code in `src` is referenced by
simply performing `import src` from anywhere of the workflow.

One primary use cases of importing `src` from the workflow is to read the
`.env` file by leveraging the pydantic settings defined in `src.config`.

Due to the way how Metaflow packages the directories, a relative symbolic link
to the `./src` directory of the repository root should exist alongside the workflow files.
The example of this can be found in `./pipelines/example/`. The symbolic link `../../src -> src`
is committed into git as with other files.


## Tests

We provide a pytest `pipeline` marker (example in `./pipelines/example/flow_test.py`) and
a makefile directive `make pipeline-test` that runs all the tests that should
be defined in `***_test.py` alongside the main workflow definition python
files.

The main approach is to do the below.

* Simply run a DAG check (i.e., `python3 pipelines/<project>/flow.py check`) to
  see if the workflow can load all dependencies and has a sane structure.
* Running the pipeline in test mode consuming much less resources with
  potentially synthetic data all self contained to the test itself.
  (i.e., `python3 pipelines/<project>/flow.py run --test` assuming the workflow
   itself implementing a test run mode)


# Development Environment

## Independant node local development `SITE_PROFILE=localdev make env`

This section describes the node local development setup. This development environment will not require additional infrastructure such as kubernetes or a spark cluster and will only rely on a single node configuration.

Initialization of the node local development environment is by default done through defaults that are embedded in the codebase which will be initialized by `make localdev` (or `./site/local-creds.sh`) if there are no `.env` file existing in the root directory of the repository.

### Pre-requisites

#### System

* CPU: More than 4 physical cores 
* RAM: At least 8GB of memory to be able to serve models confortably with existing programs.
* GPU: Local models run faster with better GPUs.  Supports NVidia GPUs or Apple silicon (M1, M2, M3).  With at least one required for normal operations, more GPUs will help inference workloads as well as find tuning workloads.

#### OpenAI API Key

Optionally, OpenAI API Key is required for the use of larger models during various phases of our model development.  Simply set the `OPENAI_API_KEY` somewhere in the shell or add it into the current `.env` file.


### Localhost storage space

The localdev environment will use local storage space defaulting to the locations below within the repository.

* `.scratch`: location for temporary data files, downloads & etc
* `.shared`: location for persistant data potentially shared among multiple individuals

### Object store access

For localdev, object store access is simply done via local paths similar to the local storage spaces.
If the path does not start with `s3a://`, the tools will understand that the object store is actually
served by plain POSIX paths local to the machine.

These locations are meant to replicate the telemetry archival locations served by a specific deploy site.
In a localdev environment, these locations are meant to be populated either manually or by data loading
scripts that fetches 

* `.data/shared`: location for the shared open objects.
* `.data/private`: location for objects with sensitive data.

Private vs. shared is a minimal structure inherited from the authors to support data that needs special care (i.e., PII).


## OLCF data access via ./site/aaims-envs.sh `SITE_PROFILE=aaims make env`

This section will be removed in the future when we sever the dependencies
from OLCF resources.  But for now, data needs to be downloaded or be accessed
from OLCF systems. `./site/aaims-envs.sh` depends on this process to be able
to fetch the credentials and give developers access to OLCF telemetry data.

### Prerequisites 

Fetching data from OLCF resources require the OpenShift CLI client and the
`mc` minio client.

#### openshift CLI tool 'oc'

```bash
# MacOS
$ brew install openshift-cli

# Linux (Manual installation)
$ mkdir -p ~/bin ~/bin/oc && cd ~/bin/oc
$ wget https://mirror.openshift.com/pub/openshift-v4/clients/oc/latest/linux/oc.tar.gz
$ tar -xvf oc.tar.gz
$ cp ./oc ~/bin
$ cd ~/bin && rm -rf ./oc
$ export PATH=$PATH:~/bin
```

#### Minio CLI client 'mc'

```bash
# MacOS
$ brew install minio-cli 

# Linux
$ curl https://dl.min.io/client/mc/release/linux-amd64/mc \
  --create-dirs \
  -o $HOME/minio-binaries/mc

$ chmod +x $HOME/minio-binaries/mc
$ export PATH=$PATH:$HOME/minio-binaries/
$ mc --help
```

### Setup credentials

For development, the credentials are stored as an openshift secret
"epic-app-env" in the namespace "stf218-aaimsgpu". We need to fetch this and
store it as a .env file in the root of the project.

This is achieved using the `Makefile` directives `make env` and `make env-fini`.


#### Login and fetch credentials

Below will create a `.env` file in the root directory and program the necessary
S3 credentials to `~/.mc/config.json` and
`<repository_root>/.dvc/config.local` so one can access the telemetry data.

The `.env` file is in `.gitignore`. Should not be committed to the repository.

```bash
# Login to the OLCF system
make login
```

#### Logging out

This will unlink `.env`, remove the alias entry in `~/.mc/config.json` and
`<reporsitory_root>/.dvc/config.local`.

```bash
# Logout from the OLCF system
make logout
```

### Data access

Data access is through the minio client `mc` or `dvc` or the `s3` credentials
generated in the `.env` file.

```bash
TODO: Data access method
```
