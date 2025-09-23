#!/bin/bash
# This script is used to setup a local development environment

echo ""
echo "Localdev environment setup teardown"
echo ""
ROOT_DIR=$(cd $(dirname $(dirname $0)) && pwd)
APP_NAME=$(basename ${ROOT_DIR})
SITE_PROFILE=localdev
echo ${ROOT_DIR}
if [ $# -lt 1 ]; then
	echo "Usage: $(basename $0) <setup|teardown>"
	exit 1
fi
CMD=$1

#
# Environment variables for generic localhost development
#
# Passed to the .env file for the application
#

# A posix based local path used for local storage
SCRATCH_BASE=$(pwd)/.scratch
SHARED_BASE=$(pwd)/.shared
DEFAULT_PREFIX=$(pwd)/.data

# Object storage location (posix in this case)
DATA_BASE=${DEFAULT_PREFIX}/shared
DATA_PRIV_BASE=${DEFAULT_PREFIX}/private

# Compilation of the final .env file
DOTENV_PAYLOAD="
# Site profile
SITE_PROFILE=${SITE_PROFILE}

# Application information
APP_NAME=${APP_NAME}
# BASE_PATH=""
# LARGE_MODEL="openai"
# SQL_MODEL="defog/llama-3-sqlcoder-8b"

# Obsidian data access
SCRATCH_BASE=${SCRATCH_BASE}
SHARED_BASE=${SHARED_BASE}
DATA_BASE=${DATA_BASE}
DATA_PRIV_BASE=${DATA_PRIV_BASE}

# OpenAI API Key
# OPENAI_API_KEY=Add a relevant API key

# Nvidia GPU selection
CUDA_VISIBLE_DEVICES=0
"

#
# Main script
#

case ${CMD} in
setup)
	# Setup environment variables via dotenv
	touch ${ROOT_DIR}/.env
	chmod 600 ${ROOT_DIR}/.env
	echo "${DOTENV_PAYLOAD}" > ${ROOT_DIR}/.env

	echo "@ credential setup success"
	;;
teardown)
	echo "@ removing local credentials"
	# Remove environment variables
	rm ${ROOT_DIR}/.env

	echo "@ logout success"
	;;
esac
echo ""
exit 0