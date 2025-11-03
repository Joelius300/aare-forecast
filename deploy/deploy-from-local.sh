#!/bin/bash
# script to deploy local docker file to dokku without registry. DOKKU_HOST must be set.

set -eo pipefail  # fail fast

if [[ -z "$DOKKU_HOST" ]]; then
  echo "Must set DOKKU_HOST before running this script!"
  exit 1
fi

IMAGE_NAME=${1:-aare-oraku-forecast}
IMAGE_TAG=${2:-latest}
IMAGE="$IMAGE_NAME:$IMAGE_TAG"

if [[ "$IMAGE_TAG" == v* ]]; then
  echo "Version tag detected ($IMAGE_TAG) -- using tag directly."
  IMAGE_ID="$IMAGE_TAG"
else
  echo "Non-version tag detected ($IMAGE_TAG) -- using image id (digest)."
  # re-tag with image id (first part of digest) so it's unique on server (=dokku registers change)
  IMAGE_ID=$(docker images --quiet "$IMAGE")
  docker image tag "$IMAGE" "$IMAGE_NAME:$IMAGE_ID"
fi

# using resolved IMAGE_ID (either tag or digest)
docker image save "$IMAGE_NAME:$IMAGE_ID" | ssh "dokku@$DOKKU_HOST" git:load-image "$IMAGE_NAME" "$IMAGE_NAME:$IMAGE_ID"
