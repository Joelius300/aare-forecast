#!/bin/bash
# script to deploy local docker file to dokku without registry. DOKKU_HOST must be set.

IMAGE_NAME=${1:-aare-oraku-prediction}
IMAGE_TAG=${2:-latest}
IMAGE="$IMAGE_NAME:$IMAGE_TAG"
IMAGE_ID=$(docker images --quiet "$IMAGE")

# re-tag with image id (first part of digest) so it's unique on server (dokku registers change)
docker image tag "$IMAGE" "$IMAGE_NAME:$IMAGE_ID"
docker image save "$IMAGE_NAME:$IMAGE_ID" | ssh "dokku@$DOKKU_HOST" git:load-image "$IMAGE_NAME" "$IMAGE_NAME:$IMAGE_ID"
