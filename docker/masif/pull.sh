#!/bin/bash
REGION=us-west-1
REGISTRY=943220452459.dkr.ecr.$REGION.amazonaws.com
IMAGE=$REGISTRY/surfdock-surface:latest

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
docker pull $IMAGE
docker tag $IMAGE surfdock-surface:latest
