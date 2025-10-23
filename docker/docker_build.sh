#!/bin/bash
IMAGE_NAME="latent_safety"
 
docker build -t $IMAGE_NAME -f Dockerfile .. 

