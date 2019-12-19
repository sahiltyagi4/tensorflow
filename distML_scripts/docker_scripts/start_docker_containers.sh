#!/bin/bash

docker run -it  -e HOST_PERMS="$(id -u):$(id -g)" -e TF_CONFIG="$(cat ps.env)" --name "tf-ps" --net tfnet --ip 172.18.0.3 --cpus 4 -d tf1ximage:final /bin/bash 

docker run -it  -e HOST_PERMS="$(id -u):$(id -g)" -e TF_CONFIG="$(cat master.env)" --name "tf-master" --net tfnet --ip 172.18.0.5 --cpus 4 -d tf1ximage:final /bin/bash

docker run -it  -e HOST_PERMS="$(id -u):$(id -g)" -e TF_CONFIG="$(cat worker-0.env)" --name "tf-worker0" --net tfnet --ip 172.18.0.4 --cpus 8 -d tf1ximage:final /bin/bash 

docker run -it  -e HOST_PERMS="$(id -u):$(id -g)" -e TF_CONFIG="$(cat worker-0.env)" --name "tf-worker1" --net tfnet --ip 172.18.0.2 --cpus 8 -d tf1ximage:final /bin/bash

#docker exec tf-ps  python /usr/local/tensorflow/models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=/usr/local/tensorflow/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir=/root --num-gpus=0 
