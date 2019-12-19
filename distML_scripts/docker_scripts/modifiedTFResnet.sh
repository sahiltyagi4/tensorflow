#!/bin/bash

#just to create separate log directory so output from previous run doesn't need to be saved elsewhere when running new experimental config.
#this script just assumes 1 master for now (w.r.t sleep and detach mode)
log_dir="/home/styagi/nfs2data/"
#model_dir="modeldirectory"
# false is ASP, while true runs with BSP.
sync_mode="false"
stepslogs="steplogs.txt"
dockerstats="dockerstats.txt"
sshstr=" styagi@"
nodelist=("v-020 " "v-gpu4 " "v-019 " "v-022 ")
num_ps=1
num_master=1
num_workers=2
num_nodes=$(($num_ps+$num_master+$num_workers))
#make a directory to store logs for a given run
#for i in $(seq 1 $num_nodes)
#do
#	echo "going to login to node "${nodelist[$(($i-1))]}
#	ssh $sshstr${nodelist[$(($i-1))]} "mkdir $log_dir$run_number"
#	ssh $sshstr${nodelist[$(($i-1))]} "mkdir $log_dir$run_number$model_dir"
#done
#job_dir=$log_dir$run_number$model_dir

#num_nodes is equal to the length of iplist. change this to set as external args and dynamically allocate these IPs based on # of ps, master and workers.
iplist=("172.18.0.6" "172.18.0.7" "172.18.0.8" "172.18.0.9")
#change this for cpu allocations in heterogenous mode
cpucoresalloc=("9" "10" "2" "27")
trainsteps=50000
batchsizes=("32" "32" "32" "32")
memalloc=("16G" "48G" "30G" "48G")
tcp_port=":8000"
network="overnet2"
tf_param="{\"environment\": \"cloud\", \"model_dir\": \"/root\", \"batch_size_list\": [32,32,32,32], \"cluster\": {"

# stop and remove previous containers
for i in $(seq 1 $num_nodes)
do
	tf_config=""
        service_id=""
        if [ $(($i-1)) -lt $num_ps ]
        then
                service_id="tf-ps-"$(($i-1)) 
        elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
        then
		service_id="tf-master-"$(($i-$(($num_ps+1))))
	else
                service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	fi

	echo "going to stop and remove container "$service_id " on node "${nodelist[$(($i-1))]}
        ssh $sshstr${nodelist[$(($i-1))]} "docker container stop $service_id"
        ssh $sshstr${nodelist[$(($i-1))]} "docker container rm $service_id"
done

task_index=0
worker_str="\"worker\": ["
ps_str="\"ps\": ["
master_str="\"master\": ["
for i in $(seq 1 $num_nodes)
do
	if [ $(($i-1)) -eq 0 ]
        then
                #ps here
                tf_param+="\"ps\":[\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -lt $num_ps ]
	then
		#ps here
		tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -eq $num_ps ]
	then
		tf_param+="],"
		#master here
                tf_param+="\"master\":[\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -gt $num_ps ] && [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
	then
		#master here
		tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
	elif [ $(($i-1)) -eq $(($num_ps+$num_master)) ]
	then
		#1st worker string here
		tf_param+="],\"worker\":[\"${iplist[$(($i-1))]}$tcp_port\""
	else
		#worker here
		tf_param+=",\"${iplist[$(($i-1))]}$tcp_port\""
	fi
done

tf_param+="]}, \"task\": "
#to add task type, index to tf_config and launch appropriate container with its corresponding cpu-cores allocation. \
#change this to dynamically pick task_cpus dynamically (another array or map) to accomodate heterogenous setups.
for i in $(seq 1 $num_nodes)
do
	tf_config=""
	service_id=""
	if [ $(($i-1)) -lt $num_ps ]
    	then
		tf_config=$tf_param"{\"index\": $(($i-1)), \"type\": \"ps\"}}"
		service_id="tf-ps-"$(($i-1)) 
	elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
	then
		tf_config=$tf_param"{\"index\": $(($i-$(($num_ps+1)))), \"type\": \"master\"}}"
		service_id="tf-master-"$(($i-$(($num_ps+1))))
	else
		tf_config=$tf_param"{\"index\": $(($i-$(($(($num_ps+$num_master))+1)))), \"type\": \"worker\"}}"
		service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	fi

	#start containers
	echo "going to create container "$service_id
	echo "tf_config:"$tf_config
	echo "ip assigned: "${iplist[$(($i-1))]}
	echo "cpus alloted to task: "${cpucoresalloc[$(($i-1))]}
	echo "memory allocated to container: "${memalloc[$(($i-1))]}
	echo "model directory is $job_dir"
	echo "going to launch container on "${nodelist[$(($i-1))]}
	echo "ssh string to print: "$sshstr${nodelist[$(($i-1))]}
	#--env HOST_PERMS='\"'$(id -u):$(id -g)'\"'
	ssh $sshstr${nodelist[$(($i-1))]} "docker run -i -w /resnet-cifar10 -v /home/styagi/prateeks/tensorflow:/resnet-cifar10 -v $log_dir:/root --name $service_id --cpus ${cpucoresalloc[$(($i-1))]} -m ${memalloc[$(($i-1))]} --env TF_CONFIG='$tf_config' --net $network --ip ${iplist[$(($i-1))]} -d tflogitertime:v1 /bin/bash"
	########ssh $sshstr${nodelist[$(($i-1))]} "docker stats &>> $log_dir/$service_id/$dockerstats &"
done

echo "start time logged:"$start_time
./date.sh
#run resnet model code on the launched containers
for i in $(seq 1 $num_nodes)
do
        service_id=""
        if [ $(($i-1)) -lt $num_ps ]
 	then
                service_id="tf-ps-"$(($i-1)) 
        	echo "going to launch resnet model on container $service_id"
        elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
        then
        	service_id="tf-master-"$(($i-$(($num_ps+1))))
        else
        	service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	fi

	if [ ! -d $log_dir$service_id ]
	then
		echo "directory to create:"$log_dir$service_id " on node:"${nodelist[$(($i-1))]}
		ssh $sshstr${nodelist[$(($i-1))]} "mkdir $log_dir$service_id"
	fi

	if [ $i != $num_nodes ]
        then
		echo "going to launch resnet model on container $service_id"
 		if [ "$sync_mode" == "false" ]
		then
			echo "running ASP mode"
			echo "batch size is: ${batchsizes[$(($i-1))]}"
	       		ssh $sshstr${nodelist[$(($i-1))]} "docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir='\"'/root'\"' --train-batch-size='${batchsizes[$(($i-1))]}' --num-gpus=0 --train-steps='$trainsteps''  &>> $log_dir$run_number$service_id/$stepslogs &"
		elif [ "$sync_mode" == "true" ]
		then
			echo "ALL EXCEPT LAST running in BSP mode as sync is set to true.."
			echo "batch size is: ${batchsizes[$(($i-1))]}"
			ssh $sshstr${nodelist[$(($i-1))]} "docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir='\"'/root'\"' --train-batch-size='${batchsizes[$(($i-1))]}' --num-gpus=0 --train-steps='$trainsteps' --sync'  &>> $log_dir$run_number$service_id/$stepslogs &"
		fi
	fi
done

echo "going to launch last service and redirect stdout to terminal"
echo "CHECKING SYNC_MODE VALUE......................................................"
echo "$sync_mode"
if [ "$sync_mode" == "false" ]
then
	echo "LAST running ASP mode"
	echo "batch size is: ${batchsizes[$(($i-1))]}"
	#ssh $sshstr${nodelist[$(($i-1))]} "docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir='\"'/root'\"' --num-gpus=0 --train-steps='$trainsteps'' 2>> $log_dir$run_number$service_id/'\"'stepslogs.txt'\"'"
	docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir="/root" --train-batch-size='${batchsizes[$(($i-1))]}' --num-gpus=0 --train-steps='$trainsteps'' 2>> $log_dir$run_number$service_id/$stepslogs
elif [ "$sync_mode" == "true" ]
then
	echo "LAST PROCESS running in BSP mode as sync is set to true.."
	echo "batch size is: ${batchsizes[$(($i-1))]}"
	docker exec $service_id  sh -c 'python /resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar10_main.py --data-dir=/resnet-cifar10/models/tutorials/image/cifar10_estimator/cifar-10-data --job-dir="/root" --train-batch-size='${batchsizes[$(($i-1))]}' --num-gpus=0 --train-steps='$trainsteps' --sync' 2>> $log_dir$run_number$service_id/$stepslogs
fi

echo "end time logged"
./date.sh
