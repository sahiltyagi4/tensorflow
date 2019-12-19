#!/bin/bash

#just to create separate log directory so output from previous run doesn't need to be saved elsewhere when running new experimental config.
#this script just assumes 1 master for now (w.r.t sleep and detach mode)
log_dir="/home/styagi/nfs2data/"
# false is ASP, while true runs with BSP.
sync_mode="false"
dockerstats="dockerstats.txt"
sshstr=" styagi@"
network="overnet5"
nodelist=("v-020 " "v-019 " "v-019 " "v-gpu4 " "v-022 ")
pythonpath="/mnist_official/models"

#num_nodes is equal to the length of iplist. change this to set as external args and dynamically allocate these IPs based on # of ps, master and workers.
iplist=("172.18.0.2" "172.18.0.3" "172.18.0.4" "172.18.0.5" "172.18.0.6")
#change this for cpu allocations in heterogenous mode
cpucoresalloc=("4" "12" "12" "12" "8")
memalloc=("16G" "48G" "24G" "24G" "16G")
tcp_portlist=(":8001" ":8002" ":8003" ":8004" ":8005")
num_ps=1
num_master=1
num_workers=2
#for the logic of the script. This must always be equal to 1.
num_evaluator=1
num_nodes=$(($num_ps+$num_master+$num_workers+$num_evaluator))
tf_param="{\"environment\": \"cloud\", \"model_dir\": \"/root\", \"cluster\": {"

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
		service_id="tf-chief-"$(($i-$(($num_ps+1))))
        elif [ $(($i-1)) -eq $(($num_ps+$num_master)) ] || [ $(($i-1)) -lt $(($num_ps+$num_master+$num_workers)) ]
	then
                service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	else
                service_id="tf-eval-"$(($i-$(($(($num_ps+$num_master+$num_workers))+1))))
	fi

	echo "going to stop and remove container "$service_id
	ssh $sshstr${nodelist[$(($i-1))]} "docker container stop $service_id"
        ssh $sshstr${nodelist[$(($i-1))]} "docker container rm $service_id"
done

task_index=0
worker_str="\"worker\": ["
ps_str="\"ps\": ["
master_str="\"chief\": ["
for i in $(seq 1 $num_nodes)
do
	if [ $(($i-1)) -eq 0 ]
        then
                #ps here
                tf_param+="\"ps\":[\"${iplist[$(($i-1))]}${tcp_portlist[$(($i-1))]}\""
	elif [ $(($i-1)) -lt $num_ps ]
	then
		#ps here
		tf_param+=",\"${iplist[$(($i-1))]}${tcp_portlist[$(($i-1))]}\""
	elif [ $(($i-1)) -eq $num_ps ]
	then
		tf_param+="],"
		#master here
                tf_param+="\"chief\":[\"${iplist[$(($i-1))]}${tcp_portlist[$(($i-1))]}\""
	elif [ $(($i-1)) -gt $num_ps ] && [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
	then
		#master here
		tf_param+=",\"${iplist[$(($i-1))]}${tcp_portlist[$(($i-1))]}\""
	elif [ $(($i)) -eq $num_nodes ]
	then
		echo 'nothing goes in tf-config for EVALUATOR!!'
		#evaluator here
		##tf_param+="],\"evaluator\":[\"${iplist[$(($i-1))]}${tcp_portlist[$(($i-1))]}\""
	elif [ $(($i-1)) -eq $(($num_ps+$num_master)) ]
	then
		#1st worker string here
		tf_param+="],\"worker\":[\"${iplist[$(($i-1))]}${tcp_portlist[$(($i-1))]}\""
	else
		#worker here
		tf_param+=",\"${iplist[$(($i-1))]}${tcp_portlist[$(($i-1))]}\""
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
		tf_config=$tf_param"{\"index\": $(($i-$(($num_ps+1)))), \"type\": \"chief\"}}"
		service_id="tf-chief-"$(($i-$(($num_ps+1))))
	elif [ $(($i)) -eq $num_nodes ]
	then
		tf_config=$tf_param"{\"index\": $(($i-$(($(($num_ps+$num_master+$num_workers))+1)))), \"type\": \"evaluator\"}}"
		service_id="tf-eval-"$(($i-$(($(($num_ps+$num_master+$num_workers))+1))))
	else
		tf_config=$tf_param"{\"index\": $(($i-$(($(($num_ps+$num_master))+1)))), \"type\": \"worker\"}}"
		service_id="tf-worker-"$(($i-$(($(($num_ps+$num_master))+1))))
	fi

	#start containers
	echo "going to create container "$service_id
	echo "tf_config: "$tf_config
	echo "ip assigned: "${iplist[$(($i-1))]}
	echo "cpus alloted to task: "${cpucoresalloc[$(($i-1))]}
	echo "memory allocated to container: "${memalloc[$(($i-1))]}
	echo "model directory is $log_dir"
	ssh $sshstr${nodelist[$(($i-1))]} "docker run -it -w /mnist_official -v /home/styagi/prateeks/tensorflow:/mnist_official -v $log_dir:/root --env TF_CONFIG='$tf_config' --env PYTHONPATH='$pythonpath' --name '$service_id' --cpus ${cpucoresalloc[$(($i-1))]} -m ${memalloc[$(($i-1))]} --net $network --ip ${iplist[$(($i-1))]} -d mnistv5:v5 /bin/bash"
	###ssh $sshstr${nodelist[$(($i-1))]} "docker stats &>> $log_dir$service_id/$dockerstats"
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
                echo "going to launch MNIST model on container $service_id"
        elif [ $(($i-1)) -eq $num_ps ] || [ $(($i-1)) -lt $(($num_ps+$num_master)) ]
        then
        	service_id="tf-chief-"$(($i-$(($num_ps+1))))
	elif [ $(($i)) -eq $num_nodes ]
	then
		service_id="tf-eval-"$(($i-$(($(($num_ps+$num_master+$num_workers))+1))))
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
        	echo "going to run MNIST on container $service_id"
 		if [ "$sync_mode" == "false" ]
		then
			echo "running ASP mode"
	       		ssh $sshstr${nodelist[$(($i-1))]} "docker exec $service_id  sh -c 'python models/official/mnist/mnist.py'  &>> $log_dir$service_id/steplogs.txt &"
		elif [ "$sync_mode" == "true" ]
		then
			echo "ALL EXCEPT LAST running in BSP mode as sync is set to true.."
			ssh $sshstr${nodelist[$(($i-1))]} "docker exec $service_id  sh -c 'python models/official/mnist/mnistBSP.py' &>> $log_dir$service_id/steplogs.txt &"
		fi
	fi
done

echo "going to launch last service and redirect stdout to terminal"
echo "CHECKING SYNC_MODE VALUE......................................................$sync_mode"
if [ "$sync_mode" == "false" ]
then
	echo "LAST running ASP mode"
	docker exec $service_id  sh -c 'python models/official/mnist/mnist.py' 2>> $log_dir$service_id"/steplogs.txt"
else [ "$sync_mode" == "true" ]
	echo "LAST PROCESS running in BSP mode as sync is set to true.."
	echo "service id in last step is "$service_id
	docker exec $service_id  sh -c 'python models/official/mnist/mnistBSP.py' 2>> $log_dir$service_id"/steplogs.txt"
fi
echo "end time logged"
./date.sh
