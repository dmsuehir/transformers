<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Training on Multiple CPUs from a Kubernetes Cluter

Using multiple CPUs accelerates the model training process by distributing the workload across multiple nodes. This guide
builds upon [Distributed CPU Training](https://huggingface.co/docs/transformers/perf_train_cpu_many) to move the workload
to a [Kubernetes](https://kubernetes.io) cluster.

[Kubeflow](https://www.kubeflow.org) simplifies the process of deploying machine learning workloads to Kubernetes
clusters. For this example, the [Kubeflow PyTorchJob training operator](https://www.kubeflow.org/docs/components/training/pytorch/)
is used to deploy the distributed training job to the cluster. For efficient training on CPUs,
[Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) and
[Intel® oneCCL Bindings for PyTorch](https://github.com/intel/torch-ccl) libraries are used.

## Cluster setup

This guide assume that you already have a Kubernetes clusters with multiple CPUs. Use `kubectl get nodes` to see a list
the nodes that are available on your cluster.

### Kubeflow Install

Follow the [Kubeflow installation](https://www.kubeflow.org/docs/started/installing-kubeflow/) guide to deploy the
Kubeflow resources to your cluster. To verify that the PyTorch custom resource has been deployed to your cluster, use
`kubectl get crd` and verify that the output includes `pytorchjob.kubeflow.org` like the example below:
```
#TODO Insert output
```

### Storage

Storage is required to storing the dataset, model files, checkpoints, etc. during training. There are different options
for storing data, depending on your situation. If you are using a cloud service provider like AWS or Google Cloud the
storage buckets can be setup to be used as storage with Kubernetes. Alternatively, NFS can be used with a Kubernetes
storageclass and a persistent volume claim (PVC).

<!-- TODO: Add more details here -->

## Container

Since Kubernetes requires jobs to run in a [Docker container](https://www.docker.com/resources/what-container/), you
will need a container that includes your model training script along with all of it's dependencies. The
[dockerfile](https://github.com/huggingface/transformers/blob/main/docker/transformers-pytorch-cpu/Dockerfile) would
typically need to include PyTorch, transformers, Intel Extension for PyTorch, Intel oneCCL Bindings for PyTorch, and
OpenSSH to communicate between the containers. This container can be used as a base and then you can include your
training script on top of that.

<!-- TODO: Maybe we can contribute a dockerfile here that is updated and includes everything needed to run distributed jobs -->

## Kubernetes Specification Files

The Kubernetes specification files define the resources that get deployed to the cluster. This will typically include
the `PyTorchJob` for training and optionally a `PersistentVolumeClaim` if an NFS storage class is being used for storing
datasets, trained model files, etc.

### Persistent Volume Claim

If you are using a [Persistent Volume Claim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/),
this can be deployed to the cluster before the training job. To set the PVC spec, you will need to know the name of the
NFS [storage class](https://kubernetes.io/docs/concepts/storage/storage-classes/). If you don't know have this
information, use `kubectl get storageclass` to see a list of all the storage classes on your cluster.

In addition to the storage class name, the PVC spec yaml defines the name and namespace for the PVC and the storage
size. The snippet below shows an example of yaml file for a PVC:
```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: transformers-pvc
  namespace: kubeflow
spec:
  storageClassName: nfs-client
  accessModes:
    - "ReadWriteMany"
  resources:
    requests:
      storage: 30Gi
```
The PVC can be deployed to the Kubernetes cluster using the yaml like:
```
kubectl create -f pvc.yaml
```

### PyTorchJob

The PyTorchJob is used to run the distributed training job on the cluster. The yaml file for the PyTorchJob defines
import parameters such as:
 * A name for the PyTorchJob
 * The number of replicas (workers)
 * The types of resources (node label, memory, and CPU) needed for each worker
 * The image/tag for the Docker container to use
 * Environment variables
 * The python script and it's parameters that will be used to run the training job
 * A volume mount for the PVC

The volume mount defines a path where the PVC deployed in the previous step will be mounted in the container for each
worker pod. This location can be used for the dataset, checkpoint files, and the saved model after training completes.

The snippet below is an example of a yaml file for a PyTorchJob with 4 workers running the
[question-answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).
```
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: transformers-pytorchjob
  namespace: kubeflow
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 4
    maxRestarts: 10
  pytorchReplicaSpecs:
    Worker:
      replicas: 4
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: huggingface/transformers-pytorch-cpu:latest
              imagePullPolicy: IfNotPresent
              command:
                - torchrun
                - /workspace/transformers/examples/pytorch/question-answering/run_qa.py
                - --model_name_or_path
                - "bert-large-uncased"
                - --dataset_name
                - "squad"
                - --do_train
                - --do_eval
                - --per_device_train_batch_size
                - "12"
                - --learning_rate
                - "3e-5"
                - --num_train_epochs
                - "2"
                - --max_seq_length
                - "384"
                - --doc_stride
                - "128"
                - --output_dir
                - "/tmp/pvc-mount/output"
                - --no_cuda
                - --ddp_backend
                - "ccl"
                - --use_ipex
                - --bf16
              env:
              - name: LD_PRELOAD
                value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"
              - name: TRANSFORMERS_CACHE
                value: "/tmp/pvc-mount/transformers_cache"
              - name: HF_DATASETS_CACHE
                value: "/tmp/pvc-mount/hf_datasets_cache"
              - name: LOGLEVEL
                value: "INFO"
              - name: CCL_WORKER_COUNT
                value: "1"
              resources:
                limits:
                  cpu: 200
                  memory: 128Gi
                requests:
                  cpu: 200
                  memory: 128Gi
              volumeMounts:
              - name: pvc-volume
                mountPath: /tmp/pvc-mount
              - mountPath: /dev/shm
                name: dshm
          restartPolicy: Never
          nodeSelector:
            node-type: spr
          volumes:
          - name: pvc-volume
            persistentVolumeClaim:
              claimName: transformers-pvc
          - name: dshm
            emptyDir:
              medium: Memory
```
To run this example, update the spec based on the nodes in your cluster. Use `kubectl get nodes` and
`kubectl describe node <node name>` to find the number of available nodes and the CPU and memory capacity of the nodes.
The CPU resource limits/requests in the yaml are defined in
[cpu units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu) where 1 CPU
unit is equivalent to 1 physical CPU core or 1 virtual core (depending on whether the node is a physical host or a VM).
The amount of CPU and memory limits/requests defined in the yaml should be less than the amount of available CPU/memory
capacity on a single machine. It is usually a good idea to not use the entire machine's capacity in order to leave
some resources for the kubelet and OS.

After the PyTorchJob spec has been updated with values appropriate to your cluster and trianing job, it can be deployed
to the cluster using:
```
kubectl create -f pytorchjob.yaml
```

The training job status can be monitored using:
```
kubectl describe pytorchjob transformers-pytorchjob
```

Worker pods can be monitored using:
```
kubectl get pods -n kubeflow -o wide
```

The logs for worker can be viewed using `kubectl logs -n kubeflow <pod name>`. Add `-f` to stream the logs, for example:
```
kubectl logs -n kubeflow transformers-pytorchjob-worker-0 -f
```

After the training job completes, the trained model can be copied from NFS or storage bucket, if cloud storage was
being used.

Check out this [blog post](TBD) for a full example that fine tunes
[meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) using multiple nodes from a Kubernetes cluster.
