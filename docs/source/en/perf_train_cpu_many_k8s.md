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

Using multiple CPUs accelerates the model training process by distributing the workload across multiple nodes. This
guide builds upon the [Distributed CPU Training](https://huggingface.co/docs/transformers/perf_train_cpu_many) example
to move the workload to a [Kubernetes](https://kubernetes.io) cluster.

[Kubeflow](https://www.kubeflow.org) simplifies the process of deploying machine learning workloads to Kubernetes
clusters. For this example, the [Kubeflow PyTorchJob training operator](https://www.kubeflow.org/docs/components/training/pytorch/)
is used to deploy the distributed training job to the cluster. For efficient training on CPUs,
[Intel® Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) and
[Intel® oneCCL Bindings for PyTorch](https://github.com/intel/torch-ccl) libraries are used.

## Cluster setup

> If [role-based access control (RBAC)](https://kubernetes.io/docs/reference/access-authn-authz/rbac/) is enabled on the
> cluster, listing nodes and many other cluster wide commands require specific roles to be granted to the user. The
> `kubectl auth can-i get nodes` will return "yes" if you are able to list the nodes. Otherwise, check with your cluster
> admin to get a list of the nodes that you have access to.

This guide assumes that you [installed `kubectl`](https://kubernetes.io/docs/tasks/tools/) and have access to a
Kubernetes cluster with multiple CPU nodes that will be used to run the distributed training job. If
`kubectl auth can-i get nodes` returns true, use `kubectl get nodes` to verify that kubectl is properly configured and
see a list of the nodes in the cluster. Otherwise, another command like `kubectl get pods` can be used to verify that
kubectl is working with your cluster.

Before running the distributed training job on the cluster, Kubeflow needs to be installed and there needs to be a
storage location that can be used for the dataset and model files. The next couple of sections explain this in more
detail.

### Kubeflow Install

Follow the [Kubeflow installation](https://www.kubeflow.org/docs/started/installing-kubeflow/) guide to deploy the
Kubeflow resources to your cluster. Kubeflow is typically installed in a namespace called `kubeflow`, but you can check
with your cluster admin to find out if a different namespace should be used. If you are using a namespace other than
`kubeflow`, you will need to update the `namespace: kubeflow` line in the yaml examples in this guide.

To verify that the PyTorch custom resource has been deployed to your cluster, use
`kubectl get crd pytorchjobs.kubeflow.org` and ensure that the output is similar to:
```
NAME                       CREATED AT
pytorchjobs.kubeflow.org   2023-03-24T15:42:17Z
```

### Storage

Storage is required to storing the dataset, model files, checkpoints, etc. during training. There are different options
for storing data, depending on your situation. If you are using a cloud service provider like AWS or Google Cloud the
storage buckets can be setup to be used as storage with Kubernetes.
<!-- TODO: Add more details here -->

Alternatively, NFS can be used with a Kubernetes storageclass and a persistent volume claim (PVC).
`kubectl get storageclass` can be used to list the storage classes on the cluster. The example output below shows a
storage class called `nfs-client`. The storage class name is important, because it will be used in the yaml file used
to [deploy the Persistent Volume Claim](#persistent-volume-claim) later in this guide.
```
NAME                   PROVISIONER                                     RECLAIMPOLICY   VOLUMEBINDINGMODE   ALLOWVOLUMEEXPANSION   AGE
nfs-client (default)   cluster.local/nfs-subdir-external-provisioner   Delete          Immediate           true                   310d
```

## Container

Since Kubernetes requires jobs to run in a [Docker container](https://www.docker.com/resources/what-container/), you
will need a container that includes your model training script along with all of it's dependencies. The
[dockerfile](https://github.com/huggingface/transformers/blob/main/docker/transformers-pytorch-cpu/Dockerfile) would
typically need to include PyTorch, transformers, Intel Extension for PyTorch, Intel oneCCL Bindings for PyTorch, and
OpenSSH to communicate between the containers. This container can be used as a base and then you can include your
training script on top of that.

<!--
TODO: Maybe we can contribute a dockerfile here that is updated and includes everything needed to run distributed jobs.
      There's a docker directory with dockerfiles for CPU and GPU, however the CPU dockerfiles are outdated and don't
      include things like IPEX. Also, Hugging Face has transformers images published to DockerHub, but again the ones
      for CPU are outdated (haven't been updated in 2 years) -- maybe we can find out why.
-->

## Kubernetes Specification Files

The Kubernetes specification files define the resources that get deployed to the cluster. For a distributed PyTorch
training job, this will typically include the `PyTorchJob` for worker pods and optionally a `PersistentVolumeClaim` if
an NFS storage class is being used for storing datasets, trained model files, etc.

### Persistent Volume Claim

If you are using a [Persistent Volume Claim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/),
this should be deployed to the cluster before the training job. To define the PVC spec, you will need to know the name
of the NFS [storage class](https://kubernetes.io/docs/concepts/storage/storage-classes/). If you don't know have this
information, use `kubectl get storageclass` to see a list of all the storage classes on your cluster. If a storage class
name is not provided, the cluster's default storage class will be used.

In addition to the storage class name, the PVC spec yaml defines the name and namespace for the PVC and the storage
size. The snippet below shows an example of yaml file for a PVC named `tranformers-pvc` in the `kubeflow` namespace that
uses a storage class named `nfs-client`.
```yaml
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
Deploy the persistent volume claim to the cluster using the yaml file:
```
kubectl create -f pvc.yaml
```
After that, you can verify that the PVC deployment was successful by ensuring that PVC name (in this case,
`transformers-pvc`) is displayed in the `kubectl get pvc -n kubeflow` list:
```
NAME                      STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
...
transformers-pvc          Bound    pvc-73ab011a-02e8-42f1-b078-1a3d4706509b   30Gi       RWX            nfs-client     58s
```

### PyTorchJob

The PyTorchJob is used to run the distributed training job on the cluster. The yaml file for the PyTorchJob defines
import parameters such as:
 * The name of the PyTorchJob
 * The number of replicas (workers)
 * The types of resources (node label, memory, and CPU) needed for each worker
 * The image/tag for the Docker container to use
 * Environment variables
 * The python script and it's parameters that will be used to run the training job
 * A volume mount for the PVC

The volume mount defines a path where the PVC will be mounted in the container for each worker pod. This location can be
used for the dataset, checkpoint files, and the saved model after training completes.
<!-- TODO: What would it look like if a cloud storage bucket was being used instead? -->

The snippet below is an example of a yaml file for a PyTorchJob with 4 workers running the
[question-answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).
```yaml
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
      replicas: 4  # The number of worker pods
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: huggingface/transformers-pytorch-cpu:latest  # Specify the docker image to use for the worker pods
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
                - --bf16          # Use the bf16 flag, if your hardware supports mixed precision
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
                  cpu: 200         # Update the CPU and memory limit values based on your nodes
                  memory: 128Gi
                requests:
                  cpu: 200         # Update the CPU and memory request values based on your nodes
                  memory: 128Gi
              volumeMounts:
              - name: pvc-volume
                mountPath: /tmp/pvc-mount
              - mountPath: /dev/shm
                name: dshm
          restartPolicy: Never
          nodeSelector:
            node-type: spr   #  Optionally specify a node label to specify the type of node used for the worker pods
          volumes:
          - name: pvc-volume
            persistentVolumeClaim:
              claimName: transformers-pvc
          - name: dshm
            emptyDir:
              medium: Memory
```
To run this example, update the yaml based on the nodes in your cluster. Use `kubectl get nodes` (or ask your cluster
admin for a list of nodes available to you) and `kubectl describe node <node name>` to find the number of available
nodes, node labels, and the CPU and memory capacity of the nodes.

The CPU resource limits/requests in the yaml are defined in
[cpu units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu) where 1 CPU
unit is equivalent to 1 physical CPU core or 1 virtual core (depending on whether the node is a physical host or a VM).
The amount of CPU and memory limits/requests defined in the yaml should be less than the amount of available CPU/memory
capacity on a single machine. It is usually a good idea to not use the entire machine's capacity in order to leave some
resources for the kubelet and OS. In order to get ["guaranteed"](https://kubernetes.io/docs/concepts/workloads/pods/pod-qos/#guaranteed)
[quality of service](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/) for the worker pods,
set the same CPU and memory amounts for both the resource limits and requests.

After the PyTorchJob spec has been updated with values appropriate to your cluster and training job, it can be deployed
to the cluster using:
```
kubectl create -f pytorchjob.yaml
```

The `kubectl get pods -n kubeflow` command can then be used to list the pods in the `kubeflow` namespace. You should see
the worker pods for the PyTorchJob that was just deployed. At first, they will probably have a status of "Pending" as
the containers get pulled and created, then the status should change to "Running".
```
NAME                                                     READY   STATUS                  RESTARTS          AGE
...
transformers-pytorchjob-worker-0                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-1                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-2                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-3                         1/1     Running                 0                 7m37s
...
```

The logs for worker can be viewed using `kubectl logs -n kubeflow <pod name>`. Add `-f` to stream the logs, for example:
```
kubectl logs -n kubeflow transformers-pytorchjob-worker-0 -f
```

After the training job completes, the trained model can be copied from NFS or storage bucket, if cloud storage was
being used. When you are done with the job, the PVC and PyTorch job resources can be deleted from the cluster using
`kubectl delete -f <yaml file>` or `kubectl delete <resource type> <resource name>`.

## Summary

This guide walked through the setup and best known methods for using the
[Kubeflow PyTorch training operator](https://www.kubeflow.org/docs/components/training/pytorch/)
to run a distributed training job on a Kubernetes cluster with the
[question-answering example](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering).
The PyTorchJob yaml used in this example can be used as a template to run your own script and workload. Swap in your
container image along with the python script and training parameters to the yaml file.

For a more detailed example, check out this [blog post (TBD)](TBD) that uses a Kubernetes cluster to fine tune
[meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf).
