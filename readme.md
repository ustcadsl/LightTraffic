# LightTraffic

LightTraffic provides support for large-scale random walks for hugh graphs on GPUs with the aim of reducing CPU-GPU data transmission traffic.

## Start

### Environment

Requirements:
* Ubuntu 20.04
* CUDA >= 11.5.0
* g++ >= 9.4.0

We also provide the Docker image which satisfies the above requirements. Please follow this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to install Docker with cuda support.
After the installation process is complete, you can build and run the image by the following commands:
```
make build-docker
make run-docker
```

### Datasets

LightTraffic adopts Binary CSR (bcsr) format. The id of vertices and edges are represented by unsigned 32-bit integer and unsigned 64-bit interger respectively. The header of bcsr has 12 bits storing the number of vertices and edges. The rest of bcsr is CSR format of the adjacent matrix, which consists of the row pointer array and the column array.

The [Friendster](http://konect.cc/networks/friendster/) dataset comes from the KONECT project. The [UK-Union](https://law.di.unimi.it/webdata/uk-union-2006-06-2007-05/) dataset comes from the Laboratory for Web Algorithmics. The [Yahoo](https://webscope.sandbox.yahoo.com/) dataset comes from Yahoo research. The [ClueWeb09](https://nrvis.com/download/data/massive/web-ClueWeb09.zip) dataset comes form the network repository. Other three datasets [LiveJournal](https://snap.stanford.edu/data/soc-LiveJournal1.html), [Orkut](https://snap.stanford.edu/data/com-Orkut.html) and [Twitter](https://snap.stanford.edu/data/twitter-2010.html). There are two options to get the data.

* (Recommended) Download the preprocessed graph directly.
```
make dataset
```

* If you want to run on a new graph, you should convert graph from text of edgelist to the bcsr format.

```
./converter_txt graph.el
```

## Compile & Run

The programs are compiled with make:

```
make
```

To reproduce the results in the paper, you first make sure all datasets are available in the directory /LightTraffic/dataset, and then run the shell script.
```
make dataset
./evaluation.sh
```

## Options

Here we take pagerank as an example to show how to run the built-in applications. The usage of other applications is quite similar. The "-f" option must be given to specify the graph file.

```
./ppr -f livejournal.bcsr
```

We provide 5 options in algorithmic perspective.

* "-w" option specifies the number of walks computed in a single run (default 2|V|).

* "--runs" option specifies the number of runs (default 1).

* "-l" option specifies the walk length for fixed-length random walks (default 10).

* "--source" option specifies the source vertex for PPR (default 0).

* "--prob" option specifies the probability that a walk terminates at each step (default 0.15).

We provide 4 options to fine-tune the system.

* "-p" option specifies the number of graph partitions cached on GPU memory (default: the number of graph partitions).

* "-b" option specifies the number of walks cahced on GPU memory (default: the number of walks).

* "-m" option will materialize the partition to disk to avoid recomputation. If this option is not given, LightTraffic would look for the partition file. This option must be given if this is the first time to compute random walks on this graph.

* "-s" option specifies the maximum size of graph partition in Megabyte (default: 32MB).
