# Installing AgentNet

A native way to install AgentNet is by using pip
We also provide a platform-agnostic Docker container with AgentNet and most popular analytical libraries.

Any assistance with AgentNet installation, as well as your feedback, ideas and bug reports are very welcome.


__If you have a Windows-based or otherwise non-mainstream operating system and generally prefer avoiding trouble,
 we recommend using docker installation__



## Native
This far the installation was only tested on Ubuntu, Windows 7 and random Mac OS,
yet an experienced user is unlikely to have problems installing it onto other Linux or Mac OS Machine

Currently the minimal dependencies are __bleeding edge__ Theano and Lasagne.
You can find a guide to installing them here
* http://lasagne.readthedocs.io/en/latest/user/installation.html#bleeding-edge-version

If you have both of them, you can install agentnet with
* `[sudo] pip install --upgrade https://github.com/yandexdataschool/AgentNet/archive/master.zip`

If you want to explore the examples folder, we recommend installing from repository
```
 git clone https://github.com/justheuristic/AgentNet
 cd AgentNet
 python setup.py install
```

Developer installation
```
 git clone https://github.com/justheuristic/AgentNet
 cd AgentNet
 python setup.py develop
```


## [Docker container](https://hub.docker.com/r/justheuristic/agentnet/)

We use [Yandex REP](https://github.com/yandex/rep) container to provide data analysis tools (Jupyter, Matplotlib, Pandas, etc)

__To download/install/run the container, use__
 1. install [Docker](http://docs.docker.com/installation/),
 2. make sure `docker` daemon is running (`sudo service docker start`)
 3. make sure no application is using port 1234 (this is the default port that can be changed)
 4. `[sudo] docker run -d -p 1234:8888 justheuristic/agentnet`
 5. Access via localhost:1234 or whatever port you chose
This installation contains an installation of AgentNet, along with latest Theano and Lasagne libraries.



## Notes for windows
We recommend running the docker container, using docker-machine (see docker install above).

Technically if you managed to get Theano and Lasagne working on Windows, you can follow the Linux instruction.
However, we cannot guarantee that this works on all Windows distributions.

A generic guide on how to install lasagne on windows can be found in [this awesome tutorial](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne).
If you already have Anaconda installed, we recommend these
 - Anaconda python 2 [here](http://stackoverflow.com/questions/33687103/how-to-install-theano-on-anaconda-python-2-7-x64-on-windows)
 - Anaconda python 3 [here](http://ankivil.com/installing-keras-theano-and-dependencies-on-windows-10/)


