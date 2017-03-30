# MassEvac

This repository contains all of the code that is required to replicate the work that I undertook during my thesis on modelling mass evacuation of cities using agent based modelling. This is a breakdown of how the folders are organised.

- [v1]: Early experiments
- [v2]: Videos for an earlier version of MassEvacDemo
- [v3]: PRE2016, TGF2015 paper
- [v4]: Thesis, Flood paper, Topology paper

For all intents and purposes, the instructions provided for setup below primarily only apply to [v4] as it contains the most up to date analyses presented in the [thesis]. The earlier versions [v1], [v2] and [v3] have been kept primarily to preserve the record of the state of the analyses during various stages of paper submissions and they have not been maintained in quite some time. One day, if I have more time, I will look into those too.

There are various Python 2.7 dependencies which will need to be installed as listed in [requirements.yml] by running the following command:

```bash
conda env create -f requirements.yml
source activate massevac
```

A local install of [OpenStreetMap] geospatial data and [GPWv4] population data on a PostgreSQL server is required. The instructions on how to do this are available in [SetupDB.sh] which has currently only been tested on Ubuntu 16.04 LTS.

You will need to add a file at `v4/.dbconfig` with your database credentials as follows:

```json
{
 "host": "localhost",
 "user": "username",
 "password": "password",
 "dbname": "gis"
}
```

It is also necessary to install `ffmpeg` to produce visualisation of evacuation as follows:

```bash
conda install -c menpo ffmpeg
chmod +x $(which ffmpeg)
```

After all the dependencies are met, test run an agent based evacuation simulation by running the following in the terminal:

```bash
cd v4/
python 0A_abm_test.py
```

[requirements.yml]: requirements.yml
[GPWv4]: http://sedac.ciesin.columbia.edu/data/collection/gpw-v4
[SetupDB.sh]: SetupDB.sh
[v1]: v1/
[v2]: v2/
[v3]: v3/
[v4]: v4/
[OpenStreetMap]: http://http://openstreetmap.org
[thesis]: http://www.github.com/brtknr/Thesis
