# MassEvac

This repository contains all of the code that is required to replicate the work that I undertook during my thesis on modelling mass evacuation of cities using agent based modelling. This is a breakdown of how the folders are organised.

- [v1]: Early experiments
- [v2]: Videos for an earlier version of MassEvacDemo
- [v3]: PRE2016, TGF2015 paper
- [v4]: Thesis, Flood paper, Topology paper

For all intents and purposes, the instructions provided for setup below primarily only apply to [v4] as it contains the most up to date analyses presented in the [thesis]. The earlier versions [v1], [v2] and [v3] have been kept primarily to preserve the record of the state of the analyses during various stages of paper submissions and they have not been maintained in quite some time. One day, if I have more time, I will look into those too.

There are various Python 2.7 dependencies which will need to be installed as listed in [requirements.txt].

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
sudo add-apt-repository ppa:jonathonf/ffmpeg-3
sudo apt install ffmpeg
ffmpeg -version
```

This is the output of my `ffmpeg -version` command at the time of producing these instructions:

>      ffmpeg version 3.2.4-1~16.04.york1 Copyright (c) 2000-2017 the FFmpeg developers
>      built with gcc 5.4.1 (Ubuntu 5.4.1-5ubuntu2~16.04.york1) 20170210
>      configuration: --prefix=/usr --extra-version='1~16.04.york1' --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --enable-gpl --disable-stripping --enable-avresample --enable-avisynth --enable-gnutls --enable-ladspa --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libebur128 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libmp3lame --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvorbis --enable-libvpx --enable-libwavpack --enable-libwebp --enable-libx265 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-omx --enable-openal --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libopencv --enable-libx264 --enable-shared
>      libavutil      55. 34.101 / 55. 34.101
>      libavcodec     57. 64.101 / 57. 64.101
>      libavformat    57. 56.101 / 57. 56.101
>      libavdevice    57.  1.100 / 57.  1.100
>      libavfilter     6. 65.100 /  6. 65.100
>      libavresample   3.  1.  0 /  3.  1.  0
>      libswscale      4.  2.100 /  4.  2.100
>      libswresample   2.  3.100 /  2.  3.100
>      libpostproc    54.  1.100 / 54.  1.100

After all the dependencies are met, test run an agent based evacuation simulation by running the following in the terminal:

```bash
cd v4/
python 0A_abm_test.py
```

[requirements.txt]: requirements.txt
[GPWv4]: http://sedac.ciesin.columbia.edu/data/collection/gpw-v4
[SetupDB.sh]: SetupDB.sh
[v1]: v1/
[v2]: v2/
[v3]: v3/
[v4]: v4/
[OpenStreetMap]: http://http://openstreetmap.org
[thesis]: http://www.github.com/brtknr/Thesis
