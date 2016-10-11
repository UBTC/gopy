# gopy --- seting up [golang](https://golang.org/) and [iPython](http://ipython.org/)

## Installation of golang and Julia packages

0. Clone gopy to `~/.gopy`

1. Install golang packages:

    `sh ~/.gopy/install_go_pkg.sh`

    To check golang environment variable: `go env`

2. Install Julia packages:

    ``` sh
    julia ~/.gopy/install_jl_pkg.jl
    mkdir ~/julia/juliaFunc -p
    cp ~/.gopy/_juliarc.jl ~/.juliarc.jl
    ```

    To check all installed Julia package: `julia -e 'for i in Pkg.installed() println(i) end'`


## Setup iPython
Just use the subdir `profile_default` in `~/.gopy` to replace the current ipython configuration (`~/.config/ipython/profile_default`).

By the way,

- Run the command of `ipython profile create` to get an initial default configuration. Then subdir structure of ipython configuration may look like this:
``` sh
    db/
    log/
    pid/
    security/
    startup/
    static/
    history.sqlite
    ipython_config.py
    ipython_nbconvert_config.py
    ipython_notebook_config.py
    ipython_qtconsole_config.py
```

- To run IPython CLI: `ipython pylab`; to run IPython GUI: `ipython notebook --pylab=inline`.

### That is all, have fun!
