# !/usr/bin/env julia
# -*- coding:utf-8 -*-
# collected from everywhere of the internet. Thanks a lot!~~

push!(LOAD_PATH, ENV["JULIAFUNCDIR"])
push!(LOAD_PATH, "/usr/local/bin", "/opt", "/usr/bin", "/bin")
push!(LOAD_PATH, pwd())

ENV["JULIA_SHELL"] = "/usr/bin/zsh"
ENV["JULIA_ANSWER_COLOR"] = "magenta"

using IJulia
using PyPlot
using PyCall
using Sundials
using Polynomials
using Distributions

const SEPARATOR = "-  " ^ 25
separator() = (println(); print_with_color(:cyan, SEPARATOR); println())

paste() = include_string(clipboard())

@pyimport numpy as np
@pyimport scipy as sp
@pyimport matplotlib as mpl
@pyimport scipy.linalg as la
@pyimport scipy.stats as stats
@pyimport statsmodels.api as sm
@pyimport matplotlib.mlab as mlab
@pyimport matplotlib.pylab as plab
@pyimport matplotlib.pyplot as pplt

pplt.rc("text", usetex=true)     # Use Latex formatting
pplt.rc("font", size=15)         # Fontsize
pplt.rc("xtick", labelsize=15)   # Fontsize for x-ticks
pplt.rc("ytick", labelsize=15)   # Fontsize for y-ticks
pplt.rc("legend", fontsize=15)

pplt.rc("axes", edgecolor="bcbcbc")     # axes edge color
pplt.rc("axes", linewidth=2)            # edge linewidth
pplt.rc("axes", grid=true)              # display grid or not
pplt.rc("axes", titlesize="x-large")    # fontsize of the axes title
pplt.rc("axes", labelsize="large")      # fontsize of the x any y labels
pplt.rc("axes", labelcolor="111111")    # font color of labels
pplt.rc("axes", axisbelow=true)         # whether axis gridlines and ticks are below the axes elements (lines, text, etc)

pplt.rc("xtick.major", size = 0)      # major tick size in points
pplt.rc("xtick.minor", size = 0)      # minor tick size in points
pplt.rc("xtick.major", pad = 6)       # distance to major tick label in points
pplt.rc("xtick.minor", pad = 6)       # distance to the minor tick label in points
pplt.rc("xtick", color = "111111")    # color of the tick labels
pplt.rc("xtick", direction = "in")    # direction: in or out

pplt.rc("ytick.major", size = 0)      # major tick size in points
pplt.rc("ytick.minor", size = 0)      # minor tick size in points
pplt.rc("ytick.major", pad = 6)       # distance to major tick label in points
pplt.rc("ytick.minor", pad = 6)       # distance to the minor tick label in points
pplt.rc("ytick", color = "111111")    # color of the tick labels
pplt.rc("ytick", direction = "in")    # direction: in or out

pplt.rc("grid", color="black")
pplt.rc("grid", linestyle=":")
pplt.rc("grid", linewidth=0.5)

pplt.rc("axes", color_cycle=["332288", "88ccee", "44aa99", "117733", "999933", "ddcc77", "cc6677", "882255", "aa4499"])
pplt.rc("lines", antialiased=true) # render lines in antialised (no jaggies)
pplt.rc("legend", fancybox=true)  # if True, use a rounded box for the legend, else a rectangle
# http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
pplt.rc("figure", figsize = (12, 9)) # figure size in inches
