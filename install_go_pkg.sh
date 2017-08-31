#  !/usr/bin/env bash
#  -*- coding:utf-8 -*-

echo "Installing go packages"
# cd $HOME

echo
echo "https://github.com/sbinet/go-gnuplot"
go get github.com/sbinet/go-gnuplot

echo
echo "https://github.com/gonum/plot"
go get github.com/gonum/plot/...

echo
echo "go get gopkg.in/fatih/set.v0"
#go get github.com/deckarep/golang-set
go get gopkg.in/fatih/set.v0

echo
echo "https://github.com/go-yaml/yaml"
go get gopkg.in/yaml.v2

echo
echo "https://github.com/yuin/gopher-lua"
go get github.com/yuin/gopher-lua

echo
echo "https://github.com/sabhiram/go-tracey"
go get github.com/sabhiram/go-tracey

echo
echo "https://github.com/namsral/flag"
go get github.com/namsral/flag

echo
echo "https://github.com/rogeralsing/gam"
go get github.com/gogo/protobuf/proto
go get github.com/gogo/protobuf/protoc-gen-gogo
go get github.com/gogo/protobuf/gogoproto
cd $GOPATH/src/github.com/gogo/protobuf/
make
cd -

echo
echo "https://github.com/sjwhitworth/golearn"
go get -t -u -v github.com/sjwhitworth/golearn
cd $GOPATH/src/github.com/sjwhitworth/golearn
go get -t -u -v ./...
cd -

echo "done"
