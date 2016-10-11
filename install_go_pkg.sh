#  !/usr/bin/env bash
#  -*- coding:utf-8 -*-

echo "Installing go packages"
# cd $HOME

echo "https://github.com/sbinet/go-gnuplot"
go get github.com/sbinet/go-gnuplot

echo "https://github.com/gonum/plot"
go get github.com/gonum/plot/...

echo "https://github.com/deckarep/golang-set"
go get github.com/deckarep/golang-set

echo "https://github.com/go-yaml/yaml"
go get gopkg.in/yaml.v2

echo "https://github.com/yuin/gopher-lua"
go get github.com/yuin/gopher-lua

echo "https://github.com/sabhiram/go-tracey"
go get github.com/sabhiram/go-tracey

echo "https://github.com/namsral/flag"
go get github.com/namsral/flag

echo "https://github.com/rogeralsing/gam"
go get github.com/gogo/protobuf/proto
go get github.com/gogo/protobuf/protoc-gen-gogo
go get github.com/gogo/protobuf/gogoproto
cd $GOPATH/src/github.com/gogo/protobuf/
make
cd -

echo "https://github.com/sjwhitworth/golearn"
go get -t -u -v github.com/sjwhitworth/golearn
cd $GOPATH/src/github.com/sjwhitworth/golearn
go get -t -u -v ./...
cd -

echo "done"
