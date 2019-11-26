#!/usr/bin/env bash
g++ -std=c++11 -shared -fPIC -o libHF.so main.cpp box_tracking.cpp Hungarian/Hungarian.cpp