#!/usr/bin/env bash

input=$1

mogrify -trim -gravity center -resize 64x64 -extent 64x64 "${input}"/*.png
