#!/usr/bin/env bash

input=$1

mogrify -trim -gravity center -resize 150x150 -extent 150x150 "${input}"/*.png
