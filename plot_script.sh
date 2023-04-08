#!/bin/bash

gnuplot -e "
set terminal pngcairo background rgb '#ffffffff';
set xrange[0:30];
set yrange[0:30];
set cbrange[19:100];
n_frames = 500;
do for [i=0:n_frames-1]
{
  set output sprintf('images/heat_diffusion%03d.png', i);
  plot 'heat_diffusion.dat' index i with image;
}
"
convert -verbose -loop 0 -delay 10 images/heat_diffusion* animation.gif
