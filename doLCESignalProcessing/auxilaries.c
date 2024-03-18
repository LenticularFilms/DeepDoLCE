
//
// doLCE - do Lenticular film Color rEconstruction -
// Copyright (C) 2012 Joakim Reuteler

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as
// published by the Free Software Foundation.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// _______________________________________________________________________ 
//


// auxilaries.c

#include "auxilaries.h"

#define _GNU_SOURCE
#include <stdio.h>

//#include <string.h>



int set_greeter( char **greeter ){
  int bites;
  bites = asprintf( greeter, "*\n**\n*** doLCE - do Lenticular film Color rEconstruction ***\n**\n*\n[built on %s ~ %s]\n\n", __DATE__, __TIME__);
  return bites;
}


int set_copyright( char **copyright ){
  int bites;
  bites = asprintf( copyright, "This program is free software under the terms of the GNU General Public License version 3.\nSee <http://www.gnu.org/licenses/>.\n" );
  return bites;
}


int set_help( char **help ){
  int bites;
  bites = asprintf( help, "doLCE [-help] [-mode (int) (int) (int)] [-profileRelThickness (float)] [-profileRelPosY (float)] [-relaxRaster (int)] [-rasterSpacing (float)] [-troubleshoot] 'inputDir' 'inputBaseName' 'startNo' 'endNo' 'outputDir'\n\nmode a b c\na: 1 = deconvolve colors, 2 = color output as read\nb: binning of b lines into one for output\nc: 1 = 1 raster is one pixel, 2 = edge aware upres by factor 2, 3 = one scan pixel is one output pixel (this is incompatible with a=1\n" );
  return bites;
}
