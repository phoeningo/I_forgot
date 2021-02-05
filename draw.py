import os
import mrcfile
import sys
import argparse
import numpy as np
from numba import cuda,jit
import time
import math
from skimage import measure,exposure,morphology

parser=argparse.ArgumentParser(description='T')

parser.add_argument('--good_star',type=str)
parser.add_argument('--bad_star',type=str)

parser.add_argument("--input",type=str)
parser.add_argument('--mode',type=str,default='global')
parser.add_argument("--output",type=str,default='temp.mrc')
parser.add_argument("--thre",type=int,default=0)
parser.add_argument("--size",type=int,default=30)
parser.add_argument("--min_size",type=int,default=30)
parser.add_argument("--max_size",type=int,default=150)
parser.add_argument("--x",type=int,default=3838)
parser.add_argument("--y",type=int,default=1000)
parser.add_argument("--z",type=int,default=3838)
parser.add_argument('--min_distance',type=int,default=100)
parser.add_argument('--max_distance',type=int,default=300)

args=parser.parse_args()

#======================data block============================#
# define new empty TOMO
# define temp subtomo
# subtomo list -< star file

# def names for seek

s_name='_rlnImageName'
s_x='_rlnCoordinateX'
s_pxsize='_rlnDetectorPixelSize'
if args.mode=='local':
  s_offx='_rlnOriginX'
  slist=[s_name,s_x,s_offx,s_pxsize]
if args.mode=='global':
  slist=[s_name,s_x,s_pxsize]

print(slist)
#

#=========function ===========

#def draw lines:|__ draw box:

def draw_line(startpoint,endpoint):
  return 

def draw_box(centerpoint,size):
  return

@cuda.jit
def fill(big,sub,sx,sy,sz):
   i,j,k=cuda.grid(3)
   big[sx-i,sy-j,sz-k]=sub[i,j,k]

def draw_subtomo(big,sub,center):
  six,siy,siz=sub.shape
  cx,cy,cz=center
  sx=max(0,cx-six/2)
  sy=max(0,cy-siy/2)
  sz=max(0,cz-siz/2)
  griddim=[six,siy,siz]
  blockdim=1
  fill[griddim,blockdim](big,sub,np.int32(sx),np.int32(sy),np.int32(sz))
  return

def deepsplit(listname,sp):
  newlist=[]
  for item in listname:
    item_split=item.split(sp)
    for it_sp in item_split:
      newlist.append(it_sp)
  return newlist
      
def get_lines(filename):
  file_open=open(filename)
  context=file_open.read()
  lines=context.split('\n')
  file_open.close()
  return lines

def removeblank(listname):
  new_list=[]
  for item in listname:
    if item!='':
      new_list.append(item)
  return new_list

def gethead(filename):
  all_lines=get_lines(filename)
  headlist=[]
  current=0
  print(len(slist))
  while current<len(slist):
    for line in all_lines:
      if line.find(slist[current])!=-1:
        col=line.split(' ')[1]
        col_num=int(col.split('#')[1])
        headlist.append(slist[current])
        headlist.append(col_num)
        current+=1
        break;

  return headlist

def get_col(listname,name):
  return listname[listname.index(name)+1]

def readstar(filename):
  lines=get_lines(filename)
  head=gethead(filename)
  #print(head)
  coordlist=[]
  namelist=[]      
  for line in lines:
    line_split=line.split('\t')
    line_split=deepsplit(line_split,' ')
    line_split=removeblank(line_split)
    #print(line_split)
    if len(line_split)>4:
      nam=get_col(head,s_name)-1
      namelist.append(line_split[nam])
      col=get_col(head,s_x)-1
      #print(col)
      x=float(line_split[col])
      y=float(line_split[col+1])
      z=float(line_split[col+2])
      coordlist.append([x,y,z])
      if args.mode=='local':
        off=get_col(head,s_offx)-1
        ox=float(line_split[off])
        oy=float(line_split[off+1])
        oz=float(line_split[off+2])
        coordlist.append([x,y,z,ox,oy,oz])
  return coordlist,namelist

  

#==============================================================
#----main ----


#
#good_subtomo=readstar(args.good_star)

head=gethead(args.good_star)

print(head)
coordlist,namelist=readstar(args.good_star)
#print(coordlist)
'''
for name in slist:
  print(name,get_col(head,name))
'''
x=args.x
y=args.y
z=args.z
new_tomo=np.zeros([x,y,z],dtype=np.float32)
i=0
o=mrcfile.new_mmap(args.output,overwrite='True',shape=new_tomo.shape,mrc_mode=0)
for subname in namelist:
  sub_open=mrcfile.open(subname,'r+')
  sub_data=sub_open.data
  sub_open.close()
  sub=np.float32(sub_data)
  draw_subtomo(new_tomo,sub,coordlist[i])

o.set_data(new_tomo)
o.flush()
o.close()
    #print(i,coords[0])
  


