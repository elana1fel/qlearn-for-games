# qlearn-for-

for learning run it with:
KERAS_BACKEND=theano THEANO_FLAGS=floatX=float32,device=gpu,force_device=True,cuda.root=/usr/local/cuda,lib.cnmem=0.2 python ./qlearn.py -m "Train" -l 1

if you want to use it with existing traing  weights file run it with:
KERAS_BACKEND=theano THEANO_FLAGS=floatX=float32,device=gpu,force_device=True,cuda.root=/usr/local/cuda,lib.cnmem=0.2 python ./qlearn.py -m "Run" 

you need to change the game file for different game
