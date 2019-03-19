from model import EDSR
import scipy.misc
import argparse
import data
import os
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data/test_data/set5/")
parser.add_argument("--imgsize",default=100,type=int)
parser.add_argument("--scale",default=2,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=256,type=int)
parser.add_argument("--batchsize",default=10,type=int)
parser.add_argument("--savedir",default="saved_models")
parser.add_argument("--iterations",default=1000,type=int)
parser.add_argument("--numimgs",default=5,type=int)
parser.add_argument("--outdir",default="out")
parser.add_argument("--image", default="butterfly_GT.bmp")
args = parser.parse_args()
if not os.path.exists(args.outdir):
	os.mkdir(args.outdir)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,scale=args.scale)
network.resume(args.savedir)
'''if args.image:
	x = scipy.misc.imread(args.dataset+args.image)
else:
	print("No image argument given")'''

cap = cv2.VideoCapture('456.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = cap.get(cv2.CAP_PROP_FPS)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
print(length)
while(cap.isOpened()):
	ret, frame = cap.read()

	if ret:
		outputs = network.predict(frame)
		outputs = outputs.astype(np.uint8)
		cv2.imshow('video', outputs)
		if count == 0:
			out = cv2.VideoWriter('output.avi', fourcc, fps, (int(outputs.shape[1]), int(outputs.shape[0])))
			count += 1
		out.write(outputs)
	else:
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

out.release()
cap.release()
cv2.destroyAllWindows()

'''
inputs = x
outputs = network.predict(x)
if args.image:
	scipy.misc.imsave(args.outdir+"/input_data/"+args.image,inputs)
	scipy.misc.imsave(args.outdir+"/output_data/"+args.image,outputs)
'''
