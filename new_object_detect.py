import os
import sys
import cv2
import time
from openvino.inference_engine import IENetwork, IECore
from matplotlib import pyplot as plt

class InferImage:
	def __init__(self, device):
		self.model_xml = "mobilenet-ssd.xml"
		self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin" 
		self.labels_path = "labels.txt"
		self.device = device
	
		self.dic = {}

		self.prob_threshold = 0.5

		print("Configuration parameters settings:"
     		"\n\tmodel_xml=", self.model_xml,
      		"\n\tmodel_bin=", self.model_bin,
      		"\n\tdevice=", self.device, 
      		"\n\tlabels_path=", self.labels_path, 
      		"\n\tprob_threshold=", self.prob_threshold)

		self.ie = IECore()
		print("A plugin object has been created for device", device)		

		self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
		print("Loaded model IR files")	

		self.exec_net = self.ie.load_network(network=self.net, num_requests=2, device_name=self.device)

		self.input_blob = next(iter(self.net.inputs))
		self.output_blob = next(iter(self.net.outputs))

		self.n, self.c, self.h, self.w = self.net.inputs[self.input_blob].shape
		print("Loaded model into plugin. Model input dimensions: n=",self.n,", c=",self.c,", h=",self.h,", w=",self.w)
		
		self.labels_map = None
		with open(self.labels_path, 'r') as f:
			self.labels_map = [x.strip() for x in f]
		print("Loaded label mapping file")

		self.input_w = 0
		self.input_h = 0

	def inferImage(self, image):
		in_frame = self.resizeInputImage(image)

		inf_start = time.time()
	
		res = self.exec_net.infer(inputs={self.input_blob: in_frame})

		inf_time = time.time() - inf_start
		print("Inference complete, run time: {:.3f} ms".format(inf_time*1000))	

		self.processResults(res, image)

	def processResults(self, result, image):
		res = result[self.output_blob]

		for obj in res[0][0]:
			if obj[2] > self.prob_threshold:
				xmin = int(obj[3] * self.input_w)
				ymin = int(obj[4] * self.input_h)
				xmax = int(obj[5] * self.input_w)
				ymax = int(obj[6] * self.input_h)

				class_id = int(obj[1])
			
				if self.dic.get(class_id, 0):
					self.dic[class_id] += 1
				else: self.dic[class_id] = 1

				color = (min(class_id * 12.5, 255), 255, 255)
				cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
				det_label = self.labels_map[class_id] if self.labels_map else str(class_id)
				cv2.putText(image, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
					cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

	def loadInputImage(self, input_path):
		# use OpenCV to load the input image
		cap = cv2.VideoCapture(input_path) 

		# store input width and height
		self.input_w = cap.get(3)
		self.input_h = cap.get(4)
		print("Loaded input image [",input_path,"], resolution=", self.input_w, "w x ",self.input_h,"h")
		return cap

	# define function for resizing input image
	def resizeInputImage(self, image):
		# resize image dimensions form image to model's input w x h
		in_frame = cv2.resize(image, (self.w, self.h))
		# Change data layout from HWC to CHW
		in_frame = in_frame.transpose((2, 0, 1))
		# reshape to input dimensions
		in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
		return in_frame

	#display input image
	def displayOG(self, image):
		plt.axis("off")
		plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		plt.title("Original image")
		plt.show()

	#display after inference
	def displayAfter(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		plt.axis("off")
		plt.title("Image after processing")
		plt.imshow(image)
		plt.show()

	def doit(self, path):
		cap = self.loadInputImage(path)
		ret, image = cap.read()	

		# self.displayOG(image)

		self.inferImage(image)
		for key, value in self.dic.items():
			print("\nNumber of{}s on the image is {}".format(self.labels_map[key].split(":")[1], value))
		self.dic.clear()
		self.displayAfter(image)

		

'''
a = InferImage("CPU")
cap = a.loadInputImage("car1.jpg")
ret, image = cap.read()

a.displayOG(image)

a.inferImage(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
a.displayAfter(image)
'''
		





