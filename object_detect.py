import os
import sys
import cv2
import time
from openvino.inference_engine import IENetwork, IECore
from matplotlib import pyplot as plt

def inferImage(image):
	in_frame = resizeInputImage(image)

	inf_start = time.time()
	
	res = exec_net.infer(inputs={input_blob: in_frame})

	inf_time = time.time() - inf_start
	print("Inference complete, run time: {:.3f} ms".format(inf_time*1000))	

	processResults(res)

def processResults(result):
	res = result[output_blob]

	for obj in res[0][0]:
		if obj[2] > prob_threshold:
			xmin = int(obj[3] * input_w)
			ymin = int(obj[4] * input_h)
			xmax = int(obj[5] * input_w)
			ymax = int(obj[6] * input_h)

			class_id = int(obj[1])
			
			if dic.get(class_id, 0):
				dic[class_id] += 1
			else: dic[class_id] = 1

			color = (min(class_id * 12.5, 255), 255, 255)
			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
			det_label = labels_map[class_id] if labels_map else str(class_id)
			cv2.putText(image, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
				cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

# define function to load an input image
def loadInputImage(input_path):
	# globals to store input width and height
	global input_w, input_h
	
	# use OpenCV to load the input image
	cap = cv2.VideoCapture(input_path) 

	# store input width and height
	input_w = cap.get(3)
	input_h = cap.get(4)
	print("Loaded input image [",input_path,"], resolution=", input_w, "w x ",input_h,"h")
	return cap

# define function for resizing input image
def resizeInputImage(image):
	# resize image dimensions form image to model's input w x h
	in_frame = cv2.resize(image, (w, h))
	# Change data layout from HWC to CHW
	in_frame = in_frame.transpose((2, 0, 1))
	# reshape to input dimensions
	in_frame = in_frame.reshape((n, c, h, w))
	return in_frame


#display input image
def displayOG(image):
	plt.axis("off")
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.title("Original image")
	plt.show()

#display after inference
def displayAfter(image):
	plt.axis("off")
	plt.title("Image after processing")
	plt.imshow(image)
	plt.show()

# model IR files
model_xml = "mobilenet-ssd.xml"
model_bin = os.path.splitext(model_xml)[0] + ".bin" 
labels_path = "labels.txt"
dic = {}

prob_threshold = 0.5

if __name__ == "__main__":

	device = sys.argv[1]
	
	ans = input("Print original image? y/n : ")

	print("Configuration parameters settings:"
     	"\n\tmodel_xml=", model_xml,
      	"\n\tmodel_bin=", model_bin,
      	"\n\tdevice=", device, 
      	"\n\tlabels_path=", labels_path, 
      	"\n\tprob_threshold=", prob_threshold)

	ie = IECore()
	print("A plugin object has been created for device", device)

	net = IENetwork(model=model_xml, weights=model_bin)
	print("Loaded model IR files")

	exec_net = ie.load_network(network=net, num_requests=2, device_name=device)

	input_blob = next(iter(net.inputs))
	output_blob = next(iter(net.outputs))

	n, c, h, w = net.inputs[input_blob].shape
	print("Loaded model into plugin. Model input dimensions: n=",n,", c=",c,", h=",h,", w=",w)

	labels_map = None

	with open(labels_path, 'r') as f:
		labels_map = [x.strip() for x in f]
	print("Loaded label mapping file")

	while(True):
		while(True):
			input_path = input("\nEnter image name or path (exit to exit): ")
			if input_path.strip():
				break

		if input_path == "exit": break

		#Load the input image
		cap = loadInputImage(input_path)
		ret, image = cap.read()
		
		if ans.lower() == "y":
			displayOG(image)

		inferImage(image)
		
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		displayAfter(image)
		for key, value in dic.items():
			print("\nNumber of{}s on the image is {}".format(labels_map[key].split(":")[1], value))
		dic.clear()













