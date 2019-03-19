# Tiny_Yolo3
Tiny_Yolo3 based on Pytorch (#including training)

Note:
	1. darknet uses kernels in Caffe-style : (out_channels, in_channels, kernel_height, kernel_width)
	
	2. Darknet uses left and top padding instead of 'same' mode
	e.g.: 11 max: uses left and top padding instead of 'same' mode
  15 conv 18 conv 22 conv : padding = 0

	3. yolo uses (x, y, w, h, objness, class probs)
  
  

The weight of the official website was used for testing, and the accuracy was low.
e.g.: dog.jpg
[tensor([[ 70.2367, -34.6988, 555.1447, 218.6484,   0.9345,   0.5558,   7.0000],
        [255.0177,  60.3746, 366.7945, 122.2587,   0.9700,   0.7868,   2.0000],
        [ 66.3961, 158.9476, 209.7903, 374.7690,   0.7907,   0.9940,  16.0000]])]
data/dog.jpg: Predicted in 0.373784 seconds.
	 Label: truck, Conf: 0.55585
	 Label: car, Conf: 0.78678
	 Label: dog, Conf: 0.99401
save plot results to prediction.jpg

The bounding box is too large, and the recognition is poor for the overlapping object .
