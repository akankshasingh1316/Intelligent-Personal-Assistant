import tensorflow as tf
import numpy as np 
import csv

import os



def add_layer(inputs,in_size,out_size,activation_function=None):
	#global sess
	weights = tf.Variable(tf.random_normal([in_size,out_size])) #### 2-D 
	weights2 = tf.Variable(tf.random_normal([in_size,out_size]))
	weights3 = tf.Variable(tf.random_normal([in_size,out_size]))
	
	bias = tf.Variable(tf.zeros([1,out_size]) +0.1) ### 1 row and out_size columns
	#print(sess.run(bias))
	comp = tf.matmul(inputs,weights) +bias ### wght*x + bias
	#print ("here")
	if activation_function is None:
		out = comp

	else:
		out = activation_function(comp)
	return out

def compute_accuracy(v_xs,v_ys):
	global prediction
	y_pre = sess.run(prediction,feed_dict={xs:v_xs})
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(((tf.cast(correct_prediction,tf.float32))))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})

	
	#print(sess.run(accuracy))
	#print(sess.run(result))
	return result


## define placeholder

xs = tf.placeholder(tf.float32,[None,1]) # 28*28
ys = tf.placeholder(tf.float32,[None,3]) ### output have the 10 positions or classes to represent

#add output layer only no hidden layer 


prediction = add_layer(xs,1,3,activation_function=tf.nn.softmax) ## softmax is used to calculate the probability of each class and choose the highest probability number as the class 



## error detection between prediction and real data and using CROSS ENTROPY
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

####IMPOrtant step
j = 0
with open("answer.csv") as fl:
	for line in fl:
		j=j+1





s = (j,1)
result2 = np.ones((s),dtype=np.float32)

# sess.run(tf.global_variables_initializer())

reader = csv.reader(open("answer.csv","rb"),delimiter=',')
x = list(reader)
result1 = np.array(x).astype(np.float32)



value = " "
col2 = 0
col3 = 0
col4 = 0
i = 0.0
cls = [ ]
filename = "dataset.csv"
filename2 = "test.csv"
	#filename = "dataset.csv"

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
#print((result1))
#print((result2))
with open(filename) as fl:
	
	for line in fl:
		i= i + 1
		value,col2,col3,col4 = line.strip().split(",")
		col2 = tf.string_to_number(col2)
		col3 = tf.string_to_number(col3)
		col4 = tf.string_to_number(col4)
		

		batch_ys= np.array([[sess.run(col2),sess.run(col3),sess.run(col4)]])
		
		batch_xs = np.ones((1,1))
		#print(batch_xs,batch_ys,value)

		sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
		print(sess.run(cross_entropy,feed_dict={xs:batch_xs,ys:batch_ys}),"CROSS ENTROPY")
		if i%20== 0:
			# print((np.shape(batch_ys)))
			print(compute_accuracy(result2,result1) ,"THIS IS THE ACCURACY")
			
	print("Training finish")

	#print(np.shape((mnist.test.images)))
	print(np.shape(result2))
	print(np.shape(result1))

opt = "yes"
while opt == "yes":
	input_string = raw_input("Enter the string to be tokenize :  ")
	list1 = []
	list1 = input_string.split(" ")
	print(list1)


	for x in list1:
		count = 0
		with open("2of12id.txt") as fl:
			for line in fl:
				lines = line.split()
				for y in range (len(lines)):

					if(x == lines[y]):
						count = count+1
						k = lines[1]
						#print(lines[y])
						#print(k)

						#print(lines[y])
					#print(lines[0])
			
			if(count >= 1):
				if(k == 'N:' or k == 'P:'):
					print("Found",x," & class is :" ,"objects")


				elif(k == 'V:'):
					print("Found",x," & class is : ","actions")

				else:
					print("Found",x," & class is :","unidentified")
			else:
				print("Found",x," & class is :","unidentified")

	opt = raw_input("Do you want to check another statement ?    : ") 





com = raw_input("Do you want to test OS Commands?    ")


while(com == "yes"):
	oscom = raw_input("Enter the command to execute : ")


	if(oscom == "create a text file"):
		os.system("touch test.txt")
		com = raw_input("Do you want to continue os commands ? :")

	elif(oscom == "open application"):
		os.system("open -a Google\ Chrome")
		com = raw_input("Do you want to continue os commands ? :")
	elif(oscom == "play a song"):
		os.system("afplay -t 10 /Users/siddharthagarwal/Desktop/Media/Songs/test.mp3")
		com = raw_input("Do you want to continue os commands ? :")

	elif(oscom == "delete a file"):
		os.system("rm test.txt")
		com = raw_input("Do you want to continue os commands ? :")


	else:
		pass


















