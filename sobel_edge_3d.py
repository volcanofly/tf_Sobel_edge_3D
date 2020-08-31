import tensorflow as tf
def sobel_edge_3d(inputTensor):
    # This function computes Sobel edge maps on 3D images
    # inputTensor: input 3D images, with size of [batchsize,W,H,D,1]
    # output: output 3D edge maps, with size of [batchsize,W-2,H-2,D-2,3]
    sobel1 = tf.constant([1,0,-1],tf.float32) # 1D edge filter
    sobel2 = tf.constant([1,2,1],tf.float32) # 1D blur weight
    
    # generate sobel1 and sobel2 on x- y- and z-axis, saved in sobel1xyz and sobel2xyz
    sobel1xyz = [sobel1,sobel1,sobel1]
    sobel2xyz = [sobel2,sobel2,sobel2]
    for xyz in range(3):
        newShape = [1,1,1,1,1]
        newShape[xyz] = 3
        sobel1xyz[xyz] = tf.reshape(sobel1,newShape)
        sobel2xyz[xyz] = tf.reshape(sobel2,newShape)
        
    # outputTensor_x will be the Sobel edge map in x-axis
    outputTensor_x = tf.nn.conv3d(inputTensor,sobel1xyz[0],strides=[1,1,1,1,1],padding='VALID') # edge filter in x-axis
    outputTensor_x = tf.nn.conv3d(outputTensor_x,sobel2xyz[1],strides=[1,1,1,1,1],padding='VALID') # blur filter in y-axis
    outputTensor_x = tf.nn.conv3d(outputTensor_x,sobel2xyz[2],strides=[1,1,1,1,1],padding='VALID') # blur filter in z-axis
    
    outputTensor_y = tf.nn.conv3d(inputTensor,sobel1xyz[1],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_y = tf.nn.conv3d(outputTensor_y,sobel2xyz[0],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_y = tf.nn.conv3d(outputTensor_y,sobel2xyz[2],strides=[1,1,1,1,1],padding='VALID')
    
    outputTensor_z = tf.nn.conv3d(inputTensor,sobel1xyz[2],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_z = tf.nn.conv3d(outputTensor_z,sobel2xyz[0],strides=[1,1,1,1,1],padding='VALID')
    outputTensor_z = tf.nn.conv3d(outputTensor_z,sobel2xyz[1],strides=[1,1,1,1,1],padding='VALID')
    
    return tf.concat([outputTensor_x,outputTensor_y,outputTensor_z],4)
