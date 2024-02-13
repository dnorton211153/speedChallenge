# @author Norton 2023 (AI/Machine Learning capstone project - Roadsign Detection
#
# Objective: Use Keras to train a model which can identify roadsigns in a video.

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import keras
import os
# import speedTest.apps as apps

modelFile = "../model.h5"
learning_rate = 0.05
# The learning_rate is the rate at which the model learns.
# The learning_rate is multiplied by the gradient to determine the amount to adjust the weights.
# The learning_rate is a hyperparameter that can be tuned to improve the model, like how much
# room to wiggle the weights when adjusting them on each pass.

epochs = 10
batch_size = 20
neurons = batch_size * 2
testMode = True
filename = "speedTestTrainingData/test.mp4"

def test_single_image():
    frames = loadTestingData()
    
    # Remove the second dimension
    frames = np.squeeze(frames, axis=1)
    print("frames.shape = ", frames.shape)
    
    image = frames[400]
    superImposedImage = superimposeImage(18.01, image)
    display_image("superimposed", superImposedImage)
    imageFinal = image_with_lanes(superImposedImage)
    display_image("final", imageFinal)
    return 
    
# Lane finding Pipeline based on https://www.kaggle.com/code/soumya044/lane-line-detection
def display_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def image_with_lanes(image):

    # Convert the image to grayscale
    gray = grayscale(image)

    # Apply Gaussian blur to the image
    blur = gaussian_blur(gray, 5)

    # Apply Canny edge detection to the image
    edges = canny(blur, 50, 150)

    # Identify the vertices of the wedge shape
    height = image.shape[0]
    width = image.shape[1]
    vertices = np.array(
        [[(0, height-10), (width, height-10), (width/2, height*0.45)]], dtype=np.int32)

    # Ignore everything outside the wedge
    roi = region_of_interest(edges, vertices)

    # Identify contiguous line segments in the image, straight or curved
    lines = identify_line_slopes(roi)

    # Draw the lines on the image
    line_image = draw_lines(image, lines)

    output = line_image
    return output

# Identify the slopes of the contiguous white lines in the image
def identify_line_slopes(img):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # "contours": a matrix with 3 columns:
    # "id": the contour identity (indicates the set of points belonging to the same contour).
    # "x": the x coordinates of the contour points.
    # "y": the y coordinates of the contour points.

    # reduce contours to only contain the two largest contours (most number of matching id's)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # For each line, calculate the average slope
    # empty numpy array to store each line, where each line is a 4-element array
    lines = []
    for contour in contours:
        # print("contour = " + str(contour))
        x, y, w, h = cv2.boundingRect(contour)
        lines.append([x, y, x + w, y + h])

    print("lines = " + str(lines))
    return lines

# Draw the lines on the image
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # Create a blank image that matches the original image
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # Does lines contain any data?
    if lines is None:
        return img

    # convert lines to a numpy array
    lines = np.array(lines)

    # Iterate over the lines and draw them on the blank image
    for line in lines:
        line = np.array(line)
        print("line = " + str(line))
        cv2.line(line_img, (line[0], line[1]), (line[2], line[3]), color, thickness)

    # Overlay img and line_img
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return img 

# Region of interest - return an image of only the region of interest, inside the vertices
def region_of_interest(img, vertices):
    # Define a blank mask to start with
    mask = np.zeros_like(img)

    # Fill the mask
    cv2.fillPoly(mask, vertices, 255)

    # Return the image only where the mask pixels are not zero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Grayscale the image
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Gaussian blur the image
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Canny edge detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def shrink_img(img):
    resized_img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    return resized_img

# Load the training data
def loadTrainingData():

    # Load the video file
    cap = cv2.VideoCapture("speedTestTrainingData/train.mp4")
    # Get the number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 1000
    if (testMode is True):
        num_frames = 1000
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create an array to store the frames using window_nd
    frames = np.empty((num_frames, int(height/2), int(width/2), 3), np.dtype('uint8'))

    # Load the train.txt file
    train = np.loadtxt("speedTestTrainingData/train.txt")

    # Read the frames
    for i in range(num_frames):
        if i % 100 == 0:
            print("Loading training frame", i)
        frames[i] = shrink_img(cap.read()[1])

    # Close the video file
    cap.release()

    print("frames.shape = ", frames.shape)
    print("train.shape = ", train.shape)

    frames = np.expand_dims(frames, axis=1)
    train = np.expand_dims(train, axis=1)

    # Return the frames
    return frames[0:num_frames], train[0:num_frames]


def checkIfModelExists():
    # If the model file exists, load the model
    if os.path.exists(modelFile):
        print("WARNING: Loading model from pre-existing file.  Delete the file to retrain the model from scratch.")
        model = keras.models.load_model(modelFile)
        model.summary()
        return True, model
    else:
        return False, None

# Define and return the compiled model


def createModel(frameLength):

    model = keras.models.Sequential(
        [
            keras.Input(batch_size=batch_size, 
                        batch_shape=(batch_size, 1, 240, 320, 3)),
        ]
    )
    
    # All things being equal, why not Dropout right at the beginning?
    model.add(keras.layers.TimeDistributed(keras.layers.Dropout(0.2)))
    
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Conv2D(
                neurons, (3, 3), strides=(2, 2), activation='relu')
    ))

    # flatten each frame
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Flatten() 
        )
    )   

    currentShape = model.output_shape
    print("model.output_shape before LSTM = " + str(currentShape))

    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(neurons, return_sequences=True, 
        stateful=True, unroll=False, use_bias=True,
        # Attempting to ignore the second dimension with batch_input_shape (?)
        batch_size=batch_size, batch_input_shape=(batch_size, None, currentShape[2]),
    )))

    # model.add(keras.layers.Bidirectional(
    #     keras.layers.LSTM(neurons, return_sequences=False, 
    #     stateful=True, unroll=False, use_bias=True,
    # )))

    model.add(
        keras.layers.Dense(neurons)
    )  
        
    model.add(
        keras.layers.Dense(1)
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, decay=learning_rate/epochs)
    loss = keras.losses.MeanSquaredError()
    metrics = [keras.metrics.MeanSquaredError()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.build()

    model.summary()
    return model

# trainModel will train the model on the training data, save the model, and return the model
def trainModel(model, frames, train):

    # Convert frames and train from numpy arrays to keras pydatasets
    model.fit(frames, train, epochs=epochs, batch_size=batch_size, shuffle=False, validation_split=0.2, verbose=2)

    model.save(modelFile)
    return

# testModel will test the model on the test data and create a prediction file
def testModel(testFrames):

    trainedModel = keras.models.load_model(modelFile)
    # testFrames.shape is currently (12630, 32, 32, 3)
    print("testFrames.shape = ", testFrames.shape)
    outputFromPredict = trainedModel.predict(testFrames, batch_size=batch_size, verbose=2)
    # outputFromPredict.shape should be (12630, 43),
    # where each row has the array of probabilities that the image belongs to each class
    print("outputFromPredict.shape = ", outputFromPredict.shape)

    # frameLength = first dimension of testFrames
    frameLength = testFrames.shape[0]
    # print("frameLength = ", frameLength, " will determine rows of predictions.txt file.")
    
    predictions = np.empty((frameLength, 1), np.dtype('float'))
    # print(outputFromPredict)
    for i in range(0, frameLength):
        
        # get the highest-probability class per row
        # greatestChange = np.argmax(outputFromPredict[i]).round(2).astype(float)
        predictions[i] = outputFromPredict[i][0].round(4).astype(float)

    # Create a prediction file from the predictions
    np.savetxt("predictions.txt", predictions, fmt='%d')
    
    return predictions

# publishPredictions will superimpose the predictions on the testFrames,
# add the lanelines, and display the resulting video 
def publishPredictions(predictions, testFrames, filename = filename):

    newVideo = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (320, 240))

    # Remove the second dimension
    testFrames = np.squeeze(testFrames, axis=1)
    print("frames.shape = ", testFrames.shape)
    
    for i in range(len(predictions)):
        print("predictions[i] = ", predictions[i])
        print("testFrames[i] = ", testFrames[i])
        superImposedImage = superimposeImage(predictions[i], testFrames[i])
        # display_image("superimposed", superImposedImage)
        imageFinal = image_with_lanes(superImposedImage)
        # display_image("final", imageFinal)
        newVideo.write(imageFinal)

    # Release the video
    newVideo.release()
    
    return filename

# Open the video and play it
def displayVideo(filename):
    cap = cv2.VideoCapture(filename)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break    

            
    cap.release()
    cv2.destroyAllWindows()
    return

def superimposeImage(prediction, image):
    superimposedImage = cv2.putText(image, str(prediction), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return superimposedImage

# loadTestingData will load the testing data and return the frames
def loadTestingData(filename):

    # Load the video file
    cap = cv2.VideoCapture(filename)
    # Get the number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 1000
    if (testMode is True):
        num_frames = 1000
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("height = ", height)
    print("width = ", width)

    # Create an array to store the frames using window_nd
    frames = np.empty((num_frames, int(height/2), int(width/2), 3), np.dtype('uint8'))

    # Read the frames
    for i in range(num_frames):
        if i % 100 == 0:
            print("Loading test frame", i)
        frames[i] = shrink_img(cap.read()[1])

    frames = np.expand_dims(frames, axis=1)

    # Close the video file
    cap.release()
    
    # Return the frames
    return frames[0:num_frames]

def main():

    modelExists, model = checkIfModelExists()

    if (modelExists is False):
        print("Model does not exist, creating model...")
        print("Loading training data...")
        frames, train = loadTrainingData()

        print("frames.shape = ", frames.shape)
        print("train.shape = ", train.shape)

        print("Loading model...")
        model = createModel(len(frames))

        model.summary()

        print("Training model...")
        trainModel(model, frames, train)
        print("Training complete...")

    model.summary()
    
    print("Loading Testing data...")
    testFrames = loadTestingData()

    print("Testing complete, creating predictions...")
    predictions = testModel(testFrames)

    filename = publishPredictions(predictions, testFrames)
    displayVideo(filename)
    
    return

# processNewVideo will process a new video file, editing the video in place
def processNewVideo(filename):
    err = "testing 123"
    
    modelExists, model = checkIfModelExists()

    if (modelExists is False):
        return "Model does not exist, please train the model first."

    print("Loading data...")
    testFrames = loadTestingData(filename)

    print("Testing complete, creating predictions...")
    predictions = testModel(testFrames)

    _ = publishPredictions(predictions, testFrames, filename)
    
    return err

if __name__ == "__main__":
    main()
