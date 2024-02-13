# Use keras to train a model which accepts 
# an mp4 video and a train.txt file (which contains the speeds of the video),
# and outputs a model which can predict the speed of a video.

# - data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
# - data/train.txt contains the speed of the car at each frame, one speed on each line.
# - data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.

import numpy as np
import cv2
import keras
import tensorflow

def shrink_img(img):

    # log 
    # print("shrink_img: img.shape = " + str(img.shape))
    # Shrink the image to 320x240 and maintain aspect ratio
    resized_img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA)
    # log
    # print("shrink_img: resized_img.shape = " + str(resized_img.shape))
    return resized_img

# load_data loads the full training data set, which is a video file and a text file
def load_data():
    
    # Load the video file
    cap = cv2.VideoCapture("data/train.mp4")

    # Get the number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create an array to store the frames
    frames = np.empty((num_frames, int(height/2), int(width/2), 3), np.dtype('uint8'))

    # Load the train.txt file
    train = np.loadtxt("data/train.txt")

    # Read the frames
    for i in range(num_frames):
        if i % 100 == 0:
            print("Loading train frame", i)
        frames[i] = shrink_img(cap.read()[1])

    # Close the video file
    cap.release()
    
    # Return the frames and the speeds
    return frames, train

# Load the test data
def load_test_data():
        
    # Load the video file
    cap = cv2.VideoCapture("data/test.mp4")
    # Get the number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get the width and height of the frames
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create an array to store the frames using window_nd
    frames = np.empty((num_frames, int(height/2), int(width/2), 3), np.dtype('uint8'))

    # Read the frames
    for i in range(num_frames):
        if i % 100 == 0:
            print("Loading test frame", i)
        frames[i] = shrink_img(cap.read()[1])

    # Close the video file
    cap.release()
    
    # Return the frames
    return frames

batch_size = 20

# Create the model
# The input is the "frames" construct, which is a 4D array of shape (num_frames, height, width, 3)
# The output is the "train" construct, which is a 1D array of shape (num_frames)
def create_model():

    model = keras.models.Sequential()

    # Input shape of our Time Distributed Layer = 
    # (No. of Images per sample,height of the image, width of the image, No. of channels)
    # Input to this layer is (X, h, w, 3), where X is the number of frames in the video
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Conv2D(8, (3, 3), strides=(2, 2), activation='relu'),
            input_shape=(None, 240, 320, 3)
        )
    )

    # model.add(
    #     keras.layers.TimeDistributed(
    #         keras.layers.Conv2D(4, (3, 3), strides=(2, 2), padding="same", activation='relu')
    #     )
    # )

    # model.add(
    #     keras.layers.TimeDistributed(
    #         keras.layers.MaxPooling2D((2, 2), strides=(2, 2))
    #     )
    # )

    # flatten each frame
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Flatten()
        )
    )

    # LSTM layer
    model.add(
        keras.layers.LSTM(8, return_sequences=False, activation='relu', stateful=True)
    )
    
    # # flatten the sequence
    # model.add(
    #     keras.layers.Flatten()
    # )

    # # fully connected layer
    # model.add(
    #     keras.layers.Dense(4) # , activation='relu')
    # )

    # # dropout layer
    # model.add(
    #     keras.layers.Dropout(0.5)   
    # )

    # output layer
    model.add(
        keras.layers.Dense(batch_size)
    )

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.build()

    model.summary()
    return model

# Train the model
def train_model(model, frames, train):

    timesteps = 0

    # Load batches of frames and speeds from the training data
    # until we have read all the data
    while True:

        # print status
        print("training timesteps: " + str(timesteps))

        # Load the next batch of frames
        frames_batch = frames[timesteps:timesteps+batch_size]
        # Load the next batch of speeds
        train_batch = train[timesteps:timesteps+batch_size]
        # Increment the index
        timesteps += batch_size
        # If we have reached the end of the data, break
        if timesteps >= len(frames): # 1000:
            timesteps = 0
            break

        
        # Train the model on the batch

        # reshape the frames_batch to be (batches, timesteps, 240, 320, 3)
        # frames_batch = np.reshape(frames_batch, (1, batch_size, 240, 320, 3))

        # Add a dimension to the frames_batch in the 2nd position 
        frames_batch = np.expand_dims(frames_batch, axis=0)

        # reshape the train_batch to be (batches, timesteps)
        # train_batch = np.reshape(train_batch, (1, batch_size))

        # transpose the train_batch to be (timesteps, batches)
        # train_batch = np.transpose(train_batch)
        train_batch = np.expand_dims(train_batch, axis=0)

        # log info about the batch
        print("frames_batch.shape = " + str(frames_batch.shape))
        print("train_batch.shape = " + str(train_batch.shape))

        # model.fit wants 
        model.fit(frames_batch, train_batch, epochs=1, batch_size=batch_size, verbose=2)

    # save the model
    model.save("model.h5")

# Test the model
def test_model(model, frames):

    timesteps = 0

    # frames.shape on entering test_model() = (10798, 240, 320, 3)
    
    # Create an empty 1D array (with batch_size elements)
    predictions = np.empty((1,batch_size), np.dtype('float32'))

    # predictions = np.empty((1,20), np.dtype('float32'))

    # Load batches of frames and speeds from the testing data
    # until we have read all the data
    while True:

        # print status
        print("testing timesteps: " + str(timesteps))

        # Load the next batch of frames
        test_batch = frames[timesteps:timesteps+batch_size]
        # Increment the index
        timesteps += batch_size
        # If we have reached the end of the data, break
        if timesteps >= len(frames): # 1000:
            timesteps = 0
            break

        # Add a dimension to the batch in the 2nd position, 
        test_batch = np.expand_dims(test_batch, axis=0)
        # test_batch = np.reshape(test_batch, (1, batch_size, 240, 320, 3))

        # Turn test_batch into a 2D array

        # Predict the speeds for the batch
        outputFromPredictOnBatch = model.predict_on_batch(test_batch)

        # The output from the above predict_on_batch will be an array of 20x8.
        # I want to concatenate all these rows into one long row, 1x160
        # So I need to reshape the outputFromPredictOnBatch to be 1,160
        # outputFromPredictOnBatch = np.reshape(outputFromPredictOnBatch, (1, 20))

        # # Predict the speeds for the batch
        # predictions_batch = model.predict(test_batch)
        # print("outputFromPredictOnBatch.shape before squeeze = " + str(outputFromPredictOnBatch.shape))

        # remove one dimension from outputFromPredictOnBatch
        # outputFromPredictOnBatch = np.squeeze(outputFromPredictOnBatch, axis=0)
        
        # log shapes of the batch and the predictions
        # print("test_batch.shape = " + str(test_batch.shape))
        # print("outputFromPredictOnBatch.shape after squeeze = " + str(outputFromPredictOnBatch.shape))
        # print("predictions.shape = " + str(predictions.shape))

        # Use predict_on_batch to predict the speeds for the batch
        # and append the predictions to the predictions array
        predictions = np.append(predictions, outputFromPredictOnBatch, axis=1)
        # # Append the predictions to the array
        # predictions = np.append(predictions, predictions_batch, axis=0)

    return predictions

def main():

    # Load the training data
    frames, train = load_data()

    # Create the model
    model = create_model()  

    # Train the model
    train_model(model, frames, train)

    # Load the test data
    test_frames = load_test_data()

    # Test the model
    predictions = test_model(model, test_frames)

    # transpose the predictions
    # predictions = np.transpose(predictions)
    
    # Log info about predictions
    print("predictions.shape = ", predictions.shape)
    print(predictions)

    # Save the predictions to a file
    np.savetxt("predictions.txt", predictions)

main()