#Import Dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from face_utils import crop_face

#Import uuid library to generate unique image names
import uuid

#Import tensorflow dependencies - Functional API
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# #Setup paths
# POS_PATH = os.path.join('data','positive')
# NEG_PATH = os.path.join('data','negative')
# ANC_PATH = os.path.join('data','anchor')
# NEGCAM_PATH = os.path.join('data','negative_webcam')

#Setup paths
POS_PATH = os.path.join('data_face','positive')
NEG_PATH = os.path.join('data_face','negative')
ANC_PATH = os.path.join('data_face','anchor')
NEGCAM_PATH = os.path.join('data_face','negative_webcam')

#Make the directories
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)
os.makedirs(NEGCAM_PATH, exist_ok=True)


#Moving the images from the archive images folder to the Negative Images subfolder in the data folder
for root, dirs, files in os.walk('archive'):
    for file in files:
        EX_PATH = os.path.join(root, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
print("Click the webcam window. a = anchor, p = positive, n = webcam negative, q = quit")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    face = crop_face(frame)
    if face is None:
        cv2.imshow('Image Collection', frame)  # show full frame while no face is found
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue
    frame = face
    cv2.imshow('Image Collection', frame)
    # frame = frame[120:120+250, 200:200+250, :]
    # cv2.imshow('Image Collection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), frame)
        print("Anchors:", len(os.listdir(ANC_PATH)))
    elif key == ord('p'):
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), frame)
        print("Positives:", len(os.listdir(POS_PATH)))
    elif key == ord('n'):
        cv2.imwrite(os.path.join(NEGCAM_PATH, '{}.jpg'.format(uuid.uuid1())), frame)
        print("Webcam negatives:", len(os.listdir(NEGCAM_PATH)))
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)
# while cap.isOpened():   #loops through all the frames shown in the camera
#     ret, frame = cap.read()   #returns the frames captured from the Video Camera

#     #Cut down frame to 250x250px
#     frame = frame[120:120+250, 200:200+250, :]

#     #Collect anchors
#     if cv2.waitKey(1) & 0XFF == ord('a'):
#         #Create the unique file path
#         imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
#         #Write out anchor image
#         cv2.imwrite(imgname, frame)

#     #Collect positives
#     if cv2.waitKey(1) & 0XFF == ord('p'):
#          #Create the unique file path
#         imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
#         #Write out positive image
#         cv2.imwrite(imgname, frame)

#     #Shows image back to screen
#     cv2.imshow('Image Collection', frame)

#     #Breaking gracefully
#     if cv2.waitKey(1) & 0XFF == ord('q'):
#         break

# #Release the webcam
# cap.release
# #Close the image show frame
# cv2.destroyAllWindows()

def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta = 0.2, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower = 0.6, upper = 1, seed=(1,3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper = 1, seed=(np.random.randint(100), np.random.randint(100)))

        data.append(img)
    return data

img_path = os.path.join(ANC_PATH, '6b391b4b-ce56-11f1-9891-202b2082ddca.jpg')
img = cv2.imread(img_path)
augmented_images = data_aug(img)

for image in augmented_images:
    cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

for file_name in os.listdir(os.path.join(POS_PATH)):
    img_path = os.path.join(POS_PATH, file_name)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img)

    for image in augmented_images:
        cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())




#Get the image directories
anchor = tf.data.Dataset.list_files(os.path.join(ANC_PATH, '*.jpg')).take(3000)  #loops through every image in the anchor folder and creates a list or set of them 
positive = tf.data.Dataset.list_files(os.path.join(POS_PATH,'*.jpg')).take(3000)
# negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)
negative_lfw = tf.data.Dataset.list_files(os.path.join(NEG_PATH,'*.jpg')).take(1500)
negative_cam = tf.data.Dataset.list_files(os.path.join(NEGCAM_PATH,'*.jpg')).take(1500)
negative = negative_lfw.concatenate(negative_cam)

dir_test = anchor.as_numpy_iterator()

print(dir_test.next())

#Preprocessing the image - Scaling and Resizing the image
def preprocess(file_path):
    #Read in image from file path
    byte_img = tf.io.read_file(file_path)
    #Load in the image
    img = tf.io.decode_jpeg(byte_img)

    #Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    #Scale the image to be between 0 and 1
    img = img / 255.0
    #Returns the image
    return img

# img = preprocess('data\\anchor\\')
# plt.imshow(img)

#Create Labelled Dataset
#if the anchor and positive image will be same, they will be labelled as 1 means verified
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
#if the anchor and negative image will be different, they will be labelled as 0 means unverified
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
#merge all the images of positively and negatively labelled into one new folder 'data'
data = positives.concatenate(negatives)

samples = data.as_numpy_iterator()

print(samples.next())

example = samples.next()

#Build Train and Test Partition
def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

res = preprocess_twin(*example)
# print("This is the second file path",res[1])

#Build the dataloader pipeline
data = data.map(preprocess_twin)
#Stores all the preprocessed images firsthand rather than preprocessing everytime
data = data.cache()
#Shuffling the positive and negative images so while splitting the training and testing, each has equal number of positive and negative images
# data = data.shuffle(buffer_size=1024)
data = data.shuffle(buffer_size=10000, reshuffle_each_iteration=False)

#Training partition
#Takes 70% of the data for training
train_data = data.take(round(len(data)*.7))
#Takes the data in the batch of 16 images
train_data = train_data.batch(16)
#Keeps 8 images preprocessed ready to take in the next batch
train_data = train_data.prefetch(8)

#Testing Partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
#Takes the data in the batch of 16 images
test_data = test_data.batch(16)
#Keeps 8 images preprocessed ready to take in the next batch
test_data = test_data.prefetch(8)

#Model Engineering
#Building an Embedding Layer
def make_embedding():
    inp = Input(shape=(100,100,3), name = 'input_image')      #Input Layer
    
    #First Block
    c1 = Conv2D(64, (10,10), activation='relu')(inp) #Convolution Layer
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1) #MaxPooling Layer

    #Second Block
    c2 = Conv2D(128, (7,7), activation='relu')(m1) #Convolution Layer
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2) #MaxPooling Layer(

    #Third Block
    c3 = Conv2D(128, (4,4), activation='relu')(m2) #Convolution Layer
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3) #MaxPooling Layer

    #Final Embedding Block
    c4 = Conv2D(256, (4,4), activation = 'relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
 
    return Model(inputs= inp, outputs = d1, name='embedding')

embedding = make_embedding()
print(embedding.summary())

#Build a Distance Layer
#Siamese L1 Distance Class
class L1Dist(Layer):  
    #Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #Magic happens here - similarity calculation
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)
    
#Making Siamese Model
def make_siamese_model():

    #Anchor Image input in the network
    input_image = Input(name='input_img', shape = (100,100,3))

    #Validation Image in the network
    validation_image = Input(name='validation_img', shape = (100,100,3))

    #Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer([embedding(input_image), embedding(validation_image)])

    #Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs = [input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()
print(siamese_model.summary())

#Setting up the optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)   #here "1e-4 = 0.0001" is the learning rate of the model

#Establishing the checkpoints - We are going to use it when are setting the training loop
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model= siamese_model)

#Building the training function
@tf.function
def train_step(batch):

    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image as embeddings or features
        X = batch[:2]
        #Get the labels or similarity value between anchor and positive/negative images
        y = batch[2]

        #Forward pass
        yhat = siamese_model(X, training=True)
        #Calculate the loss
        loss = binary_cross_loss(y, yhat)
    print(loss)

    # Calculate gradients for the variables in the neural network
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables)) 

    #Return loss
    return loss

from tensorflow.keras.metrics import Precision, Recall

#Building Training Loop
def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        #Creating a metric object
        r = Recall()
        p = Precision()

        #Loop through each batch
        for idx, batch in enumerate(data):
            #Run train step here
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        #Save Checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)


#Training the model
EPOCHS = 10
train(train_data, EPOCHS)

#Import metric calculations
from tensorflow.keras.metrics import Precision, Recall
#Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

#Make predictions
y_hat = siamese_model.predict([test_input, test_val])
print(y_hat)

#Post processing the results into binary 0 or 1
print([1 if prediction > 0.5 else 0 for prediction in y_hat ])

#Creating a metric object
m = Recall()

#Calculating the recall value
m.update_state(y_true, y_hat)

#Return the Recall Result
print(m.result().numpy())

#Creating a metric object
n = Precision()

#Calculating the recall value
n.update_state(y_true, y_hat)

#Return the Recall Result
print(n.result().numpy())

r = Recall()
p = Precision()

for test_input, test_val, y_true in test_data.as_numpy_iterator():
    y_hat = siamese_model.predict([test_input, test_val], verbose=0)
    r.update_state(y_true, y_hat)
    p.update_state(y_true, y_hat)

print(r.result().numpy(), p.result().numpy())

# #Visualization Results
#Set the plot size
plt.figure(figsize= (18,8))

#Set first subplotplt
plt.subplot(1,2,1)   #the image to be showed in which row and column
plt.imshow(test_input[0])

#Set the second subplot
plt.subplot(1,2,2)
plt.imshow(test_val[0])

#Renders Clearly
plt.show()

#Save weights
# siamese_model.save('siamesemodelv2.h5')
siamese_model.save('siamesemodelv3.h5')

#Reload model
model = tf.keras.models.load_model('siamesemodelv3.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

#Make predictions with reloaded model
model.predict([test_input, test_val])

print(model.summary())

#Verification Function
def verify(model, detection_threshold, verification_threshold):
    ver_dir = os.path.join('application_data', 'verification_images')
    image_names = os.listdir(ver_dir)

    input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
    validation_imgs = [preprocess(os.path.join(ver_dir, name)) for name in image_names]

    input_batch = np.stack([input_img] * len(validation_imgs))
    validation_batch = np.stack(validation_imgs)

    results = model.predict([input_batch, validation_batch], batch_size=16, verbose=0)

    detection = np.sum(results > detection_threshold)
    verification = detection / len(image_names)
    verified = verification > verification_threshold

    print(f"Matches: {detection} out of {len(image_names)}")
    print("Scores:", np.round(results.flatten(), 2))
    return results, verified


print("Verification images:", len(os.listdir(os.path.join('application_data', 'verification_images'))))

cap = cv2.VideoCapture(0)
print("Camera opened:", cap.isOpened())
print("Click the webcam window, then press 'v' to verify or 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Could not read from camera")
        break
    frame = frame[120:120+250, 200:200+250, :]
    cv2.imshow('Verification', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('v'):
        print("Verifying...")
        cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
        results, verified = verify(model, 0.5, 0.2)
        print("Verified:", verified)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





# def verify(model, detection_threshold, verification_threshold):
#     #Build results array
#     results = []
#     for image in os.listdir(os.path.join('application_data', 'verification_images')):
#         input_img = preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
#         validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

#         #Make Predictions
#         result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)), verbose=0)
#         results.append(result)

#     #Detection Threshold: Metric above which a prediction is considered positive
#     detection = np.sum(np.array(results) > detection_threshold)

#     #Verification Threshold: Proportion of positive predictions / total positive samples
#     verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
#     verified = verification > verification_threshold

#     return results, verified

# print("Verification images:", len(os.listdir(os.path.join('application_data', 'verification_images'))))

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     frame = frame[120:120+250, 200:200+250, :]

#     cv2.imshow('Verification', frame)

#     #Verification Trigger
#     if cv2.waitKey(10) & 0xFF == ord('v'):
#         #Save input image to application_data/input_image folder
#         cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), frame)
#         #Run verification
#         results, verified = verify(model, 0.9, 0.7)
#         print(verified)

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
