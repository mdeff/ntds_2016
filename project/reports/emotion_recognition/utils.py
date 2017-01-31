import sys
#workaround for openCV on osX
sys.path.append('/usr/local/lib/python3.6/site-packages') 

# IMPORTANT: OPENCV 3 for Python 3 is needed, install it from : 
# http://docs.opencv.org/3.0-beta/doc/tutorials/introduction/windows_install/windows_install.html
# or on MAC : brew install opencv3 --with-contrib --with-python3 --HEAD
# http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

import cv2
import matplotlib.pyplot as plt
import PIL 
import numpy as np

from scipy.sparse import coo_matrix

# returns the array of 48x48 images of faces and the whole image with rectangles over the faces img_path = 'camera' or 'file.png' or 'file.jpg' 
def get_faces_from_img(img_path):
    
    # The face recognition properties, recognizing only frontal face
    cascPath = 'haarcascade_frontalface_default.xml'
    
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    #read image and convert to grayscale
    if (img_path == 'camera'):
        video_capture = cv2.VideoCapture(0)
        ret, image = video_capture.read()
    else:
        image = cv2.imread(img_path,1)
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    #print("Found {0} faces in ".format(len(faces)), img_path, " !")

    #preparing an array to store each face img separately 
    faces_imgs = np.zeros((len(faces),48,48))

    # iterate through the faces and save them into a separate array
    num_fac = 0;

    for (x, y, w, h) in faces:

        face_single = image[y:y+h,x:x+w];
        #resize to 48x48
        face_resized = cv2.resize(face_single, (48,48));
        #cv2.imwrite('Face'+str(num_fac)+'.png', face_resized)
        #taking only one color (because it's grey RGB)
        faces_imgs[num_fac] = face_resized[:,:,0]
        num_fac = num_fac+1;
        #adding rectangles to faces

    # adding rectangles on faces in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #cv2.imshow("Faces found", image)
    #cv2.imwrite('Faces_recognized.png', image)
    return faces_imgs, image

def convert_to_one_hot(a,max_val=None):
    N = a.size
    data = np.ones(N,dtype=int)
    sparse_out = coo_matrix((data,(np.arange(N),a.ravel())), shape=(N,max_val))
    return np.array(sparse_out.todense())

#convert single vector to a value
def convert_from_one_hot(a):
     return np.argmax(a);
    
def get_min_max( img ):
    return np.min(np.min(img)),np.max(np.max(img))

def remap(img, min, max):
    if ( max-min ):
        return (img-min) * 1.0/(max-min)
    else:
        return img
    
#constrast stretch function
def contrast_stretch(img ):
    min, max = get_min_max( img );
    return remap(img, min, max)

#calculating partial accuracy (for each clas separately)
def calc_partial_accuracy(tset, result, emlabel):
    
    tsetlabels = np.where(tset == emlabel)[0];
    resultlabels = np.where(result == emlabel)[0];

    errors =0;
    for label in resultlabels :
        if label not in tsetlabels:
            errors += 1;
    
    for label in tsetlabels :
        if label not in resultlabels:
            errors += 1;
    
    return (len(resultlabels)+ len(tsetlabels)- errors)/ (len(resultlabels)+ len(tsetlabels))

# loads the  csv labelled emotion images dataset 
def load_dataset(reader, num_data, hist_div, hist_threshold):
    #preparing arrays
    emotions = np.zeros(num_data)
    images = np.zeros((num_data,48,48))
    strange_im = np.zeros((int(num_data/10),48,48)) # the dataset contains <10% of strange img

    # for image pre-filtering
    num_strange = 0; #number of removed images
    num_skipped = 0; #hapy images skip counter
    rownum =0;
    #parsing each row
    for row in reader:
        #(column0) extract the emotion label
        #!!!! convert 1 and 0 together !!!!
        if( (row[0] == '0') or (row[0] == '1' ) ):
            emotions[rownum] = '0';
        else :
            emotions[rownum] = str(int(row[0])-1)

        #ignore 1/3 of happy cklass pic, there are too many in relative to to others  
        if( (emotions[rownum] != 2 ) or ((emotions[rownum] == 2) and (np.random.choice([0,1,1]) == 1) )): 

            #(column1) extract the image data, parse it and convert into 48x48 array of integers
            images[rownum] = np.asarray([int(s) for s in row[1].split(' ')]).reshape(48,48)

            #stretching contrast of the image
            images[rownum] = contrast_stretch(images[rownum])

            #calculating the histogram and erasing "strange" images
            y_h, x_h = np.histogram( images[ rownum ] , 100 );
            if y_h.max() > hist_threshold  : 
                # if img is 'strange'
                strange_im[num_strange,:,:] = images[rownum,:,:];
                num_data = num_data - 1;
                images = np.delete(images, rownum, axis = 0);
                emotions = np.delete(emotions, rownum)
                #print('deleted:',rownum, y_h.max())
                num_strange += 1;   
            else:
                rownum += 1
            if not rownum%500:
                print("loaded %2.0f" % ((float(rownum ) /num_data)*100) 
                      + '% of dataset ('+ str(rownum+num_strange)+'/'+ str(num_data) + '). Filtered images: ' + str(num_strange) )
        else:
            images = np.delete(images, rownum, axis = 0);
            emotions = np.delete(emotions, rownum)
            num_skipped +=1; # skip some happy images 
    
    return images, emotions, strange_im, num_strange, num_skipped
