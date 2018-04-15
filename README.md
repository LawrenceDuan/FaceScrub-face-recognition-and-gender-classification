# FaceScrub-face-recognition-and-gender-classification
Building a system for face recognition and gender classification. The dataset comes from FaceScrub.

For this project, I will build a system for face recognition and gender classification, and test it on a large(-ish) dataset of faces, getting practice with data-science-flavour projects along the way. 

I used numpy and matplotlib, etc.

copyright@http://www.teach.cs.toronto.edu/~csc411h/winter/projects/proj1/
## The input

You will work with a subset of the FaceScrub dataset. The subset of male actors is here and the subset of female actors is here. The dataset consists of URLs of images with faces, as well as the bounding boxes of the faces. The format of the bounding box is as follows (from the FaceScrub readme.txt file):

    The format is x1,y1,x2,y2, where (x1,y1) is the coordinate of the top-left 
    corner of the bounding box and (x2,y2) is that of the bottom-right corner, 
    with (0,0) as the top-left corner of the image. Assuming the image is 
    represented as a Python NumPy array I, a face in I can be obtained as 
    I[y1:y2,x1:x2].

You may find it helpful to use a modified version of our script for downloading the image data. (Note that the script is not meant to be used as is. You need to figure out how to modify it.)

At first, you should work with the faces of the following actors:

    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
For this project, you should crop out the images of the faces, convert them to grayscale, and resize them to 32x32 before proceeding further. You can use scipy.misc.imresize to scale images, and you can use rgb2gray to convert RGB images to grayscale images.

### Part 1

Describe the dataset of faces. In particular, provide at least three examples of the images in the dataset, as well as at least three examples of cropped out faces. Comment on the quality of the annotation of the dataset: are the bounding boxes accurate? Can the cropped-out faces be aligned with each other?

### Part 2

Separate the dataset into three non-overlapping parts: the training set (100 UPDATE: 70 face images per actor), the validation set (10 face images per actor), and the test set (10 face images per actor). For the report, describe the algorithm that you used to do that (any algorithm is fine, but you should use code to split up your dataset). The training set will contain faces whose labels you assume you know. The test set and the validation set will contain faces whose labels you pretend to not know and will attempt to determine using the data in the training set.

### Part 3

Use Linear Regression in order to build a classifier to distinguish pictures of Alec Baldwin form pictures of Steve Carell. In your report, specify which cost function you minimized. Report the values of the cost function on the training and the validation sets. Report the performance of the classifier (i.e., the percentage of images that were correctly classified) on the training and the validation sets.

You should use Gradient Descent in order to find the parameters θ.

In your report, include the code of the function that you used to compute the output of the classifier (i.e., either Steve Carell or Alec Baldwin).

In your report, describe what you had to do in order to get the system to work. For example, the system would not work if the parameter α is too large. Describe what happens if α is too large, and how you figure out what to set α to. Describe the other choices that you made in order to make the algorithm work.

Tip: divide the input images by 255.0 so that all inputs are in the 0...1 range.

### Part 4 (a)

In Part 3, you used the hypothesis function hθ(x)=θ0+θ1x1+...+θnxn. If (x1,...,xn) represents a flattened image, then (θ1,...,θn) can also be viewed as an image. Display the θs that you obtain by training using the full training dataset, and by training using a training set that contains only two images of each actor.

The images could look as follows.

![Image 1](https://github.com/LawrenceDuan/FaceScrub-face-recognition-and-gender-classification/blob/master/readme1.png)
![Image 2](https://github.com/LawrenceDuan/FaceScrub-face-recognition-and-gender-classification/blob/master/readme2.png)
 

### Part 4 (b)

In Part 4(a), you need to display whatever image Gradient Descent produced. In this part, you should experiment in order to produce both kinds of visualizations. Report on how to obtain both a visualization that contains a face and a visualization that does not, using the full training set. Be specific about what you did. Hint: try stopping the Gradient Descent process earlier and later in the process. Try initializing the θs using different strategies.

### Part 5

In this part, you will demonstrate overfitting. Build classifiers that classify the actors as male or female using the training set with the actors from

    act =['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
and using training sets of various sizes. Plot the performance of the classifiers on the training and validation sets vs the size of the training set.

Report the performance of the classifier on 6 actors who are not included in act.

### Part 6

Now, consider a different way of classifying inputs. Instead of assigning the output value y=1 to images of Paul McCartney and the output value y=−1 to images of John Lennon, which would not generalize to more than 2 labels, we could assign output values as follows:

    Paul McCartney:     [1, 0, 0, 0]
    John Lennon:        [0, 1, 0, 0]
    George Harrison:    [0, 0, 1, 0]
    Ringo Starr:        [0, 0, 0, 1]
    
The output could still be computed using θTx, but θ would now have to be a n×k matrix, where k is the number of possible labels, with x being a n×1 vector.

The cost function would still be the sum of squared differences between the expected outputs and the actual outputs: J(θ)=∑i (∑j (θTx(i)−y(i))2j).

### Part 6(a)

Compute ∂J/∂θpq. Show your work. Images of neatly hand-written derivations are acceptable, though you are encouraged to use LaTeX.

### Part 6(b)

Show, by referring to Part 6(a), that the derivative of J(θ) with respect to all the components of θ can be written in matrix form as 2X(θTX−Y)T.
Specify the dimensions of each matrix that you are using, and define each variable (e.g., we defined m as the number of training examples.) X is a matrix that contains all the input training data (and additional 1’s), of the appropriate dimensions.

### Part 6(c)

Implement the cost function from Part 6(a) and its vectorized gradient function in Python. Include the code in your report.

### Part 6(d)

Demonstrate that the vectorized gradient function works by computing several components of the gradient using finite-difference approximations. In your report, include the code that you used to compute the gradient components using finite differences, and to compare them to the gradient that you computed using your function. In one or two sentences, explain how you compared the approximated values to the output of the gradient function.

Recall that to compute a derivative of a 1-d function using a finite-difference approximation, you can use f′(x)≈(f(x+h)−f(x))/h for a small h.

This can be generalized to functions of several variables.

In your report, explain how you selected an h that makes sense, It is enough to compute the finite-difference approximation along 5 coordinates (as long as the partial derivatives along those coordinates aren’t all 0).

The usual practice is to run the optimization for a while and then perform gradient checking. However, you do not have to do that for this project.

### Part 7

Run gradient descent on the set of six actors act in order to perform face recognition. Report the performance (i.e., proportion of correctly-classified faces) you obtained on the training and validation sets. Indicate what parameters you chose for gradient descent and why they seem to make sense. Describe how you obtained the label from the output of the model.

### Part 8

Visualize the θs that you obtained. Note that if θ is a k×n matrix, where k is the number of possible labels and n−1 is the number of pixels in each image, the rows of θ could be visualized as images. Your outputs could look something like the ones below. Label the images with the appropriate actor names.