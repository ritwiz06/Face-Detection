import cv2


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img1 = cv2.imread("photo.jpg")
img2 = cv2.imread("news.jpg") #by default color image
#use gray scale image for better detection in next step to keep the original image intact

img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#search for cascade classifier and return the face coordinates
face1 = face_cascade.detectMultiScale(img_gray1,
scaleFactor=1.05, #starts from original size, it decreses the size of image by 5 %, then again 5% and so on for 1.05 Lower the scale more precise is result
minNeighbors=5)   #tells how many Neighbors to search around that destroyAllWindows

face2 = face_cascade.detectMultiScale(img_gray2,
scaleFactor=1.5, #starts from original size, it decreses the size of image by 5 %, then again 5% and so on for 1.05 Lower the scale more precise is result
minNeighbors=5) #not very effiecient but this is all we have for now, will detect the clear face only on increasing the scale else on decresing the scale it will detect many other no face objects.

# print(type(face))
# print(face)

#draw the rectangle around the face
for x, y, w, h in face1:
    img1=cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3)  #rectangle(image_name, diagonal_coordinates, rectangle color, width)
for x, y, w, h in face2:
    img2=cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 3)


# resized_img1  = cv2.resize(img1, (img1.shape[1]//2, img1.shape[0]//2))
# cv2.imshow("Face_detected", resized_img1)
# cv2.imwrite("Face_detected1.jpg", resized_img1)

resized_img2  = cv2.resize(img2, (img2.shape[1]//2, img2.shape[0]//2))
cv2.imshow("Face_detected2", resized_img2)
cv2.imwrite("Face_detected2.jpg", resized_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
