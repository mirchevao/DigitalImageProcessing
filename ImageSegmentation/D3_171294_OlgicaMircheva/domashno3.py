
import cv2
import glob



path = glob.glob("database/*.jpg")
imagesDatabase = []
for img in path:
    n = cv2.imread(img, 1)
    imagesDatabase.append(n)

path = glob.glob("query/*.jpg")
imagesQuery = []
for img in path:
    n = cv2.imread(img, 1)
    imagesQuery.append(n)


finalImages = []
for image1 in imagesDatabase:
    img1 = image1.copy()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(img1, 127, 255, 0, cv2.THRESH_BINARY)
    contours1, h1 = cv2.findContours(th1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(th1, contours1, -1, (120, 0, 0), 3)
    finalImages.append(th1)

for image2 in imagesQuery:
    img2 = image2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, th2 = cv2.threshold(img2, 127, 255, 0, cv2.THRESH_BINARY)
    contours2, h2 = cv2.findContours(th2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(th2, contours2, -1, (120, 0, 0), 3)
    finalImages.append(th2)


for img in finalImages:
    img = cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#konturite vo globala se vo red, ima propusti samo kaj nekolku sliki
#koga go zipuvav fajlot so site sliki bese pregolem za da se prikaci na courses, zatoa go imam staveno samo python fajlot
    