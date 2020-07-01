import cv2 as cv
import numpy as np
import dlib
import os

#定义资源路径
file_path = os.path.abspath(".")
mov_name = "twopeople"
mov_path = "./" + mov_name +".MP4"         #定义视频文件路径
detector_path = "./detector.svm"

#定义资源变量
detector = dlib.simple_object_detector(detector_path)
video = cv.VideoCapture(mov_path)
mov = cv.VideoCapture(mov_path)
predictor_path = "./shape_predictor_68_face_landmarks.dat"          #dlib官方提供的人脸 68 点特征检测器
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1 (1).dat"         #dlib官方提供的人脸模型（残差网络原理）

#dlib官方提供的正向人脸检测器
#detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

def getFrameCount():
    frame_count = 0
    all_frames = []
    while(True):
        ret,frame = mov.read()
        if ret is False:
            break
        all_frames.append(frame)
        frame_count = frame_count + 1
    return frame_count


#图片清晰度对比函数，传入灰度图
def getImgVar(img):
    imageVar = cv.Laplacian(img, cv.CV_64F).var()       #cv2.CV_64F就是拉普拉斯算子
    return  imageVar        #返回一个清晰度

#人脸对比函数，返回一个对比值
def comparePersonFace(dist1,dist2):
    diff = np.sqrt(sum((np.array(dist1) - np.array(dist2))**2))
    return diff

def main():
    frame_count = getFrameCount()
    timeF = 5       # 视频帧计数间隔频率
    m = frame_count // timeF     #定义最多图片保存数
    n = 5       #每帧最多识别人脸数量为6
    dists = [[] for i in range(m)]      #定义空的二维数组[m,n]
    images = []     #最终图片
    i = 0
    face_count = 0
    flag = True
    ret,frame = video.read()
    while ret:      #循环读取视频帧
        if i % timeF == 0:      #每隔timeF帧进行操作
            gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)     #转化为灰度图像
            faces = detector(frame,1)  #检测人脸
            if len(faces):
                for index,face in enumerate(faces):
                    shape = shape_predictor(gray,face)       # 提取68位特征点
                    cv.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),(0,255,0),2)     #标记人脸
                    face_descriptor = face_rec_model.compute_face_descriptor(frame,shape)        #计算人脸128维向量
                    if flag :
                        face_count = 1
                        dists[len(images)].append(face_descriptor)        #dist数组保存向量值
                        images.append(frame)          #images数组保存图片
                        flag = False        #第一帧先保存
                        #       print(dists)
                        #       print(images)
                    else:
                        tag = True      #同一人标记位
                        #        print(face_descriptor)
                        #print(dists)
                        for k in range(len(dists)):     #遍历整个二维数组，查找是否有同一人并更清晰的图片
                            for j in range(len(dists[0])):
                                if dists[k]:
                                    if comparePersonFace(dists[k][j],face_descriptor) < 0.5:
                                        listgray = cv.cvtColor(images[k],cv.COLOR_BGR2GRAY)
                                        if getImgVar(listgray)<getImgVar(gray):
                                            images[k] = frame
                                        tag = False
                                        break
                        if tag:
                            dists[len(images)].append(face_descriptor)
                            images.append(frame)
                            face_count += 1
        #print(len(images))
        #print(dists)
        ret,frame = video.read()
        i += 1
    print("识别的人脸数量为" + str(face_count))       #人脸数量
    return images
x = []
x = main()
q = 1
for fac in x:
    cv.imwrite(file_path + "/faces/" + mov_name + str(q) + ".jpg",fac)
    q +=1
print("保存路径为" + file_path + "/face/" )
print("以"+ mov_name + "开头的图片")
print("保存的人脸图片数量为" + str(len(x)))       #图片数量










