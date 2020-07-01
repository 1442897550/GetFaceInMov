# GetFaceInMov
## 项目描述
    通过python+dlib+opencv+numpy实现读取视频并从视频中识别人脸，提取具有人脸清晰度最高的帧（画面），并输出到本地。
## 文件说明
    GetFaceInMov文件为项目实现
    images、images.xml文件为训练集
    actor、datatest.xml文件为测试集
    detector.svm为训练的人脸识别模型
    TrainingModule.py为模型训练脚本
    ModuleAccuracy.py为模型准确度测试脚本
    ModuleTest.py为模型测试脚本（分别有TestOneFace.jpg TestTwoFace.jpg 进行测试）
    FaceInMov.py为项目脚本（分别有onepeople.mp4 twopeople.mp4进行测试）
