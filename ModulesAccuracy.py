import os
import dlib

# options用于设置训练的参数和模式
options = dlib.simple_object_detector_training_options()

options.add_left_right_image_flips = True

options.C = 5
options.num_threads = 6
options.be_verbose = True

current_path = os.getcwd()
train_folder = current_path + "/images/"
test_folder = current_path + "/actor/"
train_xml_path = "images.xml"
test_xml_path = "datatest.xml"

print("training file path" + train_xml_path)
print("testing file path" + test_xml_path)



#使用训练集测试
print("Training accuracy:{}".format(
    dlib.test_simple_object_detector(train_xml_path,"detector.svm")
))
#使用测试集测试
print("Testing accuracy:{}".format(
    dlib.test_simple_object_detector(test_xml_path,"detector.svm")
))
