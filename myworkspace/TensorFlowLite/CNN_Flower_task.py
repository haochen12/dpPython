from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model

IMG_W = 224  # 定义裁剪的图片宽度
IMG_H = 224  # 定义裁剪的图片高度
CLASS = 14  # 图片的分类数
EPOCHS = 5  # 迭代周期
BATCH_SIZE = 64  # 批次大小
TRAIN_PATH = './data/data/train'  # 训练集存放路径
TEST_PATH = 'E:/pythonProject/data/data/test'  # 测试集存放路径
SAVE_PATH = './data/flower_selector'  # 模型保存路径
LEARNING_RATE = 1e-4  # 学习率
DROPOUT_RATE = 0  # 抗拟合，不工作的神经网络百分比

train_datagen = ImageDataGenerator(
    rotation_range=40,  # 随机旋转度数
    width_shift_range=0.2,  # 随机水平平移
    height_shift_range=0.2,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=20,  # 随机错切变换
    zoom_range=0.2,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest',  # 填充方式
)
test_datagen = ImageDataGenerator(
    rescale=1 / 255,  # 数据归一化
)

model = Sequential()  # 创建一个神经网络对象

# 添加一个卷积层，传入固定宽高三通道的图片，以32种不同的卷积核构建32张特征图，
# 卷积核大小为3*3，构建特征图比例和原图相同，激活函数为relu函数。
model.add(Conv2D(input_shape=(IMG_W, IMG_H, 3), filters=32, kernel_size=3, padding='same', activation='relu'))
# 再次构建一个卷积层
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
# 构建一个池化层，提取特征，池化层的池化窗口为2*2，步长为2。
model.add(MaxPool2D(pool_size=2, strides=2))
# 继续构建卷积层和池化层，区别是卷积核数量为64。
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))
# 继续构建卷积层和池化层，区别是卷积核数量为128。
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Flatten())  # 数据扁平化
model.add(Dense(128, activation='relu'))  # 构建一个具有128个神经元的全连接层
model.add(Dense(64, activation='relu'))  # 构建一个具有64个神经元的全连接层
model.add(Dropout(DROPOUT_RATE))  # 加入dropout，防止过拟合。
model.add(Dense(CLASS, activation='softmax'))  # 输出层，一共14个神经元，对应14个分类

adam = Adam(lr=LEARNING_RATE)  # 创建Adam优化器
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # 使用交叉熵代价函数，adam优化器优化模型，并提取准确率

train_generator = train_datagen.flow_from_directory(  # 设置训练集迭代器
    TRAIN_PATH,  # 训练集存放路径
    target_size=(IMG_W, IMG_H),  # 训练集图片尺寸
    batch_size=BATCH_SIZE  # 训练集批次
)

test_generator = test_datagen.flow_from_directory(  # 设置测试集迭代器
    TEST_PATH,  # 测试集存放路径
    target_size=(IMG_W, IMG_H),  # 测试集图片尺寸
    batch_size=BATCH_SIZE,  # 测试集批次
)

try:
    model = load_model('{}.h5'.format(SAVE_PATH))  # 尝试读取训练好的模型，再次训练
    print('model upload,start training!')
except:
    print('not find model,start training')  # 如果没有训练过的模型，则从头开始训练

model.fit_generator(  # 模型拟合
    train_generator,  # 训练集迭代器
    steps_per_epoch=len(train_generator),  # 每个周期需要迭代多少步（图片总量/批次大小=11200/64=175）
    epochs=EPOCHS,  # 迭代周期
    validation_data=test_generator,  # 测试集迭代器
    validation_steps=len(test_generator)  # 测试集迭代多少步
)

model.save('{}.h5'.format(SAVE_PATH))  # 保存模型
print('finish {} epochs!'.format(EPOCHS))
