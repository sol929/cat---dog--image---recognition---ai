将张量流导入为 tf
从tensorflow.keras.preprocessing.image导入ImageDataGenerator
从tensorflow.keras.models导入顺序
从tensorflow.keras.layers导入Conv2D、MaxPooling2D、Flatten、Dense

#数据出售
train_datagen = ImageDataGenerator（重新缩放= 1./255）
test_datagen = ImageDataGenerator（重新缩放= 1./255）

train_generator = train_datagen.flow_from_directory(
    '火车目录',
    目标大小=(150, 150),
    批量大小=32，
    class_mode='二进制'
）

test_generator = test_datagen.flow_from_directory(
    '目录测试',
    目标大小=(150, 150),
    批量大小=32，
    class_mode='二进制'
）

#构建模型
模型=顺序（[
    Conv2D(32, (3, 3), 激活='relu', input_shape=(150, 150, 3)),
    最大池化2D((2, 2)),
    Conv2D(64, (3, 3), 激活='relu'),
    最大池化2D((2, 2)),
    展平（），
    密集（64，激活='relu'），
    密集（1，激活='sigmoid'）
]）

#编译模型
model.compile(优化器='亚当',
              损失='binary_crossentropy',
              指标= [ '准确性' ]）

#训练模型
模型.(
    火车货车，
    steps_per_epoch = train_generator.samples // train_generator.batch_size,
    历元 = 10,
    验证数据=测试生成器，
    validation_steps = test_generator.samples // test_generator.batch_size
）
 
