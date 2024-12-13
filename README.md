# Soil-Organic-Matter-Prediction-FFNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

# 构建 FFNN 模型
ffnn_model = Sequential()
ffnn_model.add(Dense(256, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer='l2'))
ffnn_model.add(BatchNormalization())
ffnn_model.add(Dropout(0.3))
ffnn_model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
ffnn_model.add(Dropout(0.3))
ffnn_model.add(Dense(64, activation='relu'))
ffnn_model.add(Dense(1))  # 输出层

# 编译模型
ffnn_model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error', metrics=['mae'])

# 训练模型
ffnn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 保存模型
ffnn_model.save(r'E:\LUCAS2015_topsoildata_20200323\models\ffnn_model.h5')
