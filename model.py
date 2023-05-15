from ultralytics import YOLO


# Train YOLO model with a specified dataset
def train_model(data_path, base_model, epochs):
    model = YOLO(base_model)
    model.train(data=data_path, epochs=epochs)
    model.val()
    model.export()
    return model


def get_trained_model(name):
    return YOLO(name)