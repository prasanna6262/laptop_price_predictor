import joblib
def save_model(model,encoder,model_path,encoder_path):
    joblib.dump(model,model_path)
    joblib.dump(encoder,encoder_path)
def load_model(model_path,encoder_path):
    model=joblib.load(model_path)
    encoder=joblib.load(encoder_path)
    return model,encoder