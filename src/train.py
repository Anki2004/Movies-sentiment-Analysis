import pickle
from src.data_preprocessing import prepare_data
from src.model import create_model
def train_model(file_path, max_words = 10000, max_len = 100, epochs = 10, batch_size = 32):
    X_train, X_test, y_train, y_test, tokenizer = prepare_data(file_path, max_words, max_len)

    model = create_model(max_words, max_len)

    history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = 0.2,
                        verbose = 1)
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose = 0)
    print(f"Test accuracy: {accuracy: .4f}")

    model.save('models/model.h5')
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model, tokenizer
