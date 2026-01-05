
"""

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Save the history dictionary to a file (e.g., JSON format)
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

"""

import matplotlib.pyplot as plt
import json

# Load the saved history
with open('training_history3.json', 'r') as f:
    history = json.load(f)

# Plot training and validation loss
plt.figure()
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig('fct_loss.png', bbox_inches='tight')
#plt.show()


'''
# Similarly, plot accuracy if available
if 'categorical_accuracy' in history and 'val_categorical_accuracy' in history:
    print('exists')
    plt.figure()
    plt.plot(history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('fct_accuracy.png', bbox_inches='tight')
    #plt.show()
else:
    print('not exists')
    

'''

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



'''
np.save('my_history.npy',history.history)
'''

# history2=np.load('my_history.npy',allow_pickle='TRUE').item()























