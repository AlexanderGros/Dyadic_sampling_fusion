
"""

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Save the history dictionary to a file (e.g., JSON format)
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)

"""

# Dyadic part

import matplotlib.pyplot as plt
import json

# Load the saved history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Plot training and validation loss
plt.figure()
plt.xticks(fontsize=20) 
plt.yticks(fontsize=20)
plt.plot(history['loss'][:40], linewidth=5, label='Training Loss_DD')
plt.plot(history['val_loss'][:40], linewidth=5, label='Validation Loss_DD')
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('Training and Validation Loss', fontsize=20)



#simple IQ

with open('IQ_eval/training_history3.json', 'r') as f:
    history = json.load(f)

# Plot training and validation loss
plt.plot(history['loss'][:40], linewidth=5,label='Training Loss_IQ')
plt.plot(history['val_loss'][:40], linewidth=5, label='Validation Loss_IQ')
plt.legend(fontsize=20)
plt.savefig('aa_loss_comparison.png', bbox_inches='tight')

