import re
import matplotlib.pyplot as plt


def main():
    # Initialize lists to store loss values
    epochs = []
    train_loss = []
    val_loss = []

    # Open the log file and extract loss values
    with open('training_log.txt', 'r') as file:
        for line in file:
            # Extract training loss and epoch
            train_match = re.search(r"'loss': ([0-9\.]+), .* 'epoch': ([0-9\.]+)", line)
            if train_match:
                train_loss.append(float(train_match.group(1)))
                epochs.append(float(train_match.group(2)))
            
            # Extract validation loss
            val_match = re.search(r"'eval_loss': ([0-9\.]+), .* 'epoch': ([0-9\.]+)", line)
            if val_match:
                val_loss.append(float(val_match.group(1)))

    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs[:len(train_loss)], train_loss, label='Training Loss', color='blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs[:len(val_loss)], val_loss, label='Validation Loss', color='red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


