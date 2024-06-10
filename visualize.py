import matplotlib.pyplot as plt 


def visualize(dataset, n = 10):
  
    sample_iterator = dataset.take(n).as_numpy_iterator()
    for i, sample in enumerate(sample_iterator):
        (image1,image2), label = sample
        print(f"Sample {i+1}: Image shape: {image1.shape},{image2.shape} Label: {label}")

    for i, (images, labels) in enumerate(dataset.take(n)):
        # Unpack the images and labels
        image1, image2 = images  # Assuming each element in the dataset is a tuple of two images and a label

        # Create a new figure for each pair of images
        plt.figure(figsize=(6, 3))

        # Plot the first image
        plt.subplot(1, 2, 1)
        plt.title(f"Image 1 - Label: {labels.numpy()}")
        plt.imshow(image1.numpy().astype("uint8"))
        plt.axis("off")

        # Plot the second image
        plt.subplot(1, 2, 2)
        plt.title(f"Image 2 - Label: {labels.numpy()}")
        plt.imshow(image2.numpy().astype("uint8"))
        plt.axis("off")

        # Show the figure
        plt.show()


def plot_history(history):

    """
    Plots the training and validation loss over epochs.

    Args:
        history: A history object obtained from the model.fit() method.

    Returns:
        None
    """

    # Get the loss and validation loss from the history object
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Create a figure with one subplot
    plt.figure(figsize=(10, 5))

    # Plot the loss and validation loss on the same subplot
    plt.plot(loss, label="Loss")
    plt.plot(val_loss, label="Validation Loss")

    # Add labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    # Add legend and show the plot
    plt.legend()
    plt.show()




if __name__ == '__main__' : 
    pass
