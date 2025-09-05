import os
import sys

def clear_console():4
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    clear_console()
    print("=== Air Digit Project Menu ===")
    print("1. Collect Air Digit Data (Air MNIST)")
    print("2. Train Air Digit CNN")
    print("3. Test Air Digit CNN Accuracy")
    print("4. Run Air Digit Recognizer")
    print("5. Train Classic MNIST CNN")
    print("6. Test Classic MNIST Model")
    print("0. Exit")
    return input("Select an option: ")

while True:
    choice = menu()

    if choice == '1':
        # Launch Air MNIST data collector
        os.system("python air_digit_trainer.py")
    elif choice == '2':
        # Train CNN on Air MNIST dataset
        os.system("python train_air_digit_cnn.py")
    elif choice == '3':
        # Test Air MNIST CNN accuracy
        os.system("python test_model_accuracy.py")
    elif choice == '4':
        # Run live recognizer with trained Air MNIST CNN
        os.system("python air_digit_recognizer.py")
    elif choice == '5':
        # Train CNN on classic MNIST (folder 'data' or default MNIST)
        os.system("python train_mnist_model.py")
    elif choice == '6':
        # Test classic MNIST model (assumes mnist_cnn_updated.keras exists)
        os.system("python test_mnist_model.py")
    elif choice == '0':
        print("Exiting...")
        sys.exit()
    else:
        print("Invalid option. Press Enter to continue.")
        input()
