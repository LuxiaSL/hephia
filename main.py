# main.py

import sys
from PyQt5.QtWidgets import QApplication
from visualization.pet_widget import PetWidget
from pet.pet import Pet

def main():
    """Entry point for the Hephia application."""
    app = QApplication(sys.argv)
    
    # Initialize the Pet instance
    pet = Pet()
    
    # Initialize the PetWidget with the Pet instance
    pet_widget = PetWidget(pet)
    pet_widget.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()