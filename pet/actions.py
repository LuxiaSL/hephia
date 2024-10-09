# pet/actions.py

class PetActions:
    """
    Defines caretaking actions for the pet.
    """

    @staticmethod
    def feed(pet):
        """
        Feeds the pet.

        Args:
            pet (Pet): The pet instance.
        """
        pet.feed()

    @staticmethod
    def give_water(pet):
        """
        Gives water to the pet.

        Args:
            pet (Pet): The pet instance.
        """
        pet.give_water()

    @staticmethod
    def play_with_pet(pet):
        """
        Plays with the pet.

        Args:
            pet (Pet): The pet instance.
        """
        pet.play()

    @staticmethod
    def rest_pet(pet):
        """
        Allows the pet to rest.

        Args:
            pet (Pet): The pet instance.
        """
        pet.rest()
