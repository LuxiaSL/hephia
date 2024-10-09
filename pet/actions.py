# pet/actions.py

class PetActions:
    """
    Defines caretaking actions for the pet.
    """

    @staticmethod
    def feed(pet, food_value=1, food_type=None):
        """
        Feeds the pet.

        Args:
            pet (Pet): The pet instance.
            food_value (int, optional): The strength of the food being given.
            food_type (str, optional): The type of food given.
        """
        pet.feed(food_value=food_value, type=food_type)

    @staticmethod
    def give_water(pet, thirst_value=1, water_type=None):
        """
        Gives water to the pet.

        Args:
            pet (Pet): The pet instance.
            thirst_value (int, optional): The strength of the drink being given.
            water_type (str, optional): The type of drink given.
        """
        pet.drink(thirst_value=thirst_value, type=water_type)

    @staticmethod
    def play_with_pet(pet, play_value=1, play_type=None):
        """
        Plays with the pet.

        Args:
            pet (Pet): The pet instance.
            play_value (int, optional): The strength of the play being performed.
            play_type (str, optional): The type of play activity.
        """
        pet.play(play_value=play_value, type=play_type)

    @staticmethod
    def rest_pet(pet):
        """
        Allows the pet to rest.

        Args:
            pet (Pet): The pet instance.
        """
        pet.rest()
