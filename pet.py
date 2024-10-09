import random

class Pet:
    def __init__(self):
        self.hunger = 50
        self.thirst = 50
        self.boredom = 50
        self.stamina = 100
        self.emotional_state = '😊'  # Default emotional state

    def update_needs(self):
        # Random decay to introduce stochastic behavior
        self.hunger += random.uniform(0.1, 0.5)
        self.thirst += random.uniform(0.1, 0.5)
        self.boredom += random.uniform(0.1, 0.5)
        self.stamina -= random.uniform(0.1, 0.5)

        # Keep values within 0-100%
        self.hunger = min(max(self.hunger, 0), 100)
        self.thirst = min(max(self.thirst, 0), 100)
        self.boredom = min(max(self.boredom, 0), 100)
        self.stamina = min(max(self.stamina, 0), 100)

        self.update_emotional_state()

    def update_emotional_state(self):
        # Simple logic to determine emotional state based on needs
        if self.hunger > 80 or self.thirst > 80:
            self.emotional_state = '😢'
        elif self.boredom > 80:
            self.emotional_state = '🥱'
        elif self.stamina < 20:
            self.emotional_state = '😴'
        else:
            self.emotional_state = '😊'

    def feed(self):
        self.hunger -= 20
        self.hunger = max(self.hunger, 0)

    def give_water(self):
        self.thirst -= 20
        self.thirst = max(self.thirst, 0)

    def play(self):
        self.boredom -= 20
        self.boredom = max(self.boredom, 0)
        self.stamina -= 10
        self.stamina = max(self.stamina, 0)

    def rest(self):
        self.stamina += 20
        self.stamina = min(self.stamina, 100)
