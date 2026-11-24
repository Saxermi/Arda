class Hero:
    def __init__(self):
        self.gold = 0
    def reward(self, amount):
        self.gold = self.gold + amount
hero = Hero()
i = 0
while i < 100000:
    hero.reward(3)
    i = i + 1
print(hero.gold)
