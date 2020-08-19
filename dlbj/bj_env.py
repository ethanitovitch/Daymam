import random

class Shoot():
    def __init__(self, cards_played, cards_total):
        self.cards_played = cards_played
        self.cards_total = cards_total
        self.deck = [32,32,32,32,32,32,32,32,32,128]

    def has_cards_left(self):
        return self.cards_played < self.cards_total

    def remove_card(self):
        card = random.randint(1,10)
        while self.deck[card-1] == 0:
            card = random.randint(0, 10)

        self.cards_played += 1
        self.deck[card-1] -= 1

        if card == 1:
            return 11
        else:
            return card

class Game():
    def __init__(self, shoot, players):
        self.shoot = shoot
        self.players = players

    def deal_card(self):
        for player in self.players:
            player.total += self.shoot.remove_card()


class Player():
    def __init__(self):
        self.total = 0

    def make_move(self, dealer_total, initial):
        if self.total < 9:
            move = "hit"
        elif self.total == 9:
            if dealer_total < 3 or dealer_total > 6:
                move = "hit"
            else:
                move = "double"
        elif self.total == 10:
            if dealer_total < 10:
                move = "double"
            else:
                move = "hit"
        elif self.total == 11:
            move = "double"
        elif self.total == 12:
            if dealer_total < 4 or dealer_total > 7:
                move = "hit"
            else:
                move = "stand"
        elif 12 < self.total < 17:
            if dealer_total < 6:
                move = "stand"
            else:
                move = "hit"
        else:
            move = "stand"

        if not initial and move == "double":
            move = "hit"
        return move



