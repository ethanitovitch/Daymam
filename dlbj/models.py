from dlbj.agent import super
from dlbj import bj_env
import random
import time

class BlackJack:
    def __init__(self, decks=8):
        self.decks = decks

    def reset(self):
        self.cards_in_shoot = random.randint(332, 370)
        self.shoot = Shoot(0, self.cards_in_shoot)

        self.players = []

        for num in range(random.randint(0, 7)):
            self.players.append(bj_env.Player())
        self.num_hands_for_group = random.randint(1, 5)
        self.num_hands = 0

        self.user_total = 0
        self.user_money = 100
        self.dealer_total = 0


    def get_pre_state(self):
        return [item for sublist in [self.shoot.deck, [len(self.players), self.user_money]] for item in sublist]

    def get_state(self):
        return [item for sublist in [self.shoot.deck, [len(self.players), self.user_money, self.dealer_total, self.user_total]] for item in sublist]


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





def main():


    while shoot.has_cards_left() and user_money > 0:

        print("you have:", user_money)

        if num_hands_for_group == num_hands:

            players = []

            for num in range(random.randint(0, 7)):
                players.append(bj_env.Player())
            num_hands_for_group = random.randint(1, cards_in_shoot)
            num_hands = 0

        print("there are ", len(players))
        print("they will be seated for ", num_hands_for_group-num_hands)

        print("state", shoot.deck)

        time.sleep(0.3)
        bet = int(input("bet?"))
        while bet > user_money:
            bet = int(input("please input a valid bet?"))
        user_money -= int(bet)

        game = bj_env.Game(shoot, players)
        user_total += shoot.remove_card()
        game.deal_card()

        dealer_total += shoot.remove_card()
        user_total += shoot.remove_card()
        game.deal_card()
        print("state", shoot.deck)

        print("You have: " ,user_total)
        print("Dealer has: ", dealer_total)
        move = input("hit, stand, double")
        while move != "stand":
            user_total += shoot.remove_card()
            print("You have: ", user_total)
            print("Dealer has: ", dealer_total)
            if move == "double":
                user_money -= int(bet)
                bet *= 2
                break
            if user_total > 21:
                print("you have busted", user_total)
                user_total = 0
                break
            move = input("hit, stand")


        for player in game.players:
            print("new player")
            print("  ")
            print("Player has: " , player.total)
            initial = True
            move = player.make_move(dealer_total, initial)
            initial=False
            while move != "stand":
                player.total += shoot.remove_card()
                print("Player has: " , player.total)
                if move == "double":
                    print("Player doubled")
                    break
                if player.total > 21:
                    player.total = 0
                    break
                move = player.make_move(dealer_total, initial)

        print("You have: ", user_total)
        while dealer_total <= 17:
            dealer_total += shoot.remove_card()
            print("Dealer has: " , dealer_total)
            if dealer_total > 21:
                dealer_total = 1
                break

        if 22 > user_total > dealer_total:
            user_money += 2*bet
            print("YOU WON!", bet)
        elif user_total == dealer_total and user_total <= 21:
            print("TIE")
            user_money += bet
        else:
            print("LOOOOSER")

        user_total = 0
        dealer_total = 0
        for player in players:
            player.total = 0

        num_hands += 1


if __name__ == '__main__':
    main()